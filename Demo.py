#Demo associated with Keyhole Imaging
#Chris Metzler 2020

#test change
#test branch
import argparse
import os
import numpy as np
import utils as utils
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
from torch.autograd import Variable
import torch.nn as nn
import torch
import h5py

os.makedirs("reconstructions", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--padded_size_recon", type=int, default=256, help="size of each image dimension used for forward model")
parser.add_argument("--reconstruction", type=str, default='Mannequin_Assymetric', help="Dataset to reconstruct")
parser.add_argument("--extra_suffix", type=str, default='', help="Suffix to be appended to the end of the filename")
parser.add_argument("--closeup", type=bool, default=True, help="Reconstruct only 60cm x 60cm")
parser.add_argument("--init_var_scalar", type=float, default=1, help="How much to scale the initial variance")
opt = parser.parse_args()
print(opt)

# cuda = True if torch.cuda.is_available() else False
cuda=False#Requires nvidia GPU. Works with 10GB of ram. Does not work with 6GB of ram.

n_t=768#Number of time bins used
bin_resolution_t=16e-12#Size of each time bin, seconds

EM_iters = 30
M_iters = (2+np.arange(0,1*EM_iters,1)).tolist()#Don't optimize for very long at the early iterations where you have a poor estimate of the object's position.
optimizer_reset_rate=1000#Resets at minimum once every M_iters
n_restarts = 1
Beta_rate=1.3

reconstruction=opt.reconstruction

#Reconstruction parameters/priors
TV_weight = 0  # 1e3
LaplacianL1_weight = 2e3
L1_weight = 2e3
HessianL1_weight = 0
sigma = 200
object_y_offset = .1 #During reconstruction, how far to place the object grid above the floor
object_scale = 1  # How many meters tall and wide is the object. Overwritten by closeup
background_sub=True

x_range = 1  #How many meters can the object move along the x-axis (horizontal)
x_offset = -x_range / 2
z_offset = .64  # At it's closest point, how far away is the object
y_offset = 1.13  # Laser's height above the floor
clip_until = 260#Ignore any counts that occur before this time bin
if reconstruction=='E':
    with h5py.File('captured_data/E/scan_10-14-19_17-37.mat', 'r') as f:
        measurements_np = np.array(f['data'])
        xpos = np.array(f['xpos'])
        zpos = np.array(f['zpos'])
    with h5py.File('captured_data/E/direct_after.mat', 'r') as f:
        direct_np = np.array(f['direct_after'])#Direct measurement of wall, used to estimate peak_center
    with h5py.File('captured_data/E/LongExpNoObject_after.mat', 'r') as f:
        NoObjectMeasurement = np.array(f['LongExpNoObject_after'])
    peak_center = 630
elif reconstruction == 'K':
    with h5py.File('captured_data/K/scan_10-15-19_21-12.mat', 'r') as f:
        measurements_np = np.array(f['data'])
        xpos = np.array(f['xpos'])
        zpos = np.array(f['zpos'])
    with h5py.File('captured_data/K/direct_after.mat', 'r') as f:
        direct_np = np.array(f['direct_after'])#Direct measurement of wall, used to estimate peak_center
    with h5py.File('captured_data/K/LongExpNoObject_after.mat', 'r') as f:
        NoObjectMeasurement = np.array(f['LongExpNoObject_after'])
    peak_center = 635
elif reconstruction == 'Y':
    with h5py.File('captured_data/Y/scan_10-15-19_21-57.mat', 'r') as f:
        measurements_np = np.array(f['data'])
        xpos = np.array(f['xpos'])
        zpos = np.array(f['zpos'])
    with h5py.File('captured_data/Y/direct_after.mat', 'r') as f:
        direct_np = np.array(f['direct_after'])#Direct measurement of wall, used to estimate peak_center
    with h5py.File('captured_data/Y/LongExpNoObject_after.mat', 'r') as f:
        NoObjectMeasurement = np.array(f['LongExpNoObject_after'])
    peak_center = 636
elif reconstruction=='Mannequin':
    with h5py.File('captured_data/Mannequin/scan_10-30-19_12-55.mat', 'r') as f:
        measurements_np = np.array(f['data'])
        xpos = np.array(f['xpos'])
        zpos = np.array(f['zpos'])
    with h5py.File('captured_data/Mannequin/direct_after.mat', 'r') as f:
        direct_np = np.array(f['direct_after'])#Direct measurement of wall, used to estimate peak_center
    with h5py.File('captured_data/Mannequin/LongExpNoObject_after.mat', 'r') as f:
        NoObjectMeasurement = np.array(f['LongExpNoObject_after'])
    peak_center = 631
elif reconstruction=='Mannequin_Assymetric':
    with h5py.File('captured_data/Mannequin_Assymetric/scan_10-30-19_16-28.mat', 'r') as f:
        measurements_np = np.array(f['data'])
        xpos = np.array(f['xpos'])
        zpos = np.array(f['zpos'])
    with h5py.File('captured_data/Mannequin_Assymetric/direct_after.mat', 'r') as f:
        direct_np = np.array(f['direct_after'])#Direct measurement of wall, used to estimate peak_center
    with h5py.File('captured_data/Mannequin_Assymetric/LongExpNoObject_after.mat', 'r') as f:
        NoObjectMeasurement = np.array(f['LongExpNoObject_after'])
    peak_center = 630
else:
    raise Exception("unrecognized reconstruction")

measurements_np = measurements_np[:, 0:4096]
if background_sub:
    NoObjectMeasurement = NoObjectMeasurement[0:4096]
    for i in range(measurements_np.shape[0]):
            measurements_np[i, :] = measurements_np[i, :] - NoObjectMeasurement.squeeze()
measurements_np=np.expand_dims(measurements_np, axis=1)

if opt.closeup:
    opt.extra_suffix=opt.extra_suffix + 'Closeup'
    object_y_offset =.5
    object_scale = .6

display_some=True#Display only important details
display=False#Display most details
display_many=False#Display all details
plot_trajectory=True

measurements_np=measurements_np[:,:,0:2048]

direct_np=direct_np[0:2048]
if display_some:
    plt.plot(direct_np[peak_center-4:peak_center+5])
    plt.show()

measurements_np=measurements_np[:,:,peak_center:peak_center+n_t]

measurements_np[:,:,0:clip_until]=0
measurements_np=np.reshape(measurements_np,(-1,measurements_np.shape[-1]))

if display:
    plt.plot(np.transpose(measurements_np[:,clip_until:]))
    plt.show()
    plt.plot(np.transpose(measurements_np[:,:]))
    plt.show()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


j_fixed=True

sampling_coordinates = []
i_max=33
j_max=20.
k_max=33

y_range=0#How many meters can the object move along the y-axis (vertical)
z_range=.15#How many meters can the object move along the z-axis (depth)
for k in range(int(k_max)):
    if j_fixed:
        js=[int(np.floor(j_max/2))]
    else:
        js=range(int(j_max))
    for j in js:
        for i in range(int(i_max)):
            x_samp = x_offset+x_range * i / (i_max-1)
            y_samp = y_offset+y_range * j / (j_max-1)
            z_samp = z_range * k / (k_max-1)
            sampling_coordinates += [[x_samp, y_samp, z_samp]]


voxel_coordinates=[]
for j in range(opt.padded_size_recon):#Outer indexes must be y (vertical) because that is how images are stored
    for i in range(opt.padded_size_recon):
        x_loc=-object_scale/2 + i/(opt.padded_size_recon-1)*object_scale#put the center of the object at 0
        y_loc=object_y_offset + object_scale - j/(opt.padded_size_recon-1)*object_scale#Largest y should go first, images are indexed from the top down
        z_loc=z_offset+z_range
        voxel_coordinates+=[[x_loc,y_loc,z_loc]]

sampling_coordinates_np=np.array(sampling_coordinates)
voxel_coordinates_np=np.array(voxel_coordinates)

if display:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sampling_coordinates_np[:, 0], sampling_coordinates_np[:, 2], sampling_coordinates_np[:, 1], c='r', marker='o')
    ax.scatter(voxel_coordinates_np[:, 0], voxel_coordinates_np[:, 2], voxel_coordinates_np[:, 1], c='b', marker='o')
    ax.set_xlabel('Width')
    ax.set_ylabel('Depth')
    ax.set_zlabel('Height')
    plt.show()


measurements = Tensor(measurements_np)
measurements[:,0]=0#My forward model discards the first index


measurements_np=measurements.cpu().data.numpy()
n_samples = measurements.shape[0]

if display_some:
    plt.plot(np.transpose(measurements_np[:,:]))
    plt.show()


scale_factor_recon=int(opt.padded_size_recon/opt.img_size)
upsampler=torch.nn.Upsample(scale_factor=scale_factor_recon,mode='nearest')

## Reconstruct using EM algorithm
def Q_loss(new_recon, sigma, old_recon=None, W_in=None, Beta=1):
    if W_in is not None:
        W = W_in.clone()
    else:
        assert old_recon is not None, "Either old_recon or W_in must be provided"
        W = torch.zeros(n_samples, len(sampling_coordinates))
        if cuda:
            W = W.cuda()
    Q = 0
    if W_in is None:
        with torch.no_grad():
            f_x_old_allthetas = utils.Sample_Lambertian(voxel_values=old_recon,
                                                          sampling_coordinates=sampling_coordinates,
                                                          voxel_coordinates=voxel_coordinates, n_t=n_t,
                                                          bin_resolution_t=bin_resolution_t, cuda=cuda)
            if cuda:
                f_x_old_allthetas = f_x_old_allthetas.cuda()
    f_x_new_allthetas = utils.Sample_Lambertian(voxel_values=new_recon, sampling_coordinates=sampling_coordinates,
                                                  voxel_coordinates=voxel_coordinates, n_t=n_t,
                                                  bin_resolution_t=bin_resolution_t, cuda=cuda)
    if cuda:
        f_x_new_allthetas = f_x_new_allthetas.cuda()
    for l in range(n_samples):  # Sum over all the measurements
        measurement_l = measurements[l, :]
        if W_in is None:
            with torch.no_grad():
                tmp = -torch.sum((measurement_l - f_x_old_allthetas) ** 2, dim=1)
                Soft = nn.Softmax(dim=0)
                w = Soft(tmp / (2 * sigma ** 2) * Beta)
                W[l, :] = w.clone()
        Q = Q + torch.sum(W[l, :] * (torch.sum((measurement_l - f_x_new_allthetas) ** 2, dim=1)))  # Note the sign is flipped compared withthe minorizing loss. This is the slowest part of my code.
    if W_in is None and display == True:
        plt.imshow(W.cpu().data.numpy())
        plt.show()
        if plot_trajectory:
            W_np = W.cpu().data.numpy()
            Trajectory = np.zeros((33, 33))
            for i in range(W_np.shape[0]):
                Trajectory += W_np[i].reshape((33, 33))
            plt.imshow(Trajectory)
            plt.show()
    return Q, W


Q_best = np.inf
for attempt in range(n_restarts):
    if cuda:
        recons_var = Variable( np.sqrt(opt.init_var_scalar)*torch.randn([opt.img_size, opt.img_size]).cuda(),requires_grad=True)  # Init has implicitly had its square root taken
    else:
        recons_var = Variable( np.sqrt(opt.init_var_scalar)*torch.randn([opt.img_size, opt.img_size]),requires_grad=True)

    Beta = 1 / (Beta_rate ** EM_iters)
    t000 = time.time()
    for E_iter in range(EM_iters):
        recons = torch.squeeze(
            upsampler(recons_var.view(1, 1, opt.img_size, opt.img_size) ** 2))  # Enforces non-negativity on recons
        # E Step
        # Create a function that majorizes the negative log likelihood
        old_recon = recons.clone()
        sigma_this_iter = sigma
        print("Sigma this iter is " + str(sigma_this_iter))
        warm_start=False#Use oracle object positions as initial trajecotry
        # M Step
        if E_iter==0 and warm_start:
            W_init = torch.zeros(n_samples, len(sampling_coordinates))
            if cuda:
                W_init = W_init.cuda()
            for i in range(len(xpos)):
                x_samp = xpos[i][0] - .5
                y_samp = y_offset
                z_samp = zpos[i][0]
                true_samp = [x_samp, y_samp, z_samp]
                for j in range(len(sampling_coordinates)):
                    samp_coord=sampling_coordinates[j]
                    if np.linalg.norm(np.array(true_samp)-np.array(samp_coord))<.01:
                        W_init[i,j]=1
            [Q, W] = Q_loss(recons, old_recon=old_recon, sigma=sigma_this_iter, Beta=Beta,W_in=W_init)
        else:
            [Q, W] = Q_loss(recons, old_recon=old_recon, sigma=sigma_this_iter, Beta=Beta)  # Changing sigma over time is equivalent to deterministic annealing https://papers.nips.cc/paper/941-deterministic-annealing-variant-of-the-em-algorithm.pdf
        Beta = Beta * Beta_rate
        if display:
            tmp1 = np.squeeze(recons[:, :].cpu().data.numpy())
            plt.imshow(tmp1)
            plt.title("Current Recon")
            plt.show()
            f_pred = utils.Sample_Lambertian(voxel_values=recons, sampling_coordinates=sampling_coordinates,
                                               voxel_coordinates=voxel_coordinates, n_t=n_t,
                                               bin_resolution_t=bin_resolution_t, cuda=cuda)
        t00 = time.time()
        for M_iter in range(M_iters[E_iter]):
            if M_iter % optimizer_reset_rate == 0:  # Every optimizer_reset_the optimizer: resets momentum terms.
                optimizer = torch.optim.Adam([recons_var], lr=opt.lr, betas=(opt.b1, opt.b2))  # Decay learning rate over the EM iterations
            optimizer.zero_grad()
            recons = torch.squeeze(
                upsampler(recons_var.view(1, 1, opt.img_size, opt.img_size) ** 2))  # Enforces non-negativity on recons
            [Q, _] = Q_loss(recons, sigma=sigma_this_iter, W_in=W)  # Need to increase sigma to avoid all probabilities being 0
            loss = Q
            TV_loss = TV_weight * utils.TotalVariation(recons)
            L1_loss = L1_weight * recons.abs().sum()
            [L, H] = utils.Laplacian_and_Hessian(recons)
            LaplacianL1_loss = LaplacianL1_weight * L.abs().sum()
            HessianL1_loss = HessianL1_weight * H.abs().sum()
            loss = loss + TV_loss + L1_loss + LaplacianL1_loss + HessianL1_loss
            loss.backward()
            optimizer.step()
            if M_iter % 10 == 0:
                print("E iteration %d of %d, M iteration %d of %d" % (E_iter + 1, EM_iters, M_iter, M_iters[E_iter]))
                print("Q loss is ", Q.cpu().data.numpy())
            if display_many:
                tmp1 = np.squeeze(recons[:, :].cpu().data.numpy())
                plt.imshow(tmp1)
                plt.show()
        print("%d M Iterations took %f seconds" % (M_iters[E_iter], time.time() - t00))
    if Q.cpu().data.numpy() < Q_best:
        Q_best = Q.cpu().data.numpy()
        recon_best = recons.cpu().data.numpy()
    complete_time = time.time() - t000
    print("Total time: ", str(complete_time))
ReconstructedObject = np.squeeze(recon_best)
if display_some:
    plt.imshow(np.fliplr(ReconstructedObject),cmap='inferno')
    plt.title("Best Recon")
    plt.show()


#Based on W at the last attempt, visualize the best guess trajectory of the object
if plot_trajectory:
    W_np=W.cpu().data.numpy()
    # Display trajectory as sum of pdfs
    Trajectory = np.zeros((33, 33))
    for i in range(W_np.shape[0]):
        Trajectory += W_np[i].reshape((33, 33))
    Trajectory[Trajectory > 1] = 1
    if display_some:
        plt.imshow((Trajectory))
        plt.axis('off')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20)
        plt.show()
    # Display trajectory as sum of one-hot vectors
    Trajectory=np.zeros((33,33))
    for i in range(W_np.shape[0]):
        tmp=np.zeros((33*33,))
        tmp[np.argmax(W_np[i])]=1
        tmp=np.reshape(tmp,(33,33))
        Trajectory=Trajectory+tmp
    Trajectory[Trajectory>1]=1
    if display_some:
        plt.imshow((Trajectory))
        plt.axis('off')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20)
        plt.show()


#Save numpy arrays with the reconstruction and the trajectory
Save_dir = 'reconstructions'
suffix = ''
if TV_weight is not 0:
    suffix = suffix + '_TV' + str(TV_weight)
if LaplacianL1_weight is not 0:
    suffix = suffix + '_LapL1' + str(LaplacianL1_weight)
if L1_weight is not 0:
    suffix = suffix + '_L1' + str(L1_weight)
if HessianL1_weight is not 0:
    suffix = suffix + '_MyHess' + str(HessianL1_weight)

suffix = suffix + '_n' + str(opt.padded_size_recon) + '_sigma' + str(sigma)
savename = './' + Save_dir + '/'+reconstruction + '_UnknownLocation' + suffix + opt.extra_suffix +'.npz'
if plot_trajectory:
    np.savez(savename, ReconstructedObject=ReconstructedObject, Trajectory=Trajectory,W_np=W_np,xpos=xpos,zpos=zpos)#W_np records the trajectory in order
else:
    np.savez(savename, ReconstructedObject=ReconstructedObject,W_np=W_np,xpos=xpos,zpos=zpos)

#Plot reconstructions and trajectories
def LoadPlotandSave(filename,unknown=False,flip=False,legend=False):
    a=np.load(filename)
    newfilename=filename
    recon=a['ReconstructedObject']
    recon=recon[4:251,4:251]
    recon_sorted=recon.copy().flatten()
    recon_sorted.sort()
    recon[recon>recon_sorted[-100]]=recon_sorted[-100]
    recon=np.fliplr(recon)
    cmaps=['inferno']
    i=0
    plt.imshow(recon**.5,cmap=cmaps[i])
    plt.show()
    imsave(newfilename[:-4]+'_Object.png', recon,cmap=cmaps[i])

    if unknown:
        Trajectory=a['Trajectory']
        print('Test')
        W_np=a['W_np']
        xpos=a['xpos']
        zpos=a['zpos']
        W_np_onehot = np.zeros((W_np.shape[0],33*33))
        for i in range(W_np.shape[0]):
            W_np_onehot[i,np.argmax(W_np[i])]=1
        W_np_onehot=np.reshape(W_np_onehot,(W_np.shape[0],33,33))

        W_np_zs=np.array([np.nonzero(W_np_onehot[i])[0] for i in range(W_np.shape[0])])
        W_np_xs = np.array([np.nonzero(W_np_onehot[i])[1] for i in range(W_np.shape[0])])
        if flip:
            W_np_xs = 32-W_np_xs
        import matplotlib.collections as collections

        fig = plt.figure(num=None, figsize=(4, 4), dpi=200, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        gt_call = plt.scatter(-zpos,xpos,s=10,c=(0,0,0))
        for i in range(W_np.shape[0]):
            min_alpha=.2
            max_alpha=.8
            alpha=min_alpha + (max_alpha-min_alpha)*i/W_np.shape[0]
            c = ((1 - i / W_np.shape[0]), 0, (i / W_np.shape[0]),alpha)
            if i==0:
                init_call = ax.scatter(-W_np_zs[i] / 32. * .15, W_np_xs[i] / 32., s=100, c=c)#, alpha=alpha)
            else:
                final_call = ax.scatter(-W_np_zs[i] / 32. * .15, W_np_xs[i] / 32., s=100, c=c)#, alpha=alpha)
        alpha_start=min_alpha
        alpha_mid=min_alpha + (max_alpha-min_alpha)*np.floor(W_np.shape[0]/2)/W_np.shape[0]
        alpha_end=min_alpha + (max_alpha-min_alpha)*(W_np.shape[0]-1)/W_np.shape[0]
        c_start=((1-0/W_np.shape[0]),0,(0/W_np.shape[0]),alpha_start)
        c_mid=((1-np.floor(W_np.shape[0]/2)/W_np.shape[0]),0,np.floor(W_np.shape[0]/2)/W_np.shape[0],alpha_mid)
        c_end=((1-(W_np.shape[0]-1)/W_np.shape[0]),0,((W_np.shape[0]-1)/W_np.shape[0]),alpha_end)
        circles = collections.CircleCollection(sizes=[100, 100, 100], facecolors=[c_start, c_mid, c_end])
        circles.set_alpha(.5)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        plt.savefig(newfilename[:-4] + '_Trajectory.png', bbox_inches="tight")
        if legend:
            leg = ax.legend((gt_call, circles), ('Truth', 'Estimate'), fontsize='xx-large',scatterpoints=3, scatteryoffsets=[.5], handlelength=2)
            plt.savefig(newfilename[:-4] + '_Trajectory_wLegend.png', bbox_inches="tight")
        plt.show()

flip=False#Use for display only
LoadPlotandSave(savename,unknown=True,flip=flip,legend=False)

print("FINISHED RECONSTRUCTION")


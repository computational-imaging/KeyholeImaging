#Chris Metzler 2020
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist
import torch

c = 3e8
bin_resolution_t = 16e-12

# cuda = True if torch.cuda.is_available() else False
cuda=False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def TotalVariation(img):
    tv_h = ((img[1:,:] - img[:-1,:]).abs()).sum()
    tv_w = ((img[:,1:] - img[:,:-1]).abs()).sum()
    return (tv_h + tv_w)

def Laplacian_and_Hessian(img):
    if img.is_cuda:
        laplacian_filter = torch.tensor([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]]).cuda()
    else:
        laplacian_filter = torch.tensor([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
    L = torch.conv2d(img.reshape(1,1,img.shape[0],img.shape[1]),laplacian_filter.reshape(1,1,laplacian_filter.shape[0],laplacian_filter.shape[1]))
    H = torch.conv2d(L,laplacian_filter.reshape(1,1,laplacian_filter.shape[0],laplacian_filter.shape[1]))
    return L,H

def Sample_Lambertian(voxel_values=np.ones((32**2,1)),sampling_coordinates=[[0,0,-2],[1,0,-2],[0,2,0]],voxel_coordinates=[],n_t=512,bin_resolution_t=bin_resolution_t,cuda=cuda,jitter=False,filter=False):
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    bin_resolution_d = bin_resolution_t*c

    if not voxel_coordinates:
        #Use default grid
        for j in range(voxel_values.shape[0]):
            for i in range(voxel_values.shape[1]):
                x_loc=i/(voxel_values.shape[0]-1+.00000000001)
                y_loc=1.-j/(voxel_values.shape[1]-1+.00000000001)#Largest y should go first, images are indexed from the top down
                z_loc=0
                voxel_coordinates+=[[x_loc,y_loc,z_loc]]
    if cuda:
        voxel_values = voxel_values.reshape(-1,).cuda()
    else:
        voxel_values=voxel_values.reshape(-1,).cpu()

    n_s=len(sampling_coordinates)
    if cuda:
        measurement=torch.zeros((n_s,n_t)).type(torch.cuda.FloatTensor)
    else:
        measurement = torch.zeros((n_s, n_t))

    sampling_coordinates=Tensor(np.array(sampling_coordinates))
    voxel_coordinates=Tensor(np.array(voxel_coordinates))

    A=torch.abs(sampling_coordinates[:,2][:,None]-(voxel_coordinates[:,2][None,:]))#[:,None] and [None,:] perform broadcasting
    C=torch.cdist(sampling_coordinates,voxel_coordinates)

    cos_theta=A/C
    LambDropoff = cos_theta**4

    drop_off_r=1./C**4

    Y = C*2#Need to account for path there and back

    if cuda:
        time_coords = (torch.ceil(Y / bin_resolution_d)).type(torch.cuda.LongTensor)
    else:
        time_coords=(torch.ceil(Y/bin_resolution_d)).type(torch.LongTensor)


    if jitter:
        if cuda:
            time_coords=time_coords+(2*torch.rand(size=time_coords.shape)-1).type(torch.LongTensor).cuda()
        else:
            time_coords = time_coords + (2 * torch.rand(size=time_coords.shape) - 1).type(torch.LongTensor)

    time_coords[time_coords>=measurement.shape[1]]=0#Write all distances that are too far to the first index


    # start = time.time()
    for ind_samp in range(len(sampling_coordinates)):
        measurement[ind_samp].put_(time_coords[ind_samp], voxel_values*LambDropoff[ind_samp,:]*drop_off_r[ind_samp,:], accumulate=True)


    if filter:
        kernel_size=3
        myfilter = torch.nn.Conv1d(in_channels=n_s, out_channels=n_s, kernel_size=3, stride=1, padding=0, dilation=0, groups=n_s, bias=False, padding_mode='zeros')
        if cuda:
            myfilter.weight.data = torch.zeros(size=(n_s,1,kernel_size)).cuda()#Should be out_channel x in_channels/groups x kernel_size
        else:
            myfilter.weight.data = torch.zeros(size=(n_s, 1, kernel_size))
        myfilter.weight.data[:,0,0]=1/3
        myfilter.weight.data[:, 0, 1] = 1 / 3
        myfilter.weight.data[:, 0, 2] = 1 / 3
        myfilter.weight.requires_grad = False
        measurement=myfilter(measurement.unsqueeze(0)).squeeze()

    measurement[:,0]=0
    return measurement

def Sample_Retroreflective(voxel_values=np.ones((32**2,1)),sampling_coordinates=[[0,0,-2],[1,0,-2],[0,2,0]],voxel_coordinates=[],n_t=512,bin_resolution_t=bin_resolution_t,cuda=cuda,jitter=False,filter=False):
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    bin_resolution_d = bin_resolution_t*c

    if not voxel_coordinates:
        #Use default grid
        for j in range(voxel_values.shape[0]):
            for i in range(voxel_values.shape[1]):
                x_loc=i/(voxel_values.shape[0]-1+.00000000001)
                y_loc=1.-j/(voxel_values.shape[1]-1+.00000000001)#Largest y should go first, images are indexed from the top down
                z_loc=0
                voxel_coordinates+=[[x_loc,y_loc,z_loc]]
    if cuda:
        voxel_values = voxel_values.reshape(-1,).cuda()
    else:
        voxel_values=voxel_values.reshape(-1,).cpu()

    n_s=len(sampling_coordinates)
    if cuda:
        measurement=torch.zeros((n_s,n_t)).type(torch.cuda.FloatTensor)
    else:
        measurement = torch.zeros((n_s, n_t))

    sampling_coordinates=Tensor(np.array(sampling_coordinates))
    voxel_coordinates=Tensor(np.array(voxel_coordinates))

    A=torch.abs(sampling_coordinates[:,2][:,None]-(voxel_coordinates[:,2][None,:]))#[:,None] and [None,:] perform broadcasting
    C=torch.cdist(sampling_coordinates,voxel_coordinates)

    cos_theta=A/C
    LambDropoff = cos_theta**2#4 # Lets model > lambertian drop-off

    drop_off_r=1./C**2


    Y = C*2#Need to account for path there and back

    if cuda:
        time_coords = (torch.ceil(Y / bin_resolution_d)).type(torch.cuda.LongTensor)
    else:
        time_coords=(torch.ceil(Y/bin_resolution_d)).type(torch.LongTensor)


    if jitter:
        if cuda:
            time_coords=time_coords+(2*torch.rand(size=time_coords.shape)-1).type(torch.LongTensor).cuda()
        else:
            time_coords = time_coords + (2 * torch.rand(size=time_coords.shape) - 1).type(torch.LongTensor)

    time_coords[time_coords>=measurement.shape[1]]=0#Write all distances that are too far to the first index

    for ind_samp in range(len(sampling_coordinates)):
        measurement[ind_samp].put_(time_coords[ind_samp], voxel_values*LambDropoff[ind_samp,:]*drop_off_r[ind_samp,:], accumulate=True)


    if filter:
        kernel_size=3
        myfilter = torch.nn.Conv1d(in_channels=n_s, out_channels=n_s, kernel_size=3, stride=1, padding=0, dilation=0, groups=n_s, bias=False, padding_mode='zeros')
        if cuda:
            myfilter.weight.data = torch.zeros(size=(n_s,1,kernel_size)).cuda()#Should be out_channel x in_channels/groups x kernel_size
        else:
            myfilter.weight.data = torch.zeros(size=(n_s, 1, kernel_size))
        myfilter.weight.data[:,0,0]=1/3
        myfilter.weight.data[:, 0, 1] = 1 / 3
        myfilter.weight.data[:, 0, 2] = 1 / 3
        myfilter.weight.requires_grad = False
        measurement=myfilter(measurement.unsqueeze(0)).squeeze()


    measurement[:,0]=0#Discard the first index
    return measurement


if __name__ == "__main__":
    #Target is by default places along a 32x32, 1mx1m grid at the origin
    voxel_values=np.zeros((32,32))
    voxel_values[0,:]=1
    voxel_values[10,:]=1
    voxel_values[:,0]=1
    voxel_values=np.ones((32,32))

    voxel_coordinates=[]
    for j in range(32):
        for i in range(32):
            x_loc = i / (32-1)
            y_loc = 1 - j / (32-1)#Largest y should go first, images are indexed from the top down
            z_loc = 0
            voxel_coordinates += [[x_loc, y_loc, z_loc]]

    #sampling_coordinates should trace out a path along a 2m x 2m wall that is 2m from the target
    sampling_coordinates=[]
    for i in range(32):
        for j in range(32):
            x_samp=-1+2*i/(32.-1)
            y_samp=-1+2*j/(32.-1)
            z_samp=-2.
            sampling_coordinates+=[[x_samp,y_samp,z_samp]]


    voxel_values=Tensor(voxel_values)
    start = time.time()
    measurement_torch=Sample_Lambertian(voxel_values,sampling_coordinates,voxel_coordinates=voxel_coordinates)
    end = time.time()
    print("torch overall: ", end-start)

    measurement_torch=measurement_torch.cpu().data.numpy()

    plt.plot(np.squeeze(measurement_torch[0,:]))
    plt.show()
    1
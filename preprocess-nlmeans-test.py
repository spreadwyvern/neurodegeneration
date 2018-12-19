
# coding: utf-8

# In[2]:


import os

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dki as dki
import dipy.reconst.dti as dti
import dipy.reconst.dki_micro as dki_micro
from dipy.data import fetch_cfin_multib
from dipy.data import read_cfin_dwi
from dipy.segment.mask import median_otsu
from scipy.ndimage.filters import gaussian_filter
from dipy.denoise.noise_estimate import estimate_sigma
from time import time
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.adaptive_soft_matching import adaptive_soft_matching


# In[2]:


complete = []
for sub_id in os.listdir('/home/lwk/MRI/testing/dti_nlmeans/FA'):
    complete.append(sub_id)


# In[3]:


for sub_id in os.listdir('/home/lwk/MRI/img/Test'):
    if 'sub' in sub_id: 
        if sub_id not in complete: 
            if not os.path.exists(os.path.join('/home/lwk/MRI/testing/test_subjects', sub_id + '.npy')):
                fdwi = '/home/lwk/MRI/img/Test/{sub_id}/anatomy/{sub_id}_DSI.nii'.format(sub_id = sub_id)
                fbval = '/home/lwk/MRI/img/Test/{sub_id}/anatomy/{sub_id}_DSI.bval'.format(sub_id = sub_id)
                fbvec = '/home/lwk/MRI/img/Test/{sub_id}/anatomy/{sub_id}_DSI.bvec'.format(sub_id = sub_id)

                data, affine = load_nifti(fdwi)
                bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
                gtab = gradient_table(bvals, bvecs)

            # denoise with nlmeans
                sigma = estimate_sigma(data, N=4)
                mask = data[..., 0] > 80
                data_smooth = nlmeans(data, sigma=sigma, mask=mask, patch_radius= 1, block_radius = 1, rician= True)    

#             # denoise with gaussian filter
#                 fwhm = 1.25
#                 gauss_std = fwhm / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std
#                 data_smooth_g = np.zeros(data.shape)
#                 for v in range(data.shape[-1]):
#                     data_smooth_g[..., v] = gaussian_filter(data[..., v], sigma=gauss_std)

            # reconstruct DTI
                tenmodel = dti.TensorModel(gtab)
                tenfit = tenmodel.fit(data_smooth, mask=mask)
                dti_FA = tenfit.fa
                dti_MD = tenfit.md
            #     dti_AD = tenfit.ad
            #     dti_RD = tenfit.rd
            #     dti_FA_C = tenfit.color_fa

            # reconstruct DKI 
            #     dkimodel = dki.DiffusionKurtosisModel(gtab)
            #     dkifit = dkimodel.fit(data_smooth, mask=mask)
            #     FA = dkifit.fa
            #     MD = dkifit.md
            #     AD = dkifit.ad
            #     RD = dkifit.rd
            #     FA_C = dkifit.color_fa
            #     MK = dkifit.mk(0, 3)
            #     AK = dkifit.ak(0, 3)
            #     RK = dkifit.rk(0, 3)

            #     raw_data = data

                np.save(os.path.join('/home/lwk/MRI/testing/dti_nlmeans/FA_nl4', sub_id + '.npy'), dti_FA)
                np.save(os.path.join('/home/lwk/MRI/testing/dti_nlmeans/MD_nl4', sub_id + '.npy'), dti_MD)

                print(sub_id, " done")



# # Show example images

# In[17]:


import random
group_of_list = os.listdir('/home/lwk/MRI/testing/dti_nlmeans/FA')
num_to_select = 1
random_subject = random.sample(group_of_list, num_to_select)

for img_id in random_subject:
    FA = np.load( '/data/put_data/lwk/MRI/nlmeans/FA_DTI/sub-CC110069.npy')
    FA_32 = np.load( '/data/put_data/lwk/MRI/nlmeans_32/FA_DTI/sub-CC110069.npy')
    MD = np.load( '/home/lwk/MRI/testing/dti_nlmeans/MD_nl4/' + img_id)

    axial_slice = 36

    fig1, ax = plt.subplots(1, 3, figsize=(30, 20),
                            subplot_kw={'xticks': [], 'yticks': []})

    fig1.subplots_adjust(hspace=0.3, wspace=0.05)

    ax.flat[0].imshow(FA[:, :, axial_slice].transpose(), cmap='gray', vmin=0, vmax=1)
    ax.flat[0].set_title('FA')
    ax.flat[1].imshow(MD[:, :, axial_slice].transpose(), cmap='gray', vmin=0, vmax=2.0e-3)
    ax.flat[1].set_title('MD')
    ax.flat[2].imshow(FA_32[:, :, axial_slice].transpose(), cmap='gray', vmin=0, vmax=1)
    ax.flat[2].set_title('FA_32')
    plt.show()


## A 3D deep learning approach to predict metal–organic framework guest accessibility

Here, we introduce a method for predicting the guest accessibility of the MOFs using a 3D deep learning approach. We construct 3D voxels based on biophysical properties derived from the raw atom coordinates of metal-linker complex. We then fed these 3D voxels into a 3D convolutional neural network (CNN) model. Using a comprehensive dataset of MOF structures acquired from two sources, we train and test the model. Our proposed model predicts the guest accessibility of MOF with high accuracy (R2 = 0.86). Our results demonstrate that 3D-CNNs can be used to predict MOF structural characteristics based on their 3D biophysical aspects.

![mof_lpd_pipeline](https://github.com/ClinicalAI/MOF_LPD_Prediction/blob/main/MOF_LPD.png)

### Data 
We downloaded the CIF structure of 25,529 MOF from CoREMOF-2019 and Tobacco databases (downloaded in May 2022). We processed the CIF structures to extract MOFid to find the metal-ligand building blocks. We removed the redundant structures and those with more than one metal element in their inorganic core. 

### Generate 3D structure of the metal-ligand complex 
Then for each metal-ligand complex, we generated the 3D structures using molSimplify. The 3D structures have been saved in XYZ format.
The script can be found in [molsimplify_generate_xyz.py](https://github.com/ClinicalAI/MOF_LPD_Prediction/blob/main/molsimplify_generate_xyz.py)


### Generate 3D voxels from 3D metal-ligand complex 
We applied voxelization to the generated 3D structures of the metal-linkers. Also, we removed those metal cores with less than 100 voxels from our 3D voxel dataset. Finally, 3,621 voxels were created for the Zn, Cu, Mn, Cd, Co, Ag, Ni, Fe, and La metal cores.
The script can be found in [create_voxel.py](https://github.com/ClinicalAI/MOF_LPD_Prediction/blob/main/create_voxel.py)


### Train the model
We used the voxels to train a 3D CNN model. 
The script can be found in [model.py](https://github.com/ClinicalAI/MOF_LPD_Prediction/blob/main/model.py)


### Pre-trained model weight
The trained model weights can be found in the [models](https://github.com/ClinicalAI/MOF_LPD_Prediction/tree/main/models) folder.

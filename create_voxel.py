from multiprocessing import Process, Pool

from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors

import numpy as np
import htmd


dirs = np.loadtxt("all_xyz.txt",delimiter='\n',dtype='str')

def voxel_creation(pdb):
    metal_name = ""

    try:
        pdb_id = pdb.split("Runs/")[-1].split("/")[0]
        mol = SmallMol(pdb,force_reading=True)
        coo = mol.get('coords')
        center = np.mean(coo, axis=0)
        print(center)
        mol_vox, mol_centers, mol_N = getVoxelDescriptors(mol,boxsize=[50,50,50],voxelsize=1, center=center[0])
        mol_vox_t = mol_vox.transpose(0,1).reshape([50, 50, 50,8])

        np.savez_compressed(f'voxels_directory/voxels_50/{pdb_id}.npz',X=mol_vox_t)

    except Exception as e:
      print(e)


if __name__ == '__main__':
  pool = Pool(processes=30)
  pool.map_async(voxel_creation,dirs)
  pool.close()
  pool.join()

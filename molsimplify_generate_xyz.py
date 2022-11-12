import glob
import json
import subprocess
from multiprocessing import Process, Pool
import pandas as pd


input_list = glob.glob("mof_directory/*.json")
metal_charges = pd.read_csv("metal_charge.csv")

def generate_xyz(inpt):
    dir = 'directory_for_molsimplify/Runs/' + inpt.split('/')[-1].split('.json')[0]
    f = open (inpt, "r")

    valid_repeat = 0
    # Reading from file
    data = json.loads(f.read())
    ligs = data['mofid'].split(' ')[0].split('.')
    mofkey = data['mofkey'].split('.')[0]
    for lig in ligs:
        if len(lig) < 5:
            lig = lig.replace("[","")
            lig = lig.replace("]","")

            if  lig in mofkey:metal = lig


    ligands = []
    try:
        charge = metal_charges.loc[metal_charges["Metal"] == metal]["Charge"].values[0]
        cmd = ["molsimplify", "-core",metal,"-lig"]

        for el in ligs:
            if not metal in el:
                ligands.append(el)
            else:
                valid_repeat += 1

        cmd = ["molsimplify", "-core",metal,"-lig"]

        for ligand in ligands:
            cmd.append(ligand)

        repeatation = "1"

        if len(ligands)==1:
            repeatation = str(charge)
        else:
            for i in range(len(ligands)-1):
                repeatation += ",1"

        cmd.append("-ligocc")
        cmd.append(repeatation)
        cmd.append("-skipANN")
        cmd.append("True")
        cmd.append("-rundir")
        cmd.append(dir)
        print(cmd)
        if valid_repeat < 2:
            subprocess.run(cmd,check=True)
    except Exception as e:
        print(e)


if __name__ == '__main__':
  pool = Pool(processes=20)
  pool.map_async(generate_xyz,input_list)
  pool.close()
  pool.join()

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = comm.Get_size()

import numpy as np
import os


if rank == 0:
    os.chdir(r"D:\1. Documentos\0. Bases de datos\0. Dofiles y Scripts\Diplomado_PUCP\Big Data\Tarea 2")
    narray = np.genfromtxt("tarea2.csv", delimiter= ",", dtype=float)
    narray = narray[~np.isnan(narray)] 
    len_narray = int(len(narray)/3)
else:
    narray = None
    len_narray = None

len_narray = comm.bcast(len_narray, root=0)

local_data = np.empty(len_narray, dtype=float)
comm.Scatter(narray, local_data, root=0)
    
print(rank)
print(local_data)
print(f'{len_narray}')


'''
local_data = np.empty(len_narray, dtype=float)
comm.Scatter(narray, local_data, root=0)
local_max = np.max(local_data)

max_total = comm.gather(local_max, root=0)

if rank == 0:
    max_total = max(max_total)
    print(max_total)
'''

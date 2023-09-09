from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = comm.Get_size()

import numpy as np
import os
os.chdir(r"D:\1. Documentos\0. Bases de datos\0. Dofiles y Scripts\Diplomado_PUCP\Big Data\Tarea 2")
narray = np.genfromtxt("tarea2.csv", delimiter= ",", dtype=float)
narray = narray[~np.isnan(narray)]

if rank == 0:
    length = int(len(narray)/3)
    max_1 = np.max(narray[:length])
    comm.send(max_1, dest=3 , tag=3)
elif rank ==1:
    length_1 = int(len(narray)/3)
    length_2 = int(2*len(narray)/3)
    max_2 = np.max(narray[length_1:length_2])
    comm.send(max_2, dest=3 , tag=3)
elif rank ==2:
    length_2 = int(2*len(narray)/3)
    max_3 = np.max(narray[length_2:])
    comm.send(max_3, dest=3 , tag=3)
elif rank ==3:
    max_1 = comm.recv(source=0 , tag=3)
    max_2 = comm.recv(source=1 , tag=3)
    max_3 = comm.recv(source=2, tag=3)
    max_total = max(max_1,max_2,max_3)
    print(max_total)

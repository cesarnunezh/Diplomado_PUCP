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
    max_1 = np.max(narray)
    print(max_1)


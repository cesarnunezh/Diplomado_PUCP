from mpi4py import MPI
import numpy as np
import os

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = comm.Get_size()

py_size = size - 1
if rank == py_size:
    os.chdir(r"C:\Users\User\Documents\GitHub\Diplomado_PUCP_trabajos\Big Data\Tarea 2")
    narray = np.genfromtxt("tarea2.csv", delimiter= ",", dtype=float)
    narray = narray[~np.isnan(narray)]
    len_narray = int(len(narray)/py_size)
    narray = narray.reshape(py_size, len_narray)
else:
    narray = None
    len_narray = None

len_narray = comm.bcast(len_narray, root=py_size)
narray = comm.bcast(narray, root=py_size)

if rank == py_size:
    max_local = None
else:
    max_local = max(narray[rank])

max_total = comm.gather(max_local, root=py_size)

if rank == py_size:
    max_final = max(max_total[:py_size-1])
else:
    max_final = None

print(f'Resultados del procesador N° {rank}')
print(max_final)

from mpi4py import MPI

import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = np.arange(100000000, dtype=np.float)
    comm.send(data, dest=1)
elif rank == 1:
    data = np.empty(100, dtype=np.float)
    data = comm.recv(source=0)
    print(data)

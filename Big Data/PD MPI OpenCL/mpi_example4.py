from mpi4py import MPI

import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = np.arange(100000000, dtype=np.float)
    comm.Send(data, dest=1)
elif rank == 1:
    data = np.empty(100000000, dtype=np.float)
    comm.Recv(data, source=0)
    print(data)
    

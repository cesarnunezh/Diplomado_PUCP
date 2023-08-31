from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    data = {'Perú': 2, 'Australia': 0}
    comm.send(data, dest=1 , tag=1)
    comm.send(data, dest=2 , tag=1)
elif rank == 1:
    data = comm.recv(source=0 , tag=1)
    print("I am processor", rank)
    print(data)
elif rank == 2:
    data = comm.recv(source=0 , tag=1)
    print("I am processor", rank)
    print(data)

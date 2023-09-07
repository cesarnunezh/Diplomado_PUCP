from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    data1 = {'a': 7, 'b': 3.14}
    comm.send(data1, dest=1 , tag=1)
    data2 = {'Perú': 2, 'Australia': 0}
    comm.send(data2, dest=2 , tag=2)
elif rank == 1:
    data1 = comm.recv(source=0 , tag=1)
    print("I am processor", rank)
    print(data1)
elif rank == 2:
    data2 = comm.recv(source=0 , tag=2)
    print("I am processor", rank)
    print(data2)

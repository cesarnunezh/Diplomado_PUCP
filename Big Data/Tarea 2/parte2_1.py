from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    data = [x for x in range(1, 11) if x % 2 == 0]
    comm.send(data, dest=1 , tag=1)
    comm.send(data, dest=2 , tag=1)
    comm.send(data, dest=3 , tag=1)
elif rank == 1:
    data = comm.recv(source=0 , tag=1)
    print("Soy el procesador", rank, "y esta es la data:", data )
elif rank == 2:
    data = comm.recv(source=0 , tag=1)
    print("Soy el procesador", rank, "y esta es la data:", data)
elif rank == 3:
    data = comm.recv(source=0 , tag=1)
    print("Soy el procesados", rank, "y esta es la data:", data)

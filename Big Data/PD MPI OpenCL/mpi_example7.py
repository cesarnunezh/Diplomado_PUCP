from mpi4py import MPI

# Get my rank and the number of ranks
rank = MPI.COMM_WORLD.Get_rank()
n_ranks = MPI.COMM_WORLD.Get_size()

if rank != 0:
    # All ranks other than 0 should send a message
    message = "Hello World, I'm rank {:d}".format(rank)
    MPI.COMM_WORLD.send(message, dest=0, tag=0)

else:
    # Rank 0 will receive each message and print them
    for sender in range(1, n_ranks):
        message = MPI.COMM_WORLD.recv(source=sender, tag=0)
        print(message)

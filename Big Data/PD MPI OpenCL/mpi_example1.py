from mpi4py import MPI

comm = MPI.COMM_WORLD # all available processors to communicate 
rank = comm.Get_rank() # give ranks to processors, local variable for e/ processor
size = comm.Get_size() # each processor identifies the total processors 

print("Hello World from rank ", rank, " out of ", size, " processors ")

# 02_hello_world.py
# !pip install mpi4py ---> Solo en Jupyter Notebooks

from mpi4py import MPI
import numpy as np

### Ejemplo 1: Saludar 
comm = MPI.COMM_WORLD # all available processors to communicate 
rank = comm.Get_rank() # give ranks to processors, local variable for e/ processor
size = comm.Get_size() # each processor identifies the total processors 

print("Hello World from rank ", rank, " out of ", size, " processors ")

# #### Ejemplo 2: SPMD
# rank = MPI.COMM_WORLD.Get_rank()
# a = 6.0
# b = 3.0
# if rank == 0:
#     print("I am processor", rank, " : ", a + b)
# if rank == 1:
#     print("I am processor", rank, " : ", a * b)
# if rank == 2:
#     print("I am processor", rank, " : ", max(a, b))

# ### Ejemplo 3: Point-to-Point Communication (Arbitrary Python Objects)
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# data = {}

# if rank == 0:
#     data = {'a': 7, 'b': 3.14}
#     comm.send(data, dest=1)
# elif rank == 1:
#     data = comm.recv(source=0)

# print("I am processor", rank, "Data is:", data)
    
#### Ejemplo 4: Point-to-Point Communication NumPy Arrays 
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# data = np.zeros(100)

# if rank == 0:
#     data = np.arange(100, dtype=float)
#     comm.Send(data, dest=1)
# elif rank == 1:
#     data = np.empty(100, dtype=float)
#     comm.Recv(data, source=0)

# print("I am processor", rank, "Data is:", data)

# #### Ejemplo 5: Broadcast 
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# if rank == 0:
# 	data = ['info', 'only', 'in', 'processor 0']

# else:
# 	data = None

# data = comm.bcast(data, root=0) #uncomment and see what happens 
# print("I am processor", rank, "Data is:", data)
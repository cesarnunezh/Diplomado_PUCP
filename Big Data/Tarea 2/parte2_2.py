from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    data = {'cursos favoritos' : ( 'R básico', 'Python básico',  'Python intermedio', 'Big data')}
    valor = [i for i in range(size)]
else:
    data = None
    valor = None 
start = time.time()
data = comm.bcast(data, root=0)
valor = comm.scatter(valor, root=0)
print("Soy el procesador", rank, "recibí los cursos:", data['cursos favoritos'])
end = time.time()
print(end - start)
assert valor == rank
print(valor)

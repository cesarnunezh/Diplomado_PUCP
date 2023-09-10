from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    data = {'cursos favoritos' : ( 'R básico', 'Python básico',  'Python intermedio', 'Big data')}
    valor = [1,2,3]
else:
    data = None
data = comm.bcast(data, root=0)
print("Soy el procesador", rank, "recibí los cursos:", data['cursos favoritos'])

valor = None
valor = comm.scatter(valor, root=0)
assert valor == (rank)
print(valor)

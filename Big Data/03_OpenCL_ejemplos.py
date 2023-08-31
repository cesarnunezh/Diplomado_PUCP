import pyopencl as cl
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

a = np.random.rand(50000).astype(np.float32)
a_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=a.nbytes)
cl.enqueue_copy(queue, a_buf, a)

# Build program, write kernels
prg = cl.Program(ctx, """
	__kernel void twice(__global float *a)
	{
	int gid = get_global_id(0);
	a[gid] = 2*a[gid];
	}
	""").build()

# Running kernel
prg.twice(queue, a.shape, None, a_buf)

#Copy back to host 
result = np.empty_like(a)
cl.enqueue_copy(queue, result, a_buf)

# Arrays de PyOpenCL
import pyopencl.array as cl_array

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

a = np.random.rand(50000).astype(np.float32)
a_dev = cl_array.to_device(queue, a)

# Double all entries on GPU
twice = 2*a_dev

# Turn back into NumPy Array
twice_a = twice.get()

# Numeros aleatorios 
import pyopencl.clrandom as clrand

n = 10**6
a = clrand.rand(queue, n, np.float(32))
b = clrand.rand(queue, n, np.float(32))

# Map operations
a = clrand.rand(queue, n, np.float(32))
b = clrand.rand(queue, n, np.float(32))
c1 = 5*a + 6*b
result_np = c1.get()

from pyopencl.elementwise import ElementwiseKernel
lin_comb = ElementwiseKernel(ctx,
	"float a, float *x, float b, float *y, float *c",
	"c[i] = a*x[i] + b*y[i]")

c2 = cl.array.empty_like(a)
lin_comb(5, a, 6, b, c2)
result_np = c2.get()

# Reduce operations 
from pyopencl.reduction import ReductionKernel

n = 10**7
x = clrand.rand(queue, n, np.float64)

rknl = ReductionKernel(ctx, np.float64,
						neutral="0",
						reduce_expr="a+b", 
						map_expr="x[i]*x[i]",
						arguments="double *x")

result = rknl(x)
result_np = result.get()

# Scan operations 
np.cumsum([1,2,3])
array([1, 3, 6])

from pyopencl.scan import GenericScanKernel

sknl = GenericScanKernel(ctx, np.float64,
						 arguments="double *y, double *x",
						 input_expr="x[i]",
						 scan_expr="a+b", 
						 neutral="0",
						 output_statement="y[i] = item")

result = cl.array.empty_like(x)
sknl(result, x)
result_np = result.get()
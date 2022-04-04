from numba import jit
import numpy as np
# to measure exec time
from timeit import default_timer as timer

# normal function to run on cpu
def func(a,n):							
	for i in range(n):
		a[i]+= 1	

# function optimized to run on gpu
@jit()						
def func2(a,n):
	for i in range(n):
		a[i]+= 1

if __name__=="__main__":
	n = 10000000
	a = np.ones(n, dtype = np.float64)
	b = np.ones(n, dtype = np.float32)
	
	start = timer()
	func(a,n)
	print("without GPU:", timer()-start)
	
	start = timer()
	func2(a,n)
	print("with GPU:", timer()-start)

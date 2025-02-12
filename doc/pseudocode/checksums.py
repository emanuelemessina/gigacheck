dir = 0
matrix=[]
threadID = 0
blockID = 0
matrix_size = 0
threads_per_block = 0
shared_mem = 0
def log_2(): pass
def _sync(): pass
def get_element(): pass
def set_element(): pass
def log_reduction(): pass
vec = 0
def store_result(): pass

# get_element: matrix[blockID, index] or viceversa based on direction

sum = 0

for index in range(matrix_size, step=threads_per_block):
	sum += get_element(index)

shared_mem[threadID] = sum
_sync()

log_reduction()

if(threadID == 0):
	store_result(shared_mem[0])

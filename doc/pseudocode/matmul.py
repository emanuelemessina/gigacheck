num_tiles = 0
tile_size = 0
shared_A = []
shared_B = []
A, B = 0
threadY, threadX = 0
blockY, blockX = 0
def load_value(): pass
def _sync(): pass
def store_add(): pass
C = 0


val = 0

for tile in num_tiles:
	shared_A[threadY, threadX] =
		load_value(A, (blockY, tile), (threadY, threadX))
	
	shared_B[threadY, threadX] =
		load_value(B, (tile, blockX), (threadY, threadX))

	_sync()

	for el in tile_size:
		val += shared_A[threadY, el] * shared_B[el, threadX]

	_sync()

store_add(C, (blockY, blockX), (threadY, threadX), val)

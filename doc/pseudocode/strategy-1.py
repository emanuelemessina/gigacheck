result_matrix = 0
num_split_common_dim = 0

def load_block():
	pass
def store_block():
	pass
def _sync():
	pass

for result_block in result_matrix:
	for tmp_block in num_split_common_dim:
		A = load_block()
		B = load_block()
		_sync()

		C = A*B
		_sync()
		
	store_block(C)

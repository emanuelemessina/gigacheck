result_matrix = 0
num_split_common_dim = 0

def load_block():
	pass
def store_block():
	pass
def _sync():
	pass

A = load_block()
B = load_block()
_sync()

for result_block in result_matrix:
	for tmp_block in num_split_common_dim:
		A_alt = load_block()
		B_alt = load_block()
		
		_sync("copy of C")
		C = A*B
		_sync()

		A = A_alt
		B = B_alt
	
	store_block(C)

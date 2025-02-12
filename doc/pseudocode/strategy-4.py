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
		B = load_block()
		A = load_block()
		B_alt = load_block()
		
		C = A*B
		C_alt = A*B_alt
		_sync()
	
	store_block(C)
	store_block(C_alt)
	_sync()

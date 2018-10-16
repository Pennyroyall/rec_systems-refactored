from utils import *


R = pickle_load(path_data + 'R.pckl')
U, I = np.shape(R)


print(f'Nonzero entries, amount: {np.sum(R>0)}')

for u in range(U):
	print(np.mean(R[u, :].data))
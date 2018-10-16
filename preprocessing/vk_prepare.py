from utils import *



def scale_grid(R):
    grid = [0, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    R_g = (R > 0).astype('int')
    for g in grid:
        R_g += (R > g).astype('int')

    pickle_dump(R, path_data + 'R_scaled_grid.pckl')  
    return R_g   


if __name__ == '__main__': 
	R_train = pickle_load(path_data + 'vk_R_train.pckl')
	R = pickle_load(path_data + 'vk_R.pckl')

	R_test = R - R_train
	R = None

	pickle_dump(scale_grid(R_train), path_data + 'vk_R_train.pckl')
	pickle_dump(scale_grid(R_test), path_data + 'vk_R_test.pckl')



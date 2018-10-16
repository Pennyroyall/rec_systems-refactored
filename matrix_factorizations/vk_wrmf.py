from utils import *



def scale_grid(R):
    grid = [0, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    R_g = (R > 0).astype('int')
    for g in grid:
        R_g += (R > g).astype('int') 
    return R_g   



FACTORS = 30
LMBDA = 200
ITER = 25
CONFIDENCE = 13

R_train = pickle_load(path_data + 'vk_R_train.pckl')

model = AlternatingLeastSquares(factors=FACTORS, 
                                regularization=LMBDA, 
                                iterations=ITER, 
                                calculate_training_loss=False, 
                                num_threads=0)
R_train_a = (CONFIDENCE * scale_grid(R_train)).astype('double')
model.fit(R_train_a.T)

P, Q = model.user_factors, model.item_factors.T 
# pickle_dump(P, path_data + 'P_wrmf.pckl')
# pickle_dump(Q, path_data + 'Q_wrmf.pckl') 
print(f'P shape is: {np.shape(P)}')
print(f'Q shape is: {np.shape(Q)}')


R = pickle_load(path_data + 'vk_R.pckl')
R_test = R - R_train 
R = None

ndcg = matrix_ndcg(P, Q, R_train, R_test, p=0.003)
print(f'NDCG for wrmf is: {ndcg}')

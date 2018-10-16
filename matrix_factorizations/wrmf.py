from utils import *



FACTORS = 128
LMBDA = 100
ITER = 50
CONFIDENCE = 1.1

R_train = pickle_load(path_data + 'R_train.pckl')

model = AlternatingLeastSquares(factors=FACTORS, 
                                regularization=LMBDA, 
                                iterations=ITER, 
                                calculate_training_loss=False, 
                                num_threads=0)
R_train_a = (CONFIDENCE * R_train).astype('double')
model.fit(R_train_a.T)

P, Q = model.user_factors, model.item_factors.T 
pickle_dump(P, path_data + 'P_wrmf.pckl')
pickle_dump(Q, path_data + 'Q_wrmf.pckl') 
print(f'P shape is: {np.shape(P)}')
print(f'Q shape is: {np.shape(Q)}')


R_test = pickle_load(path_data + 'R_test.pckl')
ndcg = matrix_ndcg(P, Q, R_train, R_test)
print(f'NDCG for wrmf is: {ndcg}')

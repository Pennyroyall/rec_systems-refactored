from utils import *


class SBMF:

    def __init__(self, factors, lmbda, lr, alpha, verbose=None, progress=None):
        self.factors = factors
        self.lmbda = lmbda
        self.lr = lr
        self.alpha = alpha
        self.verbose = verbose
        self.progress = progress


    def _info(self, it):
        mode = '\r'
        to_print = f'Iteration: {(it+1):4d}; '
        
        if self.progress:
            ndcg = self.ndcg()
            to_print += f'NDCG :{ndcg:3.3f};   '

        if self.verbose >= 2:
            to_print += f'GRAD_P_MEAN is: {np.mean(self.grad_P):+4.4f}; '
            to_print += f'GRAD_P_MAX is: {np.max(self.grad_P):+4.4f};   '
            mode = '\n'
        if self.verbose >= 3:

            to_print += f'GRAD_Q_MEAN is: {np.mean(self.grad_Q):+4.4f}; '
            to_print += f'GRAD_Q_MAX is: {np.max(self.grad_Q):+4.4f};   '            
        print(to_print, end=mode)


    def _meow(self):
        to_print = f'Train started with following parameters: \n \
        factors : {self.factors} \n \
        learning_rate : {self.lr} \n \
        alpha : {self.alpha} \n \
        lambda : {self.lmbda} \n \
        shape of R_train : {np.shape(R_train)} \n \
        number of nonzero elements in R_train : {len(self.R_train.data)} \n'
        print(to_print)               
     


    def _get_grads(self, it, beta1=0.9, beta2=0.999, eps=10**-8):
        self.grad_P = self.P.dot(self.Q.dot(self.Q.T)) \
                      - self.R_train.dot(self.Q.T)  \
                      + self.lmbda * self.P
        self.grad_P = self.grad_P / self.U         

        self.grad_Q = self.P.T.dot(self.P).dot(self.Q) \
                      - (self.R_train.T.dot(self.P)).T \
                      + self.alpha * self.Q.dot(self.Q.T).dot(self.Q) \
                      - self.alpha * (self.S.T.dot(self.Q.T)).T \
                      + self.lmbda * self.Q
        self.grad_Q = self.grad_Q / self.I 


        self.momentum_P = beta1 * self.momentum_P \
                          + (1 - beta1) * self.grad_P
        self.disp_P = beta2 * self.disp_P \
                      + (1 - beta2) * (np.multiply(self.grad_P, self.grad_P))
        
        self.momentum_Q = beta1 * self.momentum_Q \
                          + (1 - beta1) * self.grad_Q
        self.disp_Q = beta2 * self.disp_Q \
                      + (1 - beta2) * (np.multiply(self.grad_Q, self.grad_Q))
        

        self.corrected_momentum_P = self.momentum_P / (1 - beta1**it)
        self.corrected_disp_P = self.disp_P / (1 - beta2**it)

        self.corrected_momentum_Q = self.momentum_Q / (1 - beta1**it)
        self.corrected_disp_Q = self.disp_Q / (1 - beta2**it)


        self.diff_P = np.divide(self.corrected_momentum_P, 
                                np.sqrt(self.corrected_disp_P + eps)
                               )
        self.diff_Q = np.divide(self.corrected_momentum_Q, 
                                np.sqrt(self.corrected_disp_Q + eps)
                               )


    def _update(self, it):
        self._get_grads(it)
        self.P, self.Q = (self.P - self.lr * self.diff_P,
                          self.Q - self.lr * self.diff_Q)             


    @timeit()   
    def fit(self, R_train, S, R_test, iter_number):
        self.iter_number = iter_number
        self.R_train = R_train
        self.R_test = R_test
        self.S = S

        self.U, self.I = np.shape(self.R_train)
        self.P = np.random.rand(self.U, self.factors) * 0.01
        self.Q = np.random.rand(self.factors, self.I) * 0.01

        self.momentum_P = np.random.rand(self.U, self.factors) * 0.00
        self.disp_P = np.random.rand(self.U, self.factors) * 0.00
        self.momentum_Q = np.random.rand(self.factors, self.I) * 0.00
        self.disp_Q = np.random.rand(self.factors, self.I) * 0.00

        self._meow()

        for it in range(1, self.iter_number):
            self._update(it)
            if ((it+1) % 2 ==0) and (self.verbose):
                self._info(it)

        print(f'\n           FINAL NDCG :{self.ndcg():4.5f}')    


    @staticmethod
    def dcg_score(vector):
        return vector[0] + np.sum(vector[1:] / np.log2(np.arange(2, vector.size + 1)))
                

    def ndcg(self, n=10, p=1):
        ndcg = 0
        corr = 0
        for u in range(self.U):
            if np.random.rand()<=p:
                temp = np.asarray(self.P[u, :].dot(self.Q)).flatten()
                indices_train = self.R_train[u, :].indices
                temp[indices_train] = -np.inf
                indices_top = np.argpartition(temp, -n)[-n:]
                indices_pred = np.argsort(temp[indices_top])[::-1]
                pred = indices_top[indices_pred]

                l = min(n, len(self.R_test[u, :].indices))
                vector = np.zeros(l)
                for i in range(l):
                    if self.R_test[u, pred[i]] > 0:
                        vector[i] = 1
                if l>0:
                    score = dcg_score(vector)
                    ideal = dcg_score(np.ones(l))
                    ndcg += score/ideal
                else:
                    corr += 1    
        return ndcg / (self.U - corr)      



if __name__ == '__main__':
    FACTORS = 128
    LMBDA = 21
    LR = 0.15
    ALPHA = 2

    R_train = pickle_load(path_data + 'R_train.pckl')
    R_test = pickle_load(path_data + 'R_test.pckl')
    S = pickle_load(path_data + 'S.pckl')

    model = SBMF(factors=FACTORS, lmbda=LMBDA, 
                 lr=LR, alpha=ALPHA, 
                 verbose=2, progress=False)
    model.fit(R_train, S, R_test, 200)

    
    # ndcg = model.ndcg(R_test)
    # print(f'NDCG for sbmf is: {ndcg}')
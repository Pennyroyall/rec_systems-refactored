from utils import *

L = 10 # minimum length of playsession
T = 800 # maximum time (seconds) between songs
F = 8 # minimum listeners of a given song 
K = 7 # ~number of similar songs in S

@timeit()
def build_sequences(filepath):
    L = 10 # minimum length of playsession
    T = 800 # maximum time (seconds) between songs

    with io.open(filepath, errors='ignore') as f:
        sequences = []
        start = True
        for i, line in enumerate(f):
            if (i+1) % 1_000 == 0:
                print(f'SEQUENCES: lines of file processed: {(i+1):8d}', end='\r')

            if start:
                curr_list = []
                prev = line[:-1].split('\t')
                prev_user, prev_time, prev_song = prev[0], \
                                                  datetime.strptime(prev[1], '%Y-%m-%dT%H:%M:%SZ'), \
                                                  prev[5]
                curr_list.append(prev_song)
                start = False

            else:
                curr = line[:-1].split('\t')
                curr_user, curr_time, curr_song = curr[0], \
                                                  datetime.strptime(curr[1], '%Y-%m-%dT%H:%M:%SZ'), \
                                                  curr[5]
                if len(curr) == 6: # assert that line is not completely broken
                    if curr_user == prev_user:
                        if (prev_time - curr_time) > timedelta(seconds=T):
                            if len(curr_list) >= L:
                                sequences.append(curr_list[::-1])
                            curr_list = []
                    else:
                        if len(curr_list) >= L:
                            sequences.append(curr_list[::-1])
                        curr_list = []  

                    curr_list.append(curr_song)
                    prev_user, prev_time, prev_song = curr_user, \
                                                      curr_time, \
                                                      curr_song

                else:
                    # start = True # can be used for extra protection
                    continue
    pickle_dump(sequences, path_data + 'sequences.pckl')
    print(f'\nTotal sequences: {len(sequences)}')
    return sequences                    


@timeit()
def count_users(filepath):
    with io.open(filepath, errors='ignore') as f:
        U = 0
        prev_user = ""
        for i, line in enumerate(f):
            if (i+1) % 1_000 == 0:
                print(f'USERS: lines of file processed: {(i+1):8d}', end='\r')

            curr_user = line[:-1].split('\t')[0]
            if curr_user != prev_user:
                U += 1
            prev_user = curr_user   

    print(f'\nTotal users in dataset: {U}             ')
    return U


@timeit()
def check(filepath):
    with io.open(filepath, errors='ignore') as f:
        temp_dict = defaultdict(set)
        for i, line in enumerate(f):
            if (i+1) % 1_000 == 0:
                print(f'CHECK: lines of file processed: {(i+1):8d}', end='\r')

            curr = line[:-1].split('\t')
            curr_user, curr_song = curr[0], curr[5]
            temp_dict[curr_song].add(curr_user)

        songs_check = {k : len(v) for k, v in temp_dict.items()}
        return songs_check


@timeit()
def build_model(sequences=None):
    if not sequences:
        sequences = pickle_load(path_data + 'sequences.pckl')
    gensim_iters = 25
    model = Word2Vec(sequences, min_count=10, 
                     workers=4, sg=1,
                     size=100, window=5, 
                     negative=5, iter=gensim_iters) 
    return model


def filter(songs_check, model):
    vocabulary = list(model.wv.vocab.keys())
    valid_songs = [k for k, v in songs_check.items() if v>=F]
    filtered = list(set(vocabulary) & set(valid_songs))
    print(f'Total valid songs: {len(filtered)}')
    pickle_dump(filtered, path_data + 'filtered.pckl')
    return filtered


@timeit()
def build_S(model, vocabulary, K=K):
    vocabulary_dict = {k:i for i, k in enumerate(vocabulary)}    
    I = len(vocabulary)
    S = sparse.lil_matrix((I, I))
    for i, song in enumerate(vocabulary):
        if (i+1) % 1_000 == 0:
            print(f'S: songs processed: {(i+1):8d}', end='\r')

        similar = model.wv.most_similar(positive=song, topn=K)
        for name, value in similar:
            if name in vocabulary_dict.keys():
                S[i, vocabulary_dict[name]] = value

    S = S.tocsr()   
    pickle_dump(S, path_data + 'S.pckl')
    print(f'Number of nonzero elements in S: {len(S.data)}')
            

@timeit()
def build_R(filepath, U, vocabulary):

    vocabulary_dict = {k:i for i, k in enumerate(vocabulary)}
    I = len(vocabulary)

    R = sparse.lil_matrix((U, I))
    with io.open(filepath, errors='ignore') as f:
        start = True
        idx = 0
        for i, line in enumerate(f):
            if (i+1) % 1_000 == 0:
                print(f'R: lines of file processed: {(i+1):8d}', end='\r')

            curr = line[:-1].split('\t')
            curr_user, curr_song = curr[0], curr[5]
            if start:
                start = False
            else:
                if curr_user != prev_user:
                    idx += 1
            prev_user = curr_user    

            if curr_song in vocabulary_dict.keys():
                R[idx, vocabulary_dict[curr_song]] += 1

    R = R.tocsr()
    pickle_dump(R, path_data + 'R.pckl')
    print(f'\nShape of R matrix: {np.shape(R)}')
    return R


@timeit()
def scale(R):
    U, I = np.shape(R)
    for u in range(U):
        print(f'SCALE: users processed: {(u+1):8d}', end='\r')
        maximum = np.max(R[u, :].data)
        for i in R[u, :].indices:
            R[u, i] = 1 + 4*R[u, i]/maximum

    pickle_dump(R, path_data + 'R_scaled.pckl')        
    return R


def scale_grid(R):
    grid = [2, 4, 6, 8, 10, 12, 14, 17, 21, 25, 30, 36, 43, 58, 80]
    R_g = (R > 0).astype('int')
    for g in grid:
        R_g += (R > g).astype('int')

    pickle_dump(R, path_data + 'R_scaled_grid.pckl')  
    return R_g    


@timeit()
def random_split(R, p):
    # p is test size
    U, I = np.shape(R)
    R_train = sparse.lil_matrix((U, I))
    R_test = sparse.lil_matrix((U, I))

    for u in range(U):
        print(f'SPLIT: users processed: {(u+1):8d}', end='\r')
        for i in R[u, :].indices:
            if np.random.rand() >= p:
                R_train[u, i] = R[u, i]
            else:
                R_test[u, i] = R[u, i]

    R_train = R_train.tocsr()
    R_test = R_test.tocsr()            
    pickle_dump(R_train, path_data + 'R_train.pckl')            
    pickle_dump(R_test, path_data + 'R_test.pckl')
    print(f'\nAll is ok? {np.max(np.abs(R - R_train - R_test)) == 0.0}')
    print(f'Mean of R_train is: {np.mean(R_train.data)}')
    print(f'Mean of R_test is: {np.mean(R_test.data)}')


def show_stats(R):
    U, I = np.shape(R)
    _max = R.max()
    _nnz = len(R.data)
    _sparsity = _nnz / (U * I)
    _less = np.sum(np.sum((R>0), axis=0) < 10)

    print(f'Shape of R is: {(U, I)}')
    print(f'Max element in R is: {_max}')
    print(f'Number of nonzero elements in R is: {_nnz}')
    print(f'Sparsity of R is: {_sparsity}')
    print(f'Songs with less than 10 users: {_less}')



if __name__ == "__main__":
    filepath = path_data + "sample_3.tsv"

    sequences = build_sequences(filepath)

    songs_check = check(filepath)
    U = count_users(filepath)

    model = build_model(sequences)
    filtered = filter(songs_check, model)

    S = build_S(model, filtered)
    R = build_R(filepath, U, filtered)
    show_stats(R)

    # R_scaled = scale(R)
    R_scaled = scale_grid(R)
    random_split(R_scaled, 0.2)




    


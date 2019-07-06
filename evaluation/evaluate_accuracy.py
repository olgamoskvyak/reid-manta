import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle

import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils.utils import rem_dupl
from metrics import acck, mapk


def evaluate_1_vs_all(train, train_lbl, test, test_lbl, n_eval_runs=10, move_to_db = 2, k_list = [1,5,10]):
    """ Compute accuracy on each class from test set given the training set in multiple runs.
    Input:
    train: 2D numpy float array: array of embeddings for training set, shape = (num_train, len_emb)
    train_lbl: 1D numpy integer array: array of training labels, shape = (num_train,)
    test: 2D numpy float array: array of embeddings for test set, shape = (num_test, len_emb)
    test_lbl: 1D numpy integer array: array of test labels, shape = (num_test,)
    n_eval_runs: integer, number of evaluation runs,default = 10
    move_to_db: integer, number of images to move to a database for each individual, default = 2
    k: array of integers, top-k accuracy to evaluate.
    
    Returns:
    mean_accuracy_1, mean_accuracy_5, mean_accuracy_10
    """
    print('Computing top-k accuracy for k=', k_list)
    if isinstance(k_list, int):
        k_list = [k_list]
    #Auxilary function to flatten a list
    flatten = lambda l: [item for sublist in l for item in sublist]

    #Evaluate accuracy at different k over a multiple runs. Report average results.
    acc = {k: [] for k in k_list}
    map_dict = {k: [] for k in k_list}
    
    for i in range(n_eval_runs):
        neigh_lbl_run = []
        db_emb, db_lbl, query_emb, query_lbl = get_eval_set_one_class(train, train_lbl, test, test_lbl, 
                                                           move_to_db = move_to_db)
        print('Number of classes in query set: ', len(db_emb)) 

        for j in range(len(db_emb)):
            neigh_lbl_un, _, _ = predict_k_neigh(db_emb[j], db_lbl[j], query_emb[j], k=10)
            neigh_lbl_run.append(neigh_lbl_un)

        query_lbl = flatten(query_lbl)
        neigh_lbl_run = flatten(neigh_lbl_run)

        #Calculate accuracy @k in a list of predictions
        for k in k_list:
            acc[k].append(acck(query_lbl, neigh_lbl_run, k=k, verbose=False))
            map_dict[k].append(mapk(query_lbl, neigh_lbl_run, k=k))

    #Report accuracy
    print('Accuracy over {} runs:'.format(n_eval_runs))
    acc_array = np.array([acc[k] for k in k_list], dtype=np.float32)
    acc_runs = np.mean(acc_array, axis=1)*100
    std_runs = np.std(acc_array, axis=1)*100
    print('Accuracy: ', acc_runs)
    print('Stdev: ', std_runs)
    for i, k in enumerate(k_list):
        print('ACC@{} %{:.2f} +-{:.2f}'.format(k, acc_runs[i], std_runs[i]))
    
    #Report Mean average precision at k
    print('MAP over {} runs:'.format(n_eval_runs))
    map_array = np.array([map_dict[k] for k in k_list], dtype=np.float32)
    map_runs = np.mean(map_array, axis=1)*100
    std_map_runs = np.std(map_array, axis=1)*100
    for i, k in enumerate(k_list):
        print('MAP@{} %{:.2f} +-{:.2f}'.format(k, map_runs[i], std_map_runs[i]))
       
    return dict(zip(k_list, acc_runs)), dict(zip(k_list, std_runs))

def predict_k_neigh(db_emb, db_lbls, test_emb, k=5):
    '''Predict k nearest solutions for test embeddings based on labelled database embeddings.
    Input:
    db_emb: 2D float array (num_emb, emb_size): database embeddings
    db_lbls: 1D array, string or floats: database labels
    test_emb: 2D float array: test embeddings
    k: integer, number of predictions.
    
    Returns:
    neigh_lbl_un - 2d int array of shape [len(test_emb), k] labels of predictions
    neigh_ind_un - 2d int array of shape [len(test_emb), k] labels of indices of nearest points
    neigh_dist_un - 2d float array of shape [len(test_emb), k] distances of predictions
    '''
    #Set number of nearest points (with duplicated labels)
    k_w_dupl = min(50, len(db_emb))
    nn_classifier = NearestNeighbors(n_neighbors=k_w_dupl, metric='euclidean')
    nn_classifier.fit(db_emb, db_lbls)

    #Predict nearest neighbors and distances for test embeddings
    neigh_dist, neigh_ind = nn_classifier.kneighbors(test_emb)

    #Get labels of nearest neighbors
    neigh_lbl = np.zeros(shape=neigh_ind.shape, dtype=db_lbls.dtype)
    for i, preds in enumerate(neigh_ind):
        for j, pred in enumerate(preds):
            neigh_lbl[i,j] = db_lbls[pred]
            
    #Remove duplicates
    neigh_lbl_un = []
    neigh_ind_un = []
    neigh_dist_un = []

    for j in range(neigh_lbl.shape[0]):
        indices = np.arange(0, len(neigh_lbl[j]))
        a, b = rem_dupl(neigh_lbl[j], indices)
        neigh_lbl_un.append(a[:k])
        neigh_ind_un.append(neigh_ind[j][b][:k].tolist())
        neigh_dist_un.append(neigh_dist[j][b][:k].tolist())

    return neigh_lbl_un, neigh_ind_un, neigh_dist_un


def get_eval_set_one_class(train, train_lbl, test, test_lbl, move_to_db = 1):
    """For each class in the test set get database and query set.
    For each class some samples are moved from test set to a database.
    Input:
    train: ndarray, train data, (num_train, ...)
    train_lbl: 1D numpy array, labels for train data, (num_train,)
    test: ndarray, test data, (num_test, ...)
    test_lbl: 1D numpy array, labels for test data, (num_test,)
    move_to_db: integer, number of samples to move to database from the test set, default = 1
    
    Returns:
    database: list of numpy arrays, len = num_unique_test_classes
    database_lbl:
    query:
    query_lbl:
    """
    test, test_lbl = shuffle(test, test_lbl, random_state=0)
    unique_lbl = np.unique(test_lbl)
    test_lbl = np.array(test_lbl)

    db_list = []
    db_lbl_list = []
    query_list = []
    query_lbl_list = []

    for label in unique_lbl:
        idx_to_db = np.random.choice(np.where(test_lbl == label)[0], size=move_to_db, replace=False)
        mask_query = np.array([True if i not in idx_to_db else False for i in np.where(test_lbl == label)[0]])
        idx_query = np.where(test_lbl == label)[0][mask_query]
        db_list.append(np.concatenate((train, test[idx_to_db]), axis=0))
        db_lbl_list.append(np.concatenate((train_lbl, test_lbl[idx_to_db]), axis=0))
        query_list.append(test[idx_query])
        query_lbl_list.append(test_lbl[idx_query])

    return db_list, db_lbl_list, query_list, query_lbl_list 
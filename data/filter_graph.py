import numpy as np

def filter_graph_known_nodes(X, y, edge_index, timestep):
    """
    Remove unknown nodes and rebuild edge index
    """

    # node yang dipakai
    known_mask = y != -1
    known_indices = np.where(known_mask)[0]

    # mapping old index -> new index
    new_index = {old:i for i,old in enumerate(known_indices)}

    # filter node feature
    X_new = X[known_mask]
    y_new = y[known_mask]
    t_new = timestep[known_mask]

    # rebuild edges
    src, dst = edge_index
    new_src, new_dst = [], []

    for s,d in zip(src,dst):
        if s in new_index and d in new_index:
            new_src.append(new_index[s])
            new_dst.append(new_index[d])

    edge_new = np.array([new_src,new_dst],dtype=np.int64)

    print("Filtered edges:", edge_new.shape[1])

    return X_new, y_new, edge_new, t_new

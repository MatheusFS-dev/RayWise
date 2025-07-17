from araras.ml.model.builders.gnn import check_gpu_limit

# Define your trial values
knn_values = [4, 8, 12, 16]
K_values = [2]
units_values = list(range(40, 400 + 1, 40))

check_gpu_limit(
    knn_list=knn_values,
    K_list=K_values,
    units_list=units_values,
    n=20*200,
)
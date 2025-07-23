from araras.ml.model.builders.cnn import generate_conv1d_pool_table

generate_conv1d_pool_table(
    L0=4000,
    n_layers=4,
    kernel_sizes=[8, 10, 12],
    pool_sizes=[2, 3, 4],
    filters=[512],
    csv_path="./all_combos.csv",
    plot=True,
)

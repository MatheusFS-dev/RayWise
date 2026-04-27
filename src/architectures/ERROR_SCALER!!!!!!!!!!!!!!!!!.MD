# inside objective(trial, **kwargs)
# keep raw inputs intact; work on local copies only
x_s008_lidar_train, x_s008_lidar_val, x_s008_coord_train, x_s008_coord_val, y_s008_train, y_s008_val = train_test_split(
    s008_lidar_input,
    s008_coord_input,
    s008_y_train,
    test_size=0.2,
    random_state=DATA_SEED,
    shuffle=True,
)

coord_scaler = StandardScaler()
coord_scaler.fit(x_s008_coord_train)  # fit only on raw train split

x_s008_coord_train = coord_scaler.transform(x_s008_coord_train)
x_s008_coord_val = coord_scaler.transform(x_s008_coord_val)
s009_coord_scaled = coord_scaler.transform(s009_coord_input)  # keep as local var
s008_coord_scaled = coord_scaler.transform(s008_coord_input)  # keep as local var

scaler_path = os.path.join(kwargs["scaler"], f"trial_{trial.number}.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(coord_scaler, f)
Logic: the scaler must be fitted on the raw training split used by that trial, then applied to the trial’s train/val/test inputs. By keeping s008_coord_input and s009_coord_input unchanged (no in-place overwrite), each trial’s scaler is trained on the correct raw distribution and the saved scaler reproduces the exact preprocessing used during training.
The 321M FLOPs is coming from compute over a long sequence, not from parameter count. The model reshapes (20, 200, 10) into a length‑4000 sequence and runs Conv1D layers over that length. Each weight is reused at every position, so FLOPs grow with sequence length and filter count even if params stay small.

What’s driving the 321M number:
combine_lidar_coord is (4000, 6) → Conv1D(128, k=9) over 4000 steps → ~27.6M MACs.
After pooling to length 1000 → Conv1D(256, k=4) → ~131.1M MACs.
MACs ×2 ≈ FLOPs → ~317M plus small extras, which lines up with the 321M.
Transformer based:

Number of Parameters: 705046 parameters 
FLOPs (batch size: 1): 536865806 FLOPs 
Inference Time: 2.1608×10^-3 s

Top-1: 57.12% (5505/9638)
Top-3: 78.89% (7603/9638)
Top-5: 84.40% (8134/9638)
Top-10: 90.07% (8681/9638)
Top-20: 94.72% (9129/9638)
Top-30: 96.90% (9339/9638)
Top-50: 98.34% (9478/9638)

GNN:

params: 1,467,076
FLOPs: 3.90 GFLOPs(64 batch) -> 60,937,500
Inference time: 3.5274×10^-2 s

Top-1: 60.82% (5862/9638)
Top-3: 80.90% (7797/9638)
Top-5: 86.76% (8362/9638)
Top-10: 92.14% (8880/9638)
Top-20: 96.52% (9303/9638)
Top-30: 97.71% (9417/9638)
Top-50: 98.43% (9487/9638)


CNN1d + GNN:

Number of parameters: 15,909,024
FLOPs: 4.91 GFLOPs (64 batch) -> 76.71875 MFLOPs.
Inference time: 4.594×10^-2 s
Top-1: 59.66% (5750/9638)
Top-3: 80.47% (7756/9638)
Top-5: 85.05% (8197/9638)
Top-10: 90.52% (8724/9638)
Top-20: 94.62% (9119/9638)
Top-30: 96.35% (9286/9638)
Top-50: 98.03% (9448/9638)

NAS KD

Number of Parameters: 157728 parameters (157.73 K)
Inference Time: 4.8854×10^-4 s
FLOPs (batch size: 1): 40710784 FLOPs (40.71 M)
Top-1: 61.23% (5901/9638)
Top-3: 82.19% (7921/9638)
Top-5: 86.98% (8383/9638)
Top-10: 92.30% (8896/9638)
Top-20: 96.25% (9277/9638)
Top-30: 97.87% (9433/9638)
Top-50: 98.78% (9520/9638)
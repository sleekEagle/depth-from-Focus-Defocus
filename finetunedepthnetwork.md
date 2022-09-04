# We fine tune the network that predicts depth given the blur. 
## 3 layer CNN. 
We cat the focal distance and blur image as different channels as the input to the CNN.\
Number of fileters in the first layer=10 (5 focal stacks and 5 focal distance maps).\ 
Following numbers are the number of filters in the each CNN layer\
trained with\
```
CUDA_VISIBLE_DEVICES=0 python train_fd.py --stack_num 5 --batchsize 20 --DDFF12_pth /scratch/lnw8px/depth-from-Focus-Defocus/data/my_ddff_trainVal.h5 --FoD_pth /scratch/lnw8px/depth-from-Focus-Defocus/defocus-net/data/fs_6/  --savemodel /scratch/lnw8px/depth-from-Focus-Defocus/models/fdf/1616  --use_diff 0 --use_blur 1  --fuse 1 --dchlist 10 16 16 1 --dkernel 3 --dpool 0 --lmd 0.2 --lr 0.000001
```

### 10 16 16 1   lr=0.000001
  MSE |---------RMS |----log RMS |----Abs_rel |----Sqr_rel |----------a1 |----------a2 |----------a3 |-------bump |------avgUnc | \
0.000428     0.019393     0.274599     0.217975     0.005487     62.083306     90.075798     96.910031     0.424829    -1.000000 \
runtime mean 0.19386023971902666 

### 10 16 32 1   lr=0.000001
 MSE |        RMS |    log RMS |    Abs_rel |    Sqr_rel |         a1 |         a2 |         a3 |       bump |     avgUnc |
   0.000416     0.019028     0.277995     0.277986     0.007042     61.451947     84.184519     96.133660     0.416702    -1.000000
runtime mean 0.1954678386899095

### 10 32 32 1  lr=0.000001
 MSE |        RMS |    log RMS |    Abs_rel |    Sqr_rel |         a1 |         a2 |         a3 |       bump |     avgUnc |
   0.000494     0.020434     0.288509     0.288982     0.008765     63.384124     84.792515     93.453015     0.424403    -1.000000
runtime mean 0.19506119483679382




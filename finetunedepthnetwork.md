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
 MSE |        RMS |    log RMS |    Abs_rel |    Sqr_rel |         a1 |         a2 |         a3 |       bump |     avgUnc |\
   0.000416     0.019028     0.277995     0.277986     0.007042     61.451947     84.184519     96.133660     0.416702    -1.000000\
runtime mean 0.1954678386899095

### 10 32 32 1  lr=0.000001
 MSE |        RMS |    log RMS |    Abs_rel |    Sqr_rel |         a1 |         a2 |         a3 |       bump |     avgUnc |\
   0.000494     0.020434     0.288509     0.288982     0.008765     63.384124     84.792515     93.453015     0.424403    -1.000000\
runtime mean 0.19506119483679382

### 10 64 64 1 lr=0.0000001
MSE |        RMS |    log RMS |    Abs_rel |    Sqr_rel |         a1 |         a2 |         a3 |       bump |     avgUnc |\
   0.000457     0.019991     0.280129     0.257527     0.007129     61.653223     87.368934     96.139225     0.430045    -1.000000\
runtime mean 0.1945554778803533

## 4 layer CNN
### 10 16 32 16 lr=1e-06
 MSE |        RMS |    log RMS |    Abs_rel |    Sqr_rel |         a1 |         a2 |         a3 |       bump |     avgUnc |\
   0.000490     0.020475     0.282700     0.271729     0.007906     61.759044     86.338416     95.317908     0.438767    -1.000000\ 
runtime mean 0.19821799460367942

### 10 32 64 16 lr=1e-06
  MSE |        RMS |    log RMS |    Abs_rel |    Sqr_rel |         a1 |         a2 |         a3 |       bump |     avgUnc |\
   0.000417     0.019278     0.273057     0.244236     0.006776     64.501512     88.589452     95.849792     0.430519    -1.000000\  
runtime mean 0.19326462577934841

### 10 64 32 16 lr=1e-04
 MSE |        RMS |    log RMS |    Abs_rel |    Sqr_rel |         a1 |         a2 |         a3 |       bump |     avgUnc |\
   0.000437     0.019646     0.284794     0.258328     0.006630     60.457665     86.588229     95.876301     0.418547    -1.000000\  
runtime mean 0.1935100579381588












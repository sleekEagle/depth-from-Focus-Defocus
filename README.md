# depth-from-Focus-Defocus

## Running in RIvanna 
1. log into Rivanna with SSH
2. start GPU interactive (ijob) session 
ijob -c 1 -A stressresearch -p gpu --gres=gpu --time=2400
3. pull the docker image from docker hub
4. run image with Singularity
singularity shell --writable-tmpfs --nv focus-defocus_latest.sif
in the image container, 
5. run screen 
in the screen session 
6. run visdom server and detach from screen : Ctrl+a and the press d
7. Goto sources directory of defocus net
8. python run_train.py

## How to use Docker 
remove images and containers 
https://www.digitalocean.com/community/tutorials/how-to-remove-docker-images-containers-and-volumes

general usage of Docker 
https://www.pluralsight.com/guides/create-docker-images-docker-hub


## Using Spyder from remote machine
Guide: 
https://docs.spyder-ide.org/current/panes/ipythonconsole.html?highlight=ssh#connect-to-a-remote-kernel

## copy files from Rivanna to local machine
scp mst3k@rivanna.hpc.virginia.edu:/project/mygroup_name/my_file /my_directory

## Datasets and normalization 
for each dataset, divide focal distances and distances by the max(focal distance)
### FoD500
focal distance : [0.1-1.5] \
normalized focal distance : [0.06667-1] \
distance : [0.1-2.83] \
Normalized distance : [0.0667-1.893] \
after clipping : [min(norm focal d) - max(norm focal d)] = [0.0667 - 1]

### DDFF12
focal distance : [0.02-0.28] \
normalized focal distance : [0.0714-1] \
distance : [0-0.3817] \
Normalized distance : [0-1.3633] \
after clipping : [min(norm focal d) - max(norm focal d)] = [0.0714 - 1]

## Evaluate 
```
python eval_DDFF12.py --stack_num 5 --loadmodel  /scratch/lnw8px/depth-from-Focus-Defocus/models/fdf/1632/FoD500_DDFF12_scale0.2_nsck5_lr1e-06_ep700_b20_lvl4/best.tar --use_diff 0 --outdir /scratch/lnw8px/depth-from-Focus-Defocus/models/fdf/1632/FoD500_DDFF12_scale0.2_nsck5_lr1e-06_ep700_b20_lvl4/ --data_path  /scratch/lnw8px/depth-from-Focus-Defocus/data/my_ddff_trainVal.h5 --dkernel 3 --dpool 0 --dchlist 10 16 32 1 --fuse 1
```





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

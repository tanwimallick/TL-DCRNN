#!/bin/bash                                                                                                                                     
#COBALT -n 1                                                                                                  
#COBALT -t 12:00:00  
#COBALT -A hpcbdsm                                                                                                                          

source ~/.bashrc


echo "Running Cobalt Job $COBALT_JOBID."

#mpirun -np 64 -ppn 2 python script64.py

mpirun -np 1  python dcrnn_train.py --config_filename=data/dcrnn_config_32transfer.yaml &> out_30e32c.out



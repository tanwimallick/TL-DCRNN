#!/bin/bash                                                                                                                                     
#COBALT -n 1                                                                                                  
#COBALT -t 2:00:00  
#COBALT -A hpcbdsm                                                                                                                          

source ~/.bashrc


echo "Running Cobalt Job $COBALT_JOBID."

#mpirun -np 64 -ppn 2 python script64.py

mpirun -np 1  python hdf_to_tfrecord.py --config_filename=input_files/tf_record_config.yaml



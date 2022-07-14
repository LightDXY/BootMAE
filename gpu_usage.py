import subprocess as sp
import os
import time
def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

_output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
ACCEPTABLE_AVAILABLE_MEMORY = 1024
COMMAND = "nvidia-smi --query-gpu=memory.total --format=csv"
memory_total_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
memory_total_values = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]
mem_total = memory_total_values[0]

print ('Start Checking, Total Memory:', mem_total)
while True:
    time.sleep(600)
    a = get_gpu_memory()[0]
    if a>mem_total*0.9:
        print ('Empty GPU',a)
        break
    else:
        print ("Running GPU",a)

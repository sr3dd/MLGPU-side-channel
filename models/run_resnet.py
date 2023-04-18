from resnet.inferencer import Inferencer
import sys
import pyvisa
import threading
import os
import time
from time import sleep
import subprocess
#import nvsmi as nv
import signal
import queue
import re
from datetime import datetime,timezone, timedelta

def nvidia_smi():
    while not stop_event.is_set():
        command="nvidia-smi -i 0 --format=csv --query-gpu=power.draw  | grep -oE '[0-9]+\.[0-9]+'"
        result=subprocess.run(command, shell=True, stdout=subprocess.PIPE)
        output = result.stdout.decode("utf-8")
        print(output)
        #time=datetime.now(timezone(timedelta(hours=-4),'EST')).strftime('%Y-%m-%d %H:%M:%S %Z%z')
        time=datetime.now(timezone(timedelta(hours=-4),'EST')).strftime('%F %T.%f')[:-3]
        smipower.append({'Power':output,'Time':time})
        sleep(0.1)

def power_measurement():
        
        rm = pyvisa.ResourceManager('@py')
        rm.list_resources()
        multimeter = rm.open_resource('USB0::2391::1543::MY47022388::0::INSTR')
        multimeter.write('MEAS:VOLT:DC:AUTO:RANGE ON')
        multimeter.write('MEAS:VOLT:DC:APER 0.01')
        # Continuous measurement loop
        while not stop_event.is_set():
            # Read the measured voltage value
            voltage = multimeter.query('READ?')
            voltage=float(voltage)
            power=voltage*12*1000
            # Print the voltage value
            print('Measured Power: {} W'.format(power))
            time=datetime.now(timezone(timedelta(hours=-4),'EST')).strftime('%F %T.%f')[:-3]
            powermeter_reading.append({'Power':power,'Time':time})
            # Wait for the specified delay before taking *the next measurement
            sleep(0.1)


def inference(batch_size):
    inferencer = Inferencer(custom_batch_size=batch_size)
    inferencer.load_test_data()
    inferencer.perform_inference()
    
if __name__ == '__main__':
    powermeter_reading=[]
    smipower=[] 
    
    if len(sys.argv) < 2:
        par = None
    else:
        par = int(sys.argv[1])

    #t = threading.Thread(target=power_measurement)
    t1=threading.Thread(target=nvidia_smi)
    stop_event = threading.Event()
    #t.start()  
    t1.start()  
    t2 = threading.Thread(target=inference, args=(par,))
    t2.start()
    t2.join()
    print("Inference is finished")
    stop_event.set()
    #t.join()
    t1.join()
    sleep(3)
    print("All computations are finished")    
    
    with open("powermeter.txt", "w+") as f:
        for i in powermeter_reading:
            f.write(f"{i['Power']}\t{i['Time']}\n")
    
    with open("smipower.txt", "w+") as f:
        for i in smipower:
            f.write(f"{i['Power']}\t{i['Time']}\n")

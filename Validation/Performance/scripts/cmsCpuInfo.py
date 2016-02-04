#!/usr/bin/env python
#Little script to get the number of cpus (cores) in the machine one is running on:
#Information is parsed from /proc/cpuinfo file
import os

def get_NumOfCores():
    cores=0
    cpuinfo=open("/proc/cpuinfo","r")
    if cpuinfo:
        for line in cpuinfo.readlines():
            for token in line.split():
                if token == 'processor':
                    cores=cores+1;
        return cores
    else:
        print "Could not open file /proc/cpuinfo!\n"
        return NULL
 #Further functions to parse other information will come (RAM, Cache etc)   

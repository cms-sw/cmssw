#! /usr/bin/env python
#Script to
#1-check for cmsScimarkLaunch (infinite loop) scripts
#2-kill them
#3-report their results using cmsScimarkParser.py

import subprocess,os,sys

def main():
    #Use ps -ef to look for cmsScimarkLaunch processes
    ps_stdouterr=subprocess.Popen("ps -efww|grep cmsScimarkLaunch|grep -v grep|grep -v 'sh -c'",shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    if ps_stdouterr:
        ps_lines=ps_stdouterr.readlines()
        #print ps_lines
    if ps_lines:
        for line in ps_lines:
            tokens=line.split()
            #Look up the PID
            PID=tokens[1]
            #Look up the cpu core
            core=tokens[9]
            print "Found process:\n%s"%line[:-1] #to eliminate the extra \n
            #Kill the PID
            print "Killing process with PID %s"%PID
            kill_stdouterr=subprocess.Popen("kill %s"%PID,shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read() 
            print kill_stdouterr
            #Harvest the cmsScimark scores
            #Look for the cmsScimark log:
            if os.path.exists("cmsScimark_%s.log"%core): 
                #Create the results dir
                mkdir_stdouterr=subprocess.Popen("mkdir cmsScimarkResults_cpu%s"%core,shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read()
                print mkdir_stdouterr
                #Execute the harvesting scrip cmsScimarkParser.py (it is in the release)
                harvest_stdouterr=subprocess.Popen("cmsScimarkParser.py -i cmsScimark_%s.log -o cmsScimarkResults_cpu%s"%(core,core),shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read()
                print harvest_stdouterr
            else:
                print "No cmsScimark_%s.log file was found for cpu%s, log might be in another directory!"%(core,core)
    else:
        print "No cmsScimarkLaunch processes found in the ps -ef output"
    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
#Utility script to kill with one command any cmsPerfSuite.py related process running.
######NOTE######
#USE AT YOUR OWN RISK!
#THIS SCRIPT WILL KILL ANY PROCESS THE USER HAS RUNNING THAT COULD BE PART OF
#THE PERFORMANCE SUITE EXECUTION!
import subprocess,os,sys,cmsScimarkStop,time
user=os.environ['USER']

def main():
    exitcode=0
    #First invoke cmsScimarkStop.py to stop eventual cmsScimarks running...
    cmsScimarkStop.main()
    
    #List of executables to spot in ps -ef and kill:
    scripts=['cmsPerfSuite.py',
             'cmsRelvalreportInput.py',
             'cmsRelvalreport.py',
             'cmsDriver.py',
             'cmsRun',
             'cmsScimark2',
             'cmsIgProf_Analysis.py',
             'igprof-analyse',
             'perfreport'
             ]
         
    
    print "Looking for processes by user %s"%user
    checkProcesses=subprocess.Popen("ps -efww|grep %s"%user,bufsize=4096,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    for line in checkProcesses.stdout:
        for executable in scripts:
            #print "Looking for %s script"%executable
            if executable in line:
                print "Found process %s"%line
                print "Killing it!"
                PID=line.split()[1]
                kill_stdouterr=subprocess.Popen("kill %s"%PID,shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read()
                print kill_stdouterr
                
    #There could be a few more processes spawned after some of the killings... give it a couple of iterations:
    i=0
    while i<100:
        exitcode=0
        checkProcesses=subprocess.Popen("ps -efww|grep %s"%user,bufsize=4096,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        for line in checkProcesses.stdout:
            for executable in scripts:
                if executable in line:
                    print "Something funny going on! It seems I could not kill process:\n%s"%line
                    print "SORRY!"
                    exitcode=-1
        if exitcode>=0:
            print "Finally killed all jobs!"
            break
        i+=1
        time.sleep(2)
    #Return 0 if all killed, could use the -1 exit code if we decide to put a limit to the number of cycles...
    return exitcode

if __name__ == "__main__":
    sys.exit(main())

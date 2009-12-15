#!/usr/bin/python
"""
Usage: ./cmsBenchmark.py [options]
       
Options:
  --cpu=...            specify the core on which to run the performance suite
  --cores=...          specify the number of cores of the machine (can be used with 0 to stop cmsScimark from running on the other cores)
  -n ..., --numevts    specify the number of events for each tests/each candle/each step
  --candle=...         specify the candle to run instead of all the 7 candles of the suite
  --step=...           specify the step to run instead of all steps of the suite
  --repeat=...         specify the number of times to re-run the whole suite
  -h, --help           show this help
  -d                   show debugging information

Legal entries for individual candles (--candle option):
HiggsZZ4LM190
MinBias
SingleElectronE1000
SingleMuMinusPt10
SinglePiMinusE1000
TTbar
QCD_80_120

Legal entries for specific tests (--step option):
GEN
SIM
DIGI
L1
DIGI2RAW
HLT
RAW2DIGI
RECO
and combinations of steps like:
GEN-SIM
L1-DIGI2RAW-HLT
DIGI2RAW-RAW2DIGI
and sequences of steps or combinations of steps like:
GEN-SIM,DIGI,L1-DIGI2RAW-RAW2DIGI,RECO
Note: when the necessary pre-steps are omitted, cmsPerfSuite.py will take care of it.

Examples:
./cmsBenchmark.py
     This will run with the default options --cpu=1, --cores=4, --numevts=100, --step=GEN-SIM,DIGI,RECO --repeat=1 (Note: all results will be reported in a directory called Run1).
OR
./cmsBenchmark.py --cpu=2
     This will run the test on core cpu2.
OR
./cmsBenchmark.py --cpu=0,1 --cores=8 -n 200
     This will run the suite with 200 events for all tests/candles/step, on cores cpu0 and cpu1 simulataneously, while running the cmsScimark benchmarks on the other 6 cores.
OR
./cmsBenchmark.py --cores=8 --repeat=10 --candle QCD_80_120 
     This will run the performance tests only on candle QCD_80_120, running 100 evts for all steps, and it will repeat these tests 10 times, saving the results in 10 separate directories (each called RunN, with N=1,..,10) to check for systematic/statistical uncertainties. Note that by default --repeat=1, so all results will be in a directory called Run1.
OR 
./cmsBenchmark.py --step=GEN-SIM,DIGI,RECO
     This will run the performance tests only for the steps "GEN,SIM" (at once), "DIGI" and "RECO" taking care of running the necessary intermediate steps to make sure all steps can be run.

"""
import os
#Get some environment variables to use
cmssw_base=os.environ["CMSSW_BASE"]
cmssw_release_base=os.environ["CMSSW_RELEASE_BASE"]
cmssw_version=os.environ["CMSSW_VERSION"]
host=os.environ["HOST"]
user=os.environ["USER"]

#Performance suites script used:
Script="cmsPerfSuite.py"

#Options handling
import getopt
import sys

def usage():
    print __doc__

def main(argv):
    #Some default values:
    #Number of cpu cores on the machine
    coresOption="4"
    cores=" --cores=4"
    #Cpu core(s) on which the suite is run:
    cpuOption=(1) #not necessary to use tuple for single cpu, but for type consistency use ().
    cpu=" --cpu=1"
    #Number of events per test (per candle/per step):
    numevtsOption="100"
    numevts=" --timesize=100"
    #default benchmark does not run igprof nor valgrind
    igprofevts=" --igprof=0"
    valgrindevts=" --valgrind=0"
    #Default option for candle is "" since, usually all 7 candles of the suite will be run!
    candleOption=""
    candle=""
    #Default option for step is ["GEN,SIM","DIGI","RECO"] since we don't need to profile all steps of the suite
    stepOption="GEN-SIM,DIGI,RECO"
    step=" --step="+stepOption
    #Default option for repeat
    repeatOption=1 #Use integer here since it will be used directly in the script
    #Let's check the command line arguments
    try:
        opts, args = getopt.getopt(argv, "n:hd", ["cpu=","cores=","numevts=","candle=","step=","repeat=","help"])
    except getopt.GetoptError:
        print "This argument option is not accepted"
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt == '-d':
            global _debug
            _debug = 1
        elif opt == "--cpu":
            cpuOption=arg
            cpus=cpuOption.split(",")
            cpu=" --cpu="+cpuOption
        elif opt == "--cores":
            coresOption = arg
        elif opt in ("-n", "--numevts"):
            numevtsOption = arg
            numevts=" --timesize="+arg
        elif opt == "--candle":
            candleOption = arg
            candle=" --candle="+arg
        elif opt == "--step":
            stepOption = arg
            steps=stepOption.split(",")
        elif opt == "--repeat":
            repeatOption = int(arg)
    #Case with no arguments (using defaults)
    if opts == []:
        print "No arguments given, so DEFAULT test will be run:"
    #Print a time stamp at the beginning:
    import time
    date=time.ctime()
    path=os.path.abspath(".")
    print "CMS Benchmarking started running at %s on %s in directory %s, run by user %s" % (date,host,path,user)
    #showtags=os.popen4("showtags -r")[1].read()
    #print showtags
    #For the log:
    print "This machine (%s) is assumed to have %s cores, and the suite will be run on cpu(s) %s" %(host,coresOption,cpuOption)
    print "%s events per test will be run" % numevtsOption
    if candleOption !="":
        print "Running only %s candle, instead of all the candles in the performance suite" % candleOption
    if stepOption != "":
        print "Profiling only the following steps: %s" % stepOption
        step=" --step="+stepOption
        #This "unpacking" of the steps is better done in cmsPerfSuite.py or the cmsSimPyRelVal.py (.pl for now)
        #steps=stepOption.split(",")
        #cmsPerfSuiteSteps=[]
        #for step in steps:
        #    newstep=reduce(lambda a,b:a+","+b,step.split("-"))
        #    cmsPerfSuiteSteps.append(newstep)
    if repeatOption !=1:
        print "The benchmarking will be repeated %s times" % repeatOption
    #Now let's play!
    for repetition in range(repeatOption):
        mkdircdcmd="mkdir Run"+str(repetition+1)+";cd Run"+str(repetition+1)
        #mkdircdstdout=os.popen4(mkdircmd)[1].read()
        #if mkdirstdout:
        #    print mkdirstdout,
        #print "Here we'd launch cmsPerfSuite.py!"
        PerfSuitecmd="cmsPerfSuite.py" + cpu + cores + numevts + igprofevts + valgrindevts + candle + step + ">& cmsPerfSuiteRun" + str(repetition + 1) + ".log"
        launchcmd=mkdircdcmd+";"+PerfSuitecmd
        print launchcmd
        sys.stdout.flush()
        #Obsolete popen4-> subprocess.Popen
        #launchcmdstdout=os.popen4(launchcmd)[1].read()
        launchcmdstdout=Popen(launchcmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.read()
        print launchcmdstdout
        
if __name__ == "__main__":
    main(sys.argv[1:])

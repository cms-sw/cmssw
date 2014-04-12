#!/usr/bin/env python

import os, re, sys, glob
import optparse as opt
import cmsPerfRegress as cpr
from cmsPerfCommons import Candles, CandFname, getVerFromLog

def getParameters():
    global _debug
    global PROG_NAME
    PROG_NAME = os.path.basename(sys.argv[0])
    parser = opt.OptionParser(usage="""%s [OLD_REL_DIR] [NEW_REL_DIR]

To compare 2 cmsPerfSuite.py directories pass the previous release as the first argument and the latest release as the second argument """ % PROG_NAME)
    #
    # Options
    #
    devel  = opt.OptionGroup(parser, "Developer Options",
                                     "Caution: use these options at your own risk."
                                     "It is believed that some of them bite.\n")    
    devel.add_option(
        '-d',
        '--debug',
        type='int',
        dest='debug',
        default = 0,
        help='Show debug output',
        #metavar='DEBUG',
        )
    parser.add_option_group(devel)
    (options,args) = parser.parse_args()
    _debug = options.debug
    
    if not len(args) == 2:
        print "ERROR: Not enough arguments"
        sys.exit()
        
    path1 = os.path.abspath(args[0])
    path2 = os.path.abspath(args[1])    
    if os.path.exists(path1) and os.path.exists(path2):
        return (path1, path2)
    else:
        print "Error: one of the paths does not exist"
        sys.exit()

def getOldRelName(oldRelName,adir):
    #Not sure this function is used but as it was written before it was useless.
    #Now it parses the adir directory looking for the CMSSW_X_Y_Z(_preN) in the path.
    if oldRelName == "":
        oldRelPath = os.path.dirname(adir)
        oldRelPathDirs = oldRelPath.split("/")
        for dir in oldRelPathDirs:
            if 'CMSSW' in dir:
                oldRelName=dir
    return oldRelName

def compareSimMemPair(newLog,candle,profdir,curdir,oldlog,oldRelName=""):
    print "oldlog %s"%oldlog
    print "curdir %s"%curdir
    #oldRelName = getOldRelName(oldRelName,olddir)
    oldRelName = getOldRelName(oldRelName,oldlog)
    print "OLD REL NAME: %s"%oldRelName
    #base = os.path.basename(newLog)
    #oldlog = os.path.join(olddir,curdir,base)
    rootf  = "simpmem-regress.root"
    try:
        print "TRY candle %s"%candle
        print "HERE Oldlog:%s"%oldlog
        print "HERE newLog:%s"%newLog
        cpr.cmpSimpMemReport(rootf,curdir,oldlog,newLog,1,True,candle,prevrev = oldRelName)
    except cpr.SimpMemParseErr, detail:
        print "WARNING: Could not parse data from log file %s; not performing regression" % detail.message
    except OSError, detail:
        print "WARNING: The OS returned the following error when comparing %s and %s" % (oldlog,log), detail
    except IOError, detail:
        print "IOError:", detail
    else:
        print "Successfully compared %s and %s" % (oldlog,newLog)        
        
def regressReports(olddir,newdir,oldRelName = "",newRelName=""):

    profSets = ["Callgrind",
                #"Memcheck", #No regression on Memcheck profiles!
                "IgProf",
                "TimeSize",
                #Adding the PU directories:
                "PU_Callgrind",
                "PU_IgProf",
                "PU_TimeSize"
                ]
    for candle in Candles:
        #Loop over the known profilers sets (tests) defined above:
        for profset in profSets:
            #Check there is a directory with the profile set (test) being considered:
            adir = os.path.join(newdir,"%s_%s" % (candle,profset))
            if os.path.exists(adir):
                #Start working in directory adir (e.g. MinBias_TimeSize)
                print "Found directory %s"%adir

                #Set up the profilers based on the directory name
                Profs = []
                if   profset == "Callgrind" or  profset == "PU_Callgrind":
                    Profs = ["valgrind"] # callgrind actually
                elif profset == "TimeSize" or profset == "PU_TimeSize":
                    Profs = [ "TimingReport",
                              #"TimeReport", We do not run regression on the plain html TimeReport profile...
                              "SimpleMemoryCheck",
                              "EdmSize"]
                elif profset == "IgProf" or profset == "PU_IgProf" :
                    Profs = [ "IgProfperf", #This was missing!
                              "IgProfMemTotal",
                              "IgProfMemLive"]
                #Now for each individual profile in the profile set (e.g for TimeSize TimeReport, TimingReport, SimpleMemoryCheck, EdmSize
                #collect the various logfiles
                for prof in Profs:
                    print "Checking %s profile(s)"%prof
                    if   prof == "EdmSize" or prof == "valgrind":
                        stepLogs = glob.glob("%s/%s_*_%s"       % (adir,CandFname[candle],prof))
                    elif prof == "IgProfMemLive" or prof == "IgProfMemTotal": 
                        stepLogs = glob.glob("%s/%s_*_%s.gz"       % (adir,CandFname[candle],"IgProfMemTotal")) #This hack necessary since we reuse the IgProfMemTotal profile for MemLive too (it's a unique IgProfMem profile, read with different counters) 
                    elif prof == "IgProfperf":
                        stepLogs = glob.glob("%s/%s_*_%s.gz"       % (adir,CandFname[candle],prof))
                    elif prof == "SimpleMemoryCheck":
                        #With the change in the use of tee now the SimpleMemoryCheck info will be in the _TimingReport.log too...
                        #The following lines only will work for the unprofiled steps... hence... no need to report them!
                        #stepLogs = os.path.join(adir,"%s.log" % candle)
                        stepLogs = glob.glob("%s/%s_*_%s.log"   % (adir,CandFname[candle],'TimingReport'))
                    elif prof == "TimingReport":
                        stepLogs = glob.glob("%s/%s_*_%s.log"   % (adir,CandFname[candle],prof))

                    #Debug:
                    print "Found the following step logs: %s"%stepLogs
                    
                    profdir = os.path.basename(adir)

                    #Giant if to single out the SimpleMemoryCheck case that is in the elif at the bottom... maybe should flip things around...
                    #Basically here we do everything but SimpleMemoryCheck:
                    if prof == "TimingReport" or prof == "EdmSize" or prof == "valgrind" or prof == "IgProfMemTotal" or prof == "IgProfMemLive" or prof == "IgProfperf":
                        #This hack necessary since we reuse the IgProfMemTotal profile for MemLive too
                        #(it's a unique IgProfMem profile, read with different counters)
                        if prof == "IgProfMemLive": 
                            stepreg = re.compile("%s_([^_]*(_PILEUP)?)_%s((.log)|(.gz))?" % (CandFname[candle],"IgProfMemTotal"))
                        else:
                            stepreg = re.compile("%s_([^_]*(_PILEUP)?)_%s((.log)|(.gz))?" % (CandFname[candle],prof))

                        #Loop on the step logfiles collected above
                        for log in stepLogs:
                            base = os.path.basename(log)
                            #Handle the fact the profile ("log") for IgProf is always compressed (.gz):
                            if prof == "IgProfMemTotal" or prof == "IgProfMemLive" or prof == "IgProfperf":
                                base = base.split(".gz")[0]
                            #Use the regular expression defined above to read out the step from the log/profile
                            searchob = stepreg.search(base)

                            #If in this log the regular expression was able match (and so to extract the step)
                            if searchob:
                                #print searchob.groups()
                                step = searchob.groups()[0]
                                #print "and the step taken is %s"%step
                                outpath = os.path.join(adir,"%s_%s_%s_regression" % (CandFname[candle],step,prof))
                                oldlog  = os.path.join(olddir,"%s_%s" % (candle,profset),base)
                                #Again handle the fact the profile ("log") for IgProf is always compressed (.gz):
                                if prof == "IgProfMemTotal" or prof == "IgProfMemLive" or prof == "IgProfperf":
                                    oldlog  = os.path.join(olddir,"%s_%s" % (candle,profset),base + ".gz")
                                if not os.path.exists(outpath):
                                    os.mkdir(outpath)
                                if os.path.exists(oldlog):
                                    try:
                                        print ""
                                        print "** "
                                        if not prof == "TimingReport":
                                            print "** Comparing", candle, step, prof, "previous release: %s, latest release %s" % (oldlog,log)
                                            print "**"

                                        if   prof == "EdmSize":
                                            cpr.cmpEdmSizeReport(outpath,oldlog,log)
                                        elif prof == "TimingReport":
                                            logdir = "%s_%s_%s" % (CandFname[candle],step,prof)
                                            outd   = os.path.join(adir,logdir)
                                            rootf  = "timing-regress.root" 
                                            oldlog = os.path.join(olddir,profdir,base)
                                            if os.path.exists(log) and os.path.exists(oldlog) and os.path.exists(outd):
                                                print "** Comparing", candle, step, prof, "previous release: %s and latest release: %s" % (oldlog,log)
                                                print "**"
                                                oldRelName = getOldRelName("",oldlog)
                                                #print "TIMING OLD REL extracted from %s :\n %s"%(oldlog,oldRelName)
                                                cpr.cmpTimingReport(rootf, outd, oldlog, log, 1, batch = True, prevrev = oldRelName)
                                            else:
                                                print "WARNING: While comparing", candle, step, prof, " at least one of the logfiles/directories: old (%s) or new (%s) was not found!!!" % (oldlog,log)
                                                break
                                        elif prof == "valgrind":
                                            cpr.cmpCallgrindReport(outpath,oldlog,log)
                                        elif prof == "IgProfperf":
                                            IgProfMemOpt="" #No need to specify the counter, for IgProfPerf...
                                            cpr.cmpIgProfReport(outpath,oldlog,log,IgProfMemOpt)
                                        elif prof == "IgProfMemTotal":
                                            IgProfMemOpt="-y MEM_TOTAL"
                                            cpr.cmpIgProfReport(outpath,oldlog,log,IgProfMemOpt)
                                        elif prof == "IgProfMemLive":
                                            IgProfMemOpt="-y MEM_LIVE"
                                            cpr.cmpIgProfReport(outpath,oldlog,log,IgProfMemOpt)
                                    except cpr.PerfReportErr,detail:
                                        print "WARNING: Perfreport return non-zero exit status when comparing %s and %s. Perfreport output follows" % (oldlog,log)
                                        print detail.message
                                    except cpr.TimingParseErr,detail:
                                        print "WARNING: Could not parse data from log file %s; not performing regression" % detail.message                                            
                                    except OSError, detail:
                                        print "WARNING: The OS returned the following error when comparing %s and %s" % (oldlog,log), detail
                                    except IOError, detail:
                                        print "IOError:", detail
                                    else:
                                        print "Successfully compared %s and %s" % (oldlog,log)                                            
                                else:
                                    print "WARNING: Could not find an equivalent logfile for %s in the previous release dir %s " % (log,oldlog)
                                        
                                                            
                            else:
                                continue
                    elif prof == "SimpleMemoryCheck":
                        #print "The logfiles for SimpleMemoryCheck are %s"%stepLogs
                        for log in stepLogs:
                            #print "The logfile considered now is %s"%log 
                            stepreg = re.compile("%s_([^_]*(_PILEUP)?)_%s((.log)|(.gz))?" % (CandFname[candle],"TimingReport"))
                            base = os.path.basename(log)
                            #Use the regular expression defined above to read out the step from the log/profile
                            searchob = stepreg.search(base)
                            #print "Value of searchob is %s"%searchob
                            #If in this log the regular expression was able match (and so to extract the step)
                            if searchob:
                                #print searchob.groups()
                                step = searchob.groups()[0]
                                print "and the step taken is %s"%step
                                #outpath = os.path.join(adir,"%s_%s_%s_regression" % (CandFname[candle],step,prof))
                                oldlog  = os.path.join(olddir,"%s_%s" % (candle,profset),base)
                            if os.path.exists(oldlog):
                                print ""
                                print "** "
                                print "** Comparing for SimpleMemoryCheck", candle, step, prof, "previous release: %s, latest release %s" % (oldlog,log)
                                print "**"
                                #The try/except is folded in the following function for SimpleMemoryCheck:
                                compareSimMemPair(log,candle,profdir,adir,oldlog,oldRelName="")
                               
    if newRelName == "":
        newRelName = getVerFromLog(newdir)
    regress = open("%s/REGRESSION.%s.vs.%s" % (newdir,getVerFromLog(olddir),newRelName),"w")
    regress.write(olddir)
    regress.close()

def _main():
    (oldpath,newpath) = getParameters()
    regressReports(oldpath,newpath,oldRelName=getVerFromLog(oldpath))

              
if __name__ == "__main__":
    _main()


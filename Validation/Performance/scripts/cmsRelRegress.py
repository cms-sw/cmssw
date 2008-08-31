#!/usr/bin/env python

import os, re, sys, glob
import optparse as opt
import cmsPerfRegress as cpr
from cmsPerfCommons import Candles, CandFname, getVerFromLog

def getParameters():
    global _debug
    parser = opt.OptionParser()
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

def compareLogPair(log,candle,step,prof,profdir,latrev,prevrev,oldRelName=""):
    base = os.path.basename(log)
    logdir = "%s_%s_%s" % (CandFname[candle],step,prof)
    if prof == "EdmSize":
        logdir = "%s_outdir" % logdir

    outd   = "%s/%s" % (latrev,logdir)
    rootf  = "regression.root" 
    oldLog = "%s/%s/%s" % (prevrev,profdir,base)
    newLog = "%s" % log
    if oldRelName == "":
        oldRelName = os.path.basename(prevrev)
        if not "CMSSW" in oldRelName:
            oldRelName = ""

    if _debug > 0:
        assert os.path.exists(newLog), "The current release logfile %s that we were using to perform regression analysis was not found (even though we just found it!!)" % newLog
#    print "regressCompare", rootf, outd, oldLog,newLog,1, oldRelName
    if   "TimingReport" in prof and os.path.exists(oldLog):
        # cmsPerfRegress(rootfilename, outdir, oldLogFile, newLogfile, secsperbin, batch, prevrev)
        try:
            htmNames = cpr.cmpTimingReport(rootf, outd, oldLog, newLog, 1, prevrev = oldRelName)
        except cpr.TimingParseErr, detail:
            print "WARNING: Could not parse data from log file %s; not performing regression" % detail.message
        else:
            print "Successfully compared %s and %s" % (oldLog,newLog)
            
    elif "TimingReport" in prof and not os.path.exists(oldLog):
        print "WARNING: Could not find an equivalent logfile for %s in the previous release dir %s" % (newLog,oldLog)                            

def regressTimingReport(olddir,newdir,oldRelName = ""):
    profSets = ["Valgrind", "IgProf", "TimeSize"]
    for candle in Candles:
        for profset in profSets:
            adir = "%s/%s_%s" % (newdir,candle,profset)
            if os.path.exists(adir):
                Profs = []
                if   profset == "Valgrind":
                    Profs = ["memcheck"]
                elif profset == "TimeSize":
                    Profs = [ "TimingReport",
                              "TimeReport",
                              "SimpleMemoryCheck",
                              "EdmSize"]
                elif profset == "IgProf":
                    Profs = [ "IgProf" ]

                for prof in Profs:
                    if prof == "EdmSize":
                        stepLogs = glob.glob("%s/%s_*_%s"       % (adir,CandFname[candle],prof))
                    else:
                        stepLogs = glob.glob("%s/%s_*_%s.log"   % (adir,CandFname[candle],prof))

                    profdir = os.path.basename(adir)
                    candreg = re.compile("%s_([^_]*)_%s(.log)?" % (CandFname[candle],prof))
                    for log in stepLogs:
                        base = os.path.basename(log)
                        searchob = candreg.search(base)
                        if searchob:
                            step = searchob.groups()[0]
                            compareLogPair(log,candle,step,prof,profdir,adir,olddir,oldRelName = oldRelName)
                        else:
                            continue

def _main():
    (oldpath,newpath) = getParameters()
    regressTimingReport(oldpath,newpath,oldRelName=getVerFromLog(oldpath))
    os.system("touch %s/REGRESSION.%s.vs.%s" % (newpath,getVerFromLog(oldpath),getVerFromLog(newpath)))
              
if __name__ == "__main__":
    _main()


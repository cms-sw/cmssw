#!/usr/bin/env python

import os, re, sys
import optparse as opt
import cmsPerfRegress as cpr
from cmsPerfCommons import Candles, CandFname

def getParameters():
    parser = opt.OptionParser()
    #
    # Options
    #   
    (options,args) = parser.parse_args()
    
    path1 = os.path.abspath(args[0])
    path2 = os.path.abspath(args[1])    
    if os.path.exists(path1) and os.path.exists(path2):
        return (path1, path2)
    else:
        print "Error: one of the paths does not exist"
        sys.exit()

def compareLogPair(candle,step,prof,profdir,prevrev):
    logdir = "%s_%s_%s" % (CandFname[candle],step,prof)
    if prof == "EdmSize":
        logdir = "%s_outdir" % logdir

    outd   = "%s/%s" % (adir,logdir)
    rootf  = "%s/regression.root" % outd
    oldLog = "%s/%s/%s" % (prevrev,profdir,base)
    newLog = "%s" % log
    oldRelName = os.path.basename(prevrev)
    if not "CMSSW" in oldRelName:
        oldRelName = ""

    if _debug > 0:
        assert os.path.exists(newLog), "The current release logfile %s that we were using to perform regression analysis was not found (even though we just found it!!)" % newLog

    if   "TimingReport" in prof and os.path.exists(oldLog):
        # cmsPerfRegress(rootfilename, outdir, oldLogFile, newLogfile, secsperbin, batch, prevrev)
        htmNames = cpr.regressCompare(rootf, outd, oldLog, newLog, 1, prevrev = oldRelName)
    elif "TimingReport" in prof and not os.path.exists(oldLog):
        print "WARNING: Could not find an equivalent logfile for %s in the previous release dir" % newLog                            

def compareLogs(olddir,newdir):
    profSets = ["Valgrind", "IgProf", "TimeSize"]
    for candle in Candles:
        for profset in ProfSets:
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
                            compareLogPair(candle,step,prof,profdir,olddir)
                        else:
                            continue

def _main():
    (oldpath,newpath) = getParameters()
    compareLogs(oldpath,newpath)

if __name__ == "__main__":
    _main()


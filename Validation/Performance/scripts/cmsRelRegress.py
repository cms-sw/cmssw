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

def compareSimMemPair(newLog,profdir,curdir,candle,olddir,oldRelName=""):
    if oldRelName == "":
        oldRelName = os.path.basename(olddir)
        if not "CMSSW" in oldRelName:
            oldRelName = ""    
    base = os.path.basename(newLog)
    oldlog = os.path.join(olddir,curdir,base)
    rootf  = "simpmem-regress.root"
    try:
        cpr.cmpSimpMemReport(rootf,curdir,oldlog,newLog,1,True,candle,prevrev = oldRelName)
    except cpr.SimpMemParseErr, detail:
        print "WARNING: Could not parse data from log file %s; not performing regression" % detail.message
    else:
        print "Successfully compared %s and %s" % (oldlog,newLog)        
        

def compareTimingLogPair(log,candle,step,prof,profdir,latrev,prevrev,oldRelName=""):
    base = os.path.basename(log)
    logdir = "%s_%s_%s" % (CandFname[candle],step,prof)
    if prof == "EdmSize":
        logdir = "%s_outdir" % logdir

    outd   = "%s/%s" % (latrev,logdir)
    rootf  = "timing-regress.root" 
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
            cpr.cmpTimingReport(rootf, outd, oldLog, newLog, 1, batch = True, prevrev = oldRelName)
        except cpr.TimingParseErr, detail:
            print "WARNING: Could not parse data from log file %s; not performing regression" % detail.message
        else:
            print "Successfully compared %s and %s" % (oldLog,newLog)
            
    elif "TimingReport" in prof and not os.path.exists(oldLog):
        print "WARNING: Could not find an equivalent logfile for %s in the previous release dir %s" % (newLog,oldLog)                            

def regressReports(olddir,newdir,oldRelName = ""):
    profSets = ["Valgrind", "IgProf", "TimeSize"]
    for candle in Candles:
        for profset in profSets:
            adir = os.path.join(newdir,"%s_%s" % (candle,profset))
            if os.path.exists(adir):
                Profs = []
                if   profset == "Valgrind":
                    Profs = ["valgrind"] # callgrind actually
                elif profset == "TimeSize":
                    Profs = [ "TimingReport",
                              "TimeReport",
                              "SimpleMemoryCheck",
                              "EdmSize"]
                if profset == "IgProf":
                    Profs = [ "IgProfMemTotal",
                              "IgProfMemLive"]
                    
                for prof in Profs:
                    if   prof == "EdmSize" or prof == "valgrind":
                        stepLogs = glob.glob("%s/%s_*_%s"       % (adir,CandFname[candle],prof))
                    elif prof == "IgProfMemLive" or prof == "IgProfMemTotal":
                        stepLogs = glob.glob("%s/%s_*_%s.gz"       % (adir,CandFname[candle],prof))                        
                    elif prof == "SimpleMemoryCheck":
                        stepLogs = os.path.join(adir,"%s.log" % candle)
                    elif prof == "TimingReport":
                        stepLogs = glob.glob("%s/%s_*_%s.log"   % (adir,CandFname[candle],prof))

                    profdir = os.path.basename(adir)


                    if prof == "TimingReport" or prof == "EdmSize" or prof == "valgrind" or prof == "IgProfMemTotal" or prof == "IgProfMemLive":
                        stepreg = re.compile("%s_([^_]*)_%s((.log)|(.gz))?" % (CandFname[candle],prof))
                        for log in stepLogs:
                            base = os.path.basename(log)
                            searchob = stepreg.search(base)
                            if searchob:
                                step = searchob.groups()[0]
                                if prof == "TimingReport": 
                                    compareTimingLogPair(log,candle,step,prof,profdir,adir,olddir,oldRelName = oldRelName)
                                elif prof == "EdmSize" or prof == "valgrind" or prof == "IgProfMemTotal" or prof == "IgProfMemSize":
                                    outpath = os.path.join(adir,"%s_regression" % base)
                                    oldlog  = os.path.join(olddir,"%s_%s" % (candle,profset),base)
                                    if not os.path.exists(outpath):
                                        os.mkdir(outpath)
                                    if os.path.exists(oldlog):
                                        try:
                                            try:
                                                if   prof == "EdmSize":
                                                    cpr.cmpEdmSizeReport(outpath,oldlog,log)
                                                elif prof == "valgrind":
                                                    cpr.cmpCallgrindReport(outpath,oldlog,log)
                                                elif prof == "IgProfMemTotal" or prof == "IgProfMemSize":
                                                    cpr.cmpIgProfReport(outpath,oldlog,log)
                                            except cpr.PerfReportErr,detail:
                                                print "WARNING: Perfreport return non-zero exit status when comparing %s and %s because perfreport" % (oldlog,log)
                                        finally:
                                            print "Successfully compared %s and %s" % (oldlog,log)                                            
                                    else:
                                        print "WARNING: Could not find an equivalent logfile for %s in the previous release dir %s " % (log,oldlog)
                                        
                                                            
                            else:
                                continue
                    elif prof == "SimpleMemoryCheck":
                        compareSimMemPair(stepLogs,candle,profdir,adir,olddir,oldRelName= oldRelName)


def _main():
    (oldpath,newpath) = getParameters()
    regressReports(oldpath,newpath,oldRelName=getVerFromLog(oldpath))
    os.system("touch %s/REGRESSION.%s.vs.%s" % (newpath,getVerFromLog(oldpath),getVerFromLog(newpath)))
              
if __name__ == "__main__":
    _main()


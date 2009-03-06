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
    if oldRelName == "":
        oldRelName = os.path.basename(adir)
        if not "CMSSW" in oldRelName:
            oldRelName = ""
    return oldRelName

def compareSimMemPair(newLog,profdir,curdir,candle,olddir,oldRelName=""):
    oldRelName = getOldRelName(oldRelName,olddir)
    base = os.path.basename(newLog)
    oldlog = os.path.join(olddir,curdir,base)
    rootf  = "simpmem-regress.root"
    try:
        cpr.cmpSimpMemReport(rootf,curdir,oldlog,newLog,1,True,candle,prevrev = oldRelName)
    except cpr.SimpMemParseErr, detail:
        print "WARNING: Could not parse data from log file %s; not performing regression" % detail.message
    else:
        print "Successfully compared %s and %s" % (oldlog,newLog)        
        
def regressReports(olddir,newdir,oldRelName = "",newRelName=""):
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
                            if prof == "IgProfMemTotal" or prof == "IgProfMemLive":
                                base = base.split(".gz")[0]
                            searchob = stepreg.search(base)
                            if searchob:
                                step = searchob.groups()[0]
                                outpath = os.path.join(adir,"%s_regression" % base)
                                oldlog  = os.path.join(olddir,"%s_%s" % (candle,profset),base)
                                if prof == "IgProfMemTotal" or prof == "IgProfMemLive":
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
                                            print "** Comparing", candle, step, prof, "previous release: %s and latest release: %s" % (oldlog,log)
                                            print "**"
                                            oldRelName = getOldRelName(oldRelName,olddir)
                                            
                                            cpr.cmpTimingReport(rootf, outd, oldlog, log, 1, batch = True, prevrev = oldRelName)
                                        elif prof == "valgrind":
                                            cpr.cmpCallgrindReport(outpath,oldlog,log)
                                        elif prof == "IgProfMemTotal" or prof == "IgProfMemSize":
                                            
                                            cpr.cmpIgProfReport(outpath,oldlog,log)
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
                        compareSimMemPair(stepLogs,candle,profdir,adir,olddir,oldRelName= oldRelName)
                    
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


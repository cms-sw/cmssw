#!/usr/bin/env python
from cmsPerfCommons import Candles, CandFname
import optparse as opt
import cmsPerfRegress as cpr
import sys, os, glob, re

_PROG_NAME = os.path.basename(sys.argv[0])

###################
# Parse options 
#
def optionParse():
    parser = opt.OptionParser(usage="""./%s [perf dir] [options]""" % _PROG_NAME)

    (options, args) = parser.parse_args()

    if not len(args) == 1:
        parser.error("You have not supplied a perfsuite directory.")
        sys.exit()

    args[0] = os.path.abspath(args[0])

    ###########
    # Check existance of the directory supplied
    #
    if not os.path.isdir(args[0]):
        parser.error("You have not provided a valid perfsuite output directory")
        sys.exit()

    perfdir = args[0]

    return (perfdir)

#################
# Harvest information from a timesize directory, currently only retrieves TimingReport information
#
def visit_timesize_steps(candle,profsetdir):
    out = {}
    # Just do timing report for now
    globpath = os.path.join(profsetdir,"%s_*_TimingReport.log" % CandFname[candle])
    globs = glob.glob(globpath)
    if len(globs) > 0:
        stepreg = re.compile("%s_([^_]*)_TimingReport.log" % CandFname[candle])
        for globule in globs:
            base = os.path.basename(globule)
            found = stepreg.search(base)
            if found:
                step = found.groups()[0]
                try:
                    if step == None:
                        print "Error: could not resolve step something is wrong"
                        step = "None"
                    if not out.has_key("TimingReport"):
                        out["TimingReport"] = {}
                    stepdict = out["TimingReport"]
                    stepdict[step] = cpr.getTimingLogData(globule)
                    out["TimingReport"] = stepdict
                except (OSError, IOError), detail:
                    print detail
            else:
                print "Error: Could not determine step from %s" % base
    return out

###############
# Retrieve data from a perf suite output (sub) directory, only examines TimeSize at the moment
#
def visit(visitdir):
    out = {}
    for candle in Candles:
        globpath = os.path.join(visitdir,"%s_*" % (candle))
        globs = glob.glob(globpath)
        if len(globs) > 0:
            profsetreg = re.compile("%s_(.*)" % candle)
            for globule in globs:
                base = os.path.basename(globule)
                found = profsetreg.search(base)
                if found:
                    profset = found.groups()[0]
                    if profset == "TimeSize": # Just do timesize for now!
                        if candle == None:
                            print "Error: could not resolve candle something is wrong"
                            candle = "None"
                        if profset == None:
                            print "Error: could not resolve profset something is wrong"
                            profset = "None"
                        if not out.has_key(candle):
                            out[candle] = {}
                        candledict = out[candle]
                        if candledict.has_key(profset):
                            print "Error: we already have a profset that matches %s" % str(profset)
                        else:
                            candledict[profset] = visit_timesize_steps(candle,globule)
                            out[candle] = candledict
    return out
        
###############
# Harvest all data from a perfsuite directory (if cpu subdirectories exist, it will examine them)
#
def harvest(perfdir):
    cpureg = re.compile("cpu_([0-9][0-9]*)")
    out = {}
    globpath = os.path.join(perfdir, "cpu_*")
    globs = glob.glob(globpath)
    if len(globs) > 0:
        for globule in globs:
            base  = os.path.basename(globule)
            found = cpureg.search(base)
            if found:
                cpuid = found.groups()[0]
                if out.has_key(cpuid):
                    print "Error: we already have a cpu run with this id %s ! Skipping..." % cpuid
                else:
                    if cpuid == None:
                        print "Error: could not resolve cpuid something is wrong"
                        cpuid = "None"
                    out[cpuid] = visit(globule)                
            else:
                print "Error: could not determine valid cpu id from %s ! Skipping..." % base

    else:
        out["None"] = visit(perfdir)
        
    return out  

if __name__ == "__main__":
    (perfdir) = optionParse()
    print harvest(perfdir)

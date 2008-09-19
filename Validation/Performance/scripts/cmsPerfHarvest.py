#!/usr/bin/env python
from cmsPerfCommons import Candles, CandFname
import optparse as opt
import cmsPerfRegress as cpr
import sys, os, glob, re

_PROG_NAME = os.path.basename(sys.argv[0])

def optionParse():
    parser = opt.OptionParser(usage="""./%s [perf dir] [outfile] [options]""" % _PROG_NAME)

    (options, args) = parser.parse_args()

    if not len(args) == 2:
        parser.error("You have not supplied an outfile or perfsuite directory, both are required")
        sys.exit()


    args[0] = os.path.abspath(args[0])
    args[1] = os.path.abspath(args[1])    

    if not os.path.isdir(args[0]):
        parser.error("You have not provided a valid perfsuite output directory")
        sys.exit()

    if os.path.exists(args[1]):
        parser.error("The output file you specified already exists")
        sys.exit()        

    perfdir = args[0]
    outfile = args[1]

    return (perfdir, outfile)

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
                    out[("TimingReport",step)] = cpr.getTimingLogData(globule)
                except (OSError, IOError), detail:
                    print detail
            else:
                print "Error: Could not determine step from %s" % base
    return out

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
                    key = found.groups()[0]
                    if key == "TimeSize": # Just do timesize for now!
                        if out.has_key((candle,key)):
                            print "Error: we already have a profset that matches %s" % str((candle,key))
                        else:
                            out[(candle,key)] = visit_timesize_steps(candle,globule)
    return out
        

def harvest(perfdir, outfile):
    cpureg = re.compile("cpu_([0-9][0-9]*)")
    out = {}
    globpath = os.path.join(perfdir, "cpu_*")
    globs = glob.glob(globpath)
    if len(globs) > 0:
        for globule in globs:
            base  = os.path.basename(globule)
            found = cpureg.search(base)
            if found:
                cpuid = int(found.groups()[0])
                if out.has_key(cpuid):
                    print "Error: we already have a cpu run with this id %s ! Skipping..." % cpuid
                else:
                    out[cpuid] = visit(globule)                
            else:
                print "Error: could not determine valid cpu id from %s ! Skipping..." % base

    else:
        out[None] = visit(perfdir)
        
    return out  

if __name__ == "__main__":
    (perfdir, outfile) = optionParse()
    print harvest(perfdir, outfile)

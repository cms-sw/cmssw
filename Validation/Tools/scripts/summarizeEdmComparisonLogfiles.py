#! /usr/bin/env python

import optparse
import os
from glob import glob
import re

if __name__ == "__main__":

    # compile regexs
    percentRE      = re.compile (r'%')
    startOutputRE  = re.compile (r'^problems$')
    success1RE     = re.compile (r"{'eventsCompared':\s+(\d+),\s+'count_(\S+)':\s+(\d+)\s*}")
    success2RE     = re.compile (r"{'count_(\S+)':\s+(\d+),\s+'eventsCompared':\s+(\d+)\s*}")
    labelErrorRE   = re.compile (r"labelDict = GenObject._ntupleDict\[tupleName\]\['_label'\]")
    terminatedRE   = re.compile (r'Terminated\s+\$EXE\s+\$@')
    cppExceptionRE = re.compile (r'\(C\+\+ exception\)')
    missingCfgRE   = re.compile (r"raise.+Can't open configuration")
    finishRE       = re.compile (r'finish')
    problemDict = { 'label'        : labelErrorRE,
                    'terminate'    : terminatedRE,
                    'cppException' : cppExceptionRE,
                    'missingCfg'   : missingCfgRE,
                    'finish'       : finishRE}

    parser = optparse.OptionParser ("Usage: %prog logfilePrefix [directory]")

    options, args = parser.parse_args()
    if not 1 <= len (args) <= 2:
        raise RuntimeError, "Must give directory and log file prefix"
    logfilePrefix = percentRE.sub ('*', args[0])
    if logfilePrefix[-1] != '*':
        logfilePrefix += '*'
    if len (args) == 2:
        os.chdir (args[1])
    files = glob (logfilePrefix)
    totalFiles = len (files)
    problems = {}
    succeeded = 0
    problemTypes = {}
    for log in files:
        source = open (log, 'r')
        ran = False
        success = False
        for line in source:
            if startOutputRE.search(line):
                ran = True
                continue
            if success1RE.search (line) or success2RE.search(line):
                success = True
                continue
            for key, regex in problemDict.iteritems():
                #print "considering %s for %s" % (key, line)
                if regex.search(line):
                    problems.setdefault(log,[]).append(key)
                    if not problemTypes.has_key(key):
                        problemTypes[key] = 1
                    else:
                        problemTypes[key] += 1
        source.close()
        if ran and success:
            succeeded += 1
        elif not problems.has_key (log):
            problems.setdefault(log,[]).extend ( ['other',
                                                  'ran:%s' % ran,
                                                  'success:%s' % success])
            pass

    print "total:   ", len (files)
    print "success: ", succeeded
    print "Problem types:"
    for key, value in (problemTypes.iteritems()):
        print "  %-15s: %2d" % (key, value)
    print "\nDetailed Problems list:"
    for key, problemList in sorted (problems.iteritems()):
        print "   %s:\n   %s\n" % (key, problemList)

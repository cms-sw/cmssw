#! /usr/bin/env python

import optparse
import os
from glob import glob
import re
import pprint
countRE = re.compile (r'^count_(\w+)')

def summaryOK (summary):
    """returns a tuple.  First value is true if summary hasn't found
    any problems, else false."""
    retval = True
    count    = -1
    compared = summary.get('eventsCompared', -1)
    if len( summary.keys()) != 2:
        retval = False
    for key,value in summary.iteritems():
        if countRE.search(key):
            count = value
    return (retval, {'count':count, 'compared':compared})

if __name__ == "__main__":

    # compile regexs
    percentRE      = re.compile (r'%')
    startOutputRE  = re.compile (r'^problems$')
    success1RE     = re.compile (r"{'eventsCompared':\s+(\d+),\s+'count_(\S+)':\s+(\d+)\s*}")
    success2RE     = re.compile (r"{'count_(\S+)':\s+(\d+),\s+'eventsCompared':\s+(\d+)\s*}")
    labelErrorRE   = re.compile (r"labelDict = GenObject._ntupleDict\[tupleName\]\['_label'\]")
    missingLabelRE = re.compile (r'not able to get')
    terminatedRE   = re.compile (r'Terminated\s+\$EXE\s+\$@')
    cppExceptionRE = re.compile (r'\(C\+\+ exception\)')
    missingCfgRE   = re.compile (r"raise.+Can't open configuration")
    finishRE       = re.compile (r'finish')
    dummyRE        = re.compile (r'edm::Wrapper<dummyType>')
    noEdmWrapperRE = re.compile (r"'ROOT' has no attribute 'edm::Wrapper")
    uint32RE       = re.compile (r"Config file parser error 'operatoruint32_t")
    nonSpacesRE    = re.compile (r'\S')
    problemDict = { 'labelDict'    : labelErrorRE,
                    'missingLabel' : missingLabelRE,
                    'terminate'    : terminatedRE,
                    'uint32'       : uint32RE,
                    'cppException' : cppExceptionRE,
                    'missingCfg'   : missingCfgRE,
                    'noEdmWrapper' : noEdmWrapperRE,
                    'dummy'        : dummyRE,
                    'finish'       : finishRE}

    parser = optparse.OptionParser ("Usage: %prog logfilePrefix [directory]")
    parser.add_option ("--counts", dest="counts",
                       action="store_true", default=False,
                       help="Display counts only.")
    parser.add_option ("--problem", dest="problem", type='string',
                       help="Displays problems matching PROBLEM")

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
    weird     = 0
    problemTypes = {}
    successes = {}
    for log in files:
        problemSet = set()
        source = open (log, 'r')
        ran = False
        success = False
        reading = False
        summaryLines = ''
        for line in source:
            line = line.rstrip('\n')
            if reading:
                if not nonSpacesRE.search(line):
                    reading = False
                    continue
                summaryLines += line
            if startOutputRE.search(line):
                ran     = True
                reading = True
                continue
            if success1RE.search (line) or success2RE.search(line):
                success = True
                continue
            for key, regex in problemDict.iteritems():
                #print "considering %s for %s" % (key, line)
                if regex.search(line):
                    if key in problemSet:
                        continue
                    problemSet.add (key)
                    problems.setdefault(log,[]).append(key)
                    if not problemTypes.has_key(key):
                        problemTypes[key] = 1
                    else:
                        problemTypes[key] += 1
            key = ''
        source.close()
        
        if summaryLines:
            summary = eval (summaryLines)
            ok = summaryOK (summary)
        else:
            ok = (False,)
            summary = None
        if ran and success:
            succeeded += 1
            if not ok[0]:
                weird += 1
            else:
                successes[log] = pprint.pformat (summary, indent=4)
        else:
            if ok[0]:
                weird += 1
        if not problems.has_key (log) and not ok[0]:
            if not ok[0] and summary:
                key = 'mismatch'
                problems[log] = pprint.pformat (summary, indent=4)
            else:
                problems[log] = ['other','ran:%s' % ran,
                                  'success:%s' % success]
                key = 'other'
            if not problemTypes.has_key(key):
                problemTypes[key] = 1
            else:
                problemTypes[key] += 1
    print "total:     ", len (files)
    print "success:   ", succeeded
    print "weird:     ", weird
    print "Problem types:"
    total = 0
    for key, value in sorted (problemTypes.iteritems()):
        print "  %-15s: %4d" % (key, value)
        total += value
    print " ", '-'*13, " : ----"
    print "  %-15s: %4d + %d = %d" \
          % ('total', total, succeeded, total + succeeded),
    if not options.counts:
        print "\nDetailed Problems list:"
        for key, problemList in sorted (problems.iteritems()):
            if options.problem and problemList[0] != options.problem:
                continue
            print "   %s:\n   %s\n" % (key, problemList)
        if not options.problem:
            print "\n", '='*78, '\n'
            print "Success list:"
            for key, successesList in sorted (successes.iteritems()):
                print "   %s:\n   %s\n" % (key, successesList)

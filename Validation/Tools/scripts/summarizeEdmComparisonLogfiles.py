#! /usr/bin/env python3

from __future__ import print_function
import optparse
import os
from glob import glob
import re
import pprint
import six
import commands
countRE = re.compile (r'^count_(\w+)')
avoid = ['index', 'print']

def summaryOK (summary):
    """returns a tuple.  First value is true if summary hasn't found
    any problems, else false."""
    retval = True
    count    = -1
    compared = summary.get('eventsCompared', -1)
    if len( summary) != 2:
        retval = False
    for key,value in six.iteritems(summary):
        if countRE.search(key):
            count = value
    return (retval, {'count':count, 'compared':compared})

if __name__ == "__main__":

    # compile regexs
    percentRE      = re.compile (r'%')
    startOutputRE  = re.compile (r'^Summary$')
    success1RE     = re.compile (r"{'eventsCompared':\s+(\d+),\s+'count_(\S+)':\s+(\d+)\s*}")
    success2RE     = re.compile (r"{'count_(\S+)':\s+(\d+),\s+'eventsCompared':\s+(\d+)\s*}")
    loadingSoRE    = re.compile (r'loading (genobjectrootlibs/\w+)')
    creatingSoRE   = re.compile (r'creating shared library (\S+)')
    compRootRE     = re.compile (r' --compRoot=(\S+)')
    descriptionRE  = re.compile (r'^edmOneToOneComparison.py (\w+).txt')
    edmCommandRE  = re.compile (r'^(edmOneToOneComparison.py .+?)\s*$')
    # problem regexs
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
                    'operator'     : re.compile (r"onfig file parser error 'operator"),
                    'useless'      : re.compile (r'no member functions that are useful'),
                    'lazy'         : re.compile (r': Assertion'),
                    'detset'       : re.compile (r"AttributeError: 'edm::DetSet"),
                    'doubleint'    : re.compile (r'AttributeError: (int|double)'),
                    'finish'       : finishRE}

    parser = optparse.OptionParser ("Usage: %prog logfilePrefix [directory]")
    parser.add_option ("--counts", dest="counts",
                       action="store_true", default=False,
                       help="Display counts only.")
    parser.add_option ('--mismatch', dest='mismatch',
                       action='store_true',
                       help='Displays only mismatch output')
    parser.add_option ("--diffTree", dest="diffTree",
                       action="store_true", default=False,
                       help="Shows diffTree printout.")
    parser.add_option ('--makeCompRoot', dest='makeCompRoot',
                       action='store_true',
                       help='Prints commands to make compRoot files for difftree')
    parser.add_option ("--problem", dest="problem", type='string',
                       help="Displays problems matching PROBLEM")

    options, args = parser.parse_args()
    if not 1 <= len (args) <= 2:
        raise RuntimeError("Must give directory and log file prefix")
    logfilePrefix = percentRE.sub ('*', args[0])
    if logfilePrefix[-1] != '*':
        logfilePrefix += '*'
    cwd = os.getcwd()
    logdir = ''
    if len (args) == 2:
        logdir = args[1]
        os.chdir (logdir)
    files        = glob (logfilePrefix)
    if logdir:
        oldFiles = files
        files = []
        for filename in oldFiles:
            files.append (logdir + '/' + filename)
        os.chdir (cwd)
    totalFiles   = len (files)
    problems     = {}
    succeeded    = 0
    weird        = 0
    problemTypes = {}
    successes    = {}
    objectName   = ''
    compRoot     = ''
    soName       = ''
    command      = ''
    diffOutput   = {}
    goShlib      = ''
    for log in files:
        problemSet = set()
        source = open (log, 'r')
        ran = False
        success = False
        reading = False
        summaryLines = ''
        for line in source:
            line = line.rstrip('\n')
            match = edmCommandRE.search (line)
            if match:
                command = match.group(1)
            match = loadingSoRE.search (line)
            if match:
                goShlib = match.group(1)                
            match = creatingSoRE.search (line)
            if match:
                goShlib = match.group(1)                
            if options.diffTree:
                match = descriptionRE.search (line)
                if match:
                    objectName = match.group(1)
                match = compRootRE.search (line)
                if match:
                    compRoot = match.group(1)
                match = loadingSoRE.search (line)
                if match:
                    soName = match.group(1)
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
            for key, regex in six.iteritems(problemDict):
                #print "considering %s for %s" % (key, line)
                if regex.search(line):
                    if key in problemSet:
                        continue
                    problemSet.add (key)
                    problems.setdefault(log,[]).append(key)
                    if key not in problemTypes:
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
        if log not in problems and not ok[0]:
            if not ok[0] and summary:
                key = 'mismatch'                
                problems[log] = pprint.pformat (summary, indent=4)
                #pprint.pprint (summary, indent=4)
                if objectName and compRoot and soName:
                    # do the diffTree magic
                    varNames = summary.get(objectName, {}).\
                               get('_var', {}).keys()
                    variables = ['eta', 'phi']
                    for var in sorted (varNames):
                        if var not in variables and var not in avoid:
                            variables.append (var)
                    diffCmd = 'diffTreeTool.py --skipUndefined %s %s %s' \
                              % (compRoot, soName, " ".join(variables))
                    # print diffCmd
                    diffOutput[log] = diffCmd
            else:
                problems[log] = ['other','ran:%s' % ran,
                                  'success:%s' % success]
                key = 'other'
            if key not in problemTypes:
                problemTypes[key] = 1
            else:
                problemTypes[key] += 1
    mismatches = problemTypes.get('mismatch', 0)
    if 'mismatch' in problemTypes:
        del problemTypes['mismatch']
    print("total:      ", len (files))
    print("success:    ", succeeded)
    print("mismatches: ", mismatches)
    print("weird:      ", weird)
    print("Tool issue types:")
    total = 0
    for key, value in sorted (six.iteritems(problemTypes)):
        print("  %-15s: %4d" % (key, value))
        total += value
    print(" ", '-'*13, " : ----")
    print("  %-15s: %4d + %d + %d + %d = %d" \
          % ('total', total, succeeded, mismatches, weird,
             total + succeeded + mismatches + weird))
    
    if not options.counts:
        print("\nDetailed Problems list:")
        for key, problemList in sorted (six.iteritems(problems)):
            if options.problem and problemList[0] != options.problem:
                continue
            if options.mismatch and not isinstance (problemList, str):
                continue
            #if options.mismatch and 
            print("   %s:\n   %s\n" % (key, problemList))
            if options.mismatch and goShlib and compRoot:
                print("diffTree %s %s" % (goShlib, compRoot))
            diffCmd = diffOutput.get(key)
            if diffCmd:                
                print(commands.getoutput (diffCmd))
        if not options.problem and not options.mismatch:
            print("\n", '='*78, '\n')
            print("Success list:")
            for key, successesList in sorted (six.iteritems(successes)):
                print("   %s:\n   %s\n" % (key, successesList))

#! /usr/bin/env python3

from __future__ import print_function
import optparse
import commands
import pprint
import re
import os
import sys
import six

piecesRE     = re.compile (r'(.+?)\s+"(\S+)"\s+"(\S*)"\s+"(\S+)"')
#colonRE      = re.compile (r':+')
nonAlphaRE   = re.compile (r'\W')
commaRE      = re.compile (r',')
queueCommand = '/uscms/home/cplager/bin/clpQueue.pl addjob %s'
logDir       = 'logfiles'
compRootDir  = 'compRoot'
# Containers
#vectorRE       = re.compile (r'^vector<(\S+)>')
doubleRE       = re.compile (r'^(double|int)')
vectorRE       = re.compile (r'^vector<([^<>]+)>')
detSetVecRE    = re.compile (r'^edm::DetSetVector<([^<>]+)>')
edColRE        = re.compile (r'^edm::EDCollection<([^<>]+)>')
sortedColRE    = re.compile (r'^edm::SortedCollection<([^<>]+),\S+?> >')
singletonRE    = re.compile (r'^([\w:]+)$')
containerList  = [vectorRE, detSetVecRE, edColRE, sortedColRE, doubleRE]

class EdmObject (object):

    def __init__ (self, tup):
        self.container, self.one, self.two, self.three = tup
        self.bool = False
        for regex in containerList:
            match = regex.search( self.container)
            if match:
                self.bool = True
                self.name = match.group(1)
                break

    def __str__ (self):
        return pprint.pformat (self.__dict__)

    def __bool__ (self):
        return self.bool

    def label (self):
        return "%s,%s,%s" % (self.one, self.two, self.three)


if __name__ == "__main__":

    ###################
    ## Setup Options ##
    ###################
    parser = optparse.OptionParser ("%prog [options] file1.root [file2.root]"\
                                    "\nFor more details, see\nhttps://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsToolsEdmOneToOneComparison")
    describeGroup  = optparse.OptionGroup (parser, "Description Options")
    precisionGroup = optparse.OptionGroup (parser, "Precision Options")
    summaryGroup   = optparse.OptionGroup (parser, "Summary Options")
    queueGroup     = optparse.OptionGroup (parser, "Queue Options")
    verboseGroup   = optparse.OptionGroup (parser, "Verbose Options")
    # general optionos
    parser.add_option ("--compRoot", dest="compRoot",
                       action="store_true", default=False,
                       help="Make compRoot files.")
    parser.add_option ('--strictPairing', dest='strictPairing',
                       action='store_true',
                       help="Objects are paired uniquely by order in collection")
    parser.add_option ("--prefix", dest="prefix", type="string",
                       help="Prefix to prepend to logfile name")
    parser.add_option ("--regex", dest='regex', action="append",
                       type="string", default=[],
                       help="Only run on branches containing regex")
    # describe options
    describeGroup.add_option ("--describeOnly", dest="describeOnly",
                              action="store_true", default=False,
                              help="Run description step only and stop.")
    describeGroup.add_option ("--forceDescribe", dest="forceDescribe",
                              action="store_true", default=False,
                              help="Run description step even if "\
                              "file already exists.")
    describeGroup.add_option ("--singletons", dest="singletons",
                              action="store_true", default=False,
                              help="Describe singleton objects (" \
                              "used only with --describeOnly option).")
    describeGroup.add_option ("--privateMemberData", dest="privateMemberData",
                              action="store_true", default=False,
                              help="include private member data "\
                              "(NOT for comparisons)")
    # precision options
    precisionGroup.add_option ("--precision", dest="precision", type="string",
                               help="Change precision use for floating "\
                               "point numbers")
    precisionGroup.add_option ('--absolute', dest='absolute',
                               action='store_true', default=False,
                               help='Precision is checked against '\
                               'absolute difference')
    precisionGroup.add_option ('--relative', dest='relative',
                               action='store_true', default=False,
                               help='Precision is checked against '\
                               'relative difference')
    # summary options
    summaryGroup.add_option ("--summary", dest="summary",
                             action="store_true", default=False,
                             help="Print out summary counts at end")
    summaryGroup.add_option ("--summaryFull", dest="summaryFull",
                             action="store_true", default=False,
                             help="Print out full summary at end (VERY LONG)")
    # queue options
    queueGroup.add_option ("--noQueue", dest="noQueue",
                           action="store_true", default=True,
                           help="Do not use queue, but run "\
                           "jobs serially (default).")
    queueGroup.add_option ("--queue", dest="noQueue",
                           action="store_false",
                           help="Use defaultqueueing system.")
    queueGroup.add_option ("--queueCommand", dest="queueCommand", type="string",
                           help="Use QUEUECOMMAND TO queue jobs")
    # verbose options
    verboseGroup.add_option ("--verbose", dest="verbose",
                             action="store_true", default=False,
                             help="Verbose output")
    verboseGroup.add_option ("--verboseDebug", dest="verboseDebug",
                             action="store_true", default=False,
                             help="Verbose output for debugging")
    parser.add_option_group (describeGroup)
    parser.add_option_group (precisionGroup)
    parser.add_option_group (summaryGroup)
    parser.add_option_group (queueGroup)
    parser.add_option_group (verboseGroup)
    options, args = parser.parse_args()
    # Because Root and PyRoot are _really annoying_, you have wait to
    # import this until command line options are parsed.
    from Validation.Tools.GenObject import GenObject
    if len (args) < 1 or len (args) > 2:
        raise RuntimeError("You must provide 1 or 2 root files")

    ###########################
    ## Check Queuing Options ##
    ###########################
    if options.queueCommand:
        queueCommand = options.queueCommand
        options.noQueue = False
        if not re.match (r'%%s', queueCommand):
            queueCommand += ' %s'
    if options.noQueue:
        command = 'src/Validation/Tools/scripts/runCommand.bash'
    else:
        command = 'src/Validation/Tools/scripts/runCMScommand.bash'
        # make sure we aren't trying to use options that should not be
        # used with the queueing system
        if options.compRoot or options.summary or options.summaryFull:
            raise RuntimeError("You can not use --compRoot or --summary "\
                  "in --queue mode")

    ##############################
    ## Make Sure CMSSW is Setup ##
    ##############################
    base         = os.environ.get ('CMSSW_BASE')
    release_base = os.environ.get ('CMSSW_RELEASE_BASE')
    if not base or not release_base:
        raise RuntimeError("You must have already setup a CMSSW environment.")
    # find command
    found = False
    for directory in [base, release_base]:
        fullCommand = directory + '/' + command
        if os.path.exists (fullCommand):
            found = True
            break
    if not found:
        raise RuntimeError("Can not find %s" % command)
    if not options.noQueue:
        fullCommand = queueCommand % fullCommand
    if not os.path.isdir (logDir):
        os.mkdir (logDir)
        if not os.path.isdir (logDir):
            raise RuntimeError("Can't create %s directory" % logDir)
    if options.compRoot and not os.path.isdir (compRootDir):
        os.mkdir (compRootDir)
        if not os.path.isdir (compRootDir):
            raise RuntimeError("Can't create %s directory" % compRootDir)
    logPrefix      = logDir      + '/'
    compRootPrefix = compRootDir + '/'
    if options.prefix:
        logPrefix += options.prefix + '_'
    currentDir = os.getcwd()
    filename1 = args[0]
    if len (args) == 2:
        filename2 = args[1]
    else:
        filename2 = filename1
    if not os.path.exists (filename1) or not os.path.exists (filename2):
        raise RuntimeError("Can not find '%s' or '%s'" % (filename1, filename2))
    # if verboseDebug is set, set verbose as well
    if options.verboseDebug:
        options.verbose = True
    if options.verbose:
        print("files", filename1, filename2)
    if options.singletons and not options.describeOnly:
        raise RuntimeError("--singletons can only be used with "\
              "--describeOnly option")
    if options.privateMemberData and not options.describeOnly:
        raise RuntimeError("--privateMemberData can only be used with "\
              "--describeOnly option")
    if options.singletons:
        containerList.append (singletonRE)

    #############################
    ## Run edmDumpEventContent ##
    #############################
    print("Getting edmDumpEventContent output")
    regexLine = ""
    for regex in options.regex:
        regexLine += ' "--regex=%s"' % regex
    dumpCommand = 'edmDumpEventContent %s %s' % (regexLine, filename1)
    if options.verboseDebug:
        print(dumpCommand, '\n')
    output = commands.getoutput (dumpCommand).split("\n")
    if not len(output):
        raise RuntimeError("No output from edmDumpEventContent.")
    if options.verboseDebug:
        print("output:\n", "\n".join(output))
    collection = {}
    total = failed = skipped = useless = 0
    for line in output:
        total += 1
        match = piecesRE.search(line)
        if match:
            obj = EdmObject( match.group(1,2,3,4) )
            if obj.bool:
                collection.setdefault( obj.container, [] ).append(obj)
            else:
                skipped += 1
        else:
            skipped += 1

   #########################################
   ## Run useReflexToDescribeForGenObject ##
   #########################################
    for key, value in sorted (six.iteritems(collection)):
        name      = value[0].name
        prettyName = nonAlphaRE.sub('', name)
        descriptionName = prettyName + '.txt'
        if os.path.exists (descriptionName) \
               and os.path.getsize (descriptionName) \
               and not options.forceDescribe:
            if options.verbose:
                print('%s exists.  Skipping' % descriptionName)
            continue
        #print name, prettyName, key
        describeCmd = "%s %s %s useReflexToDescribeForGenObject.py %s '--type=%s'" \
                  % (fullCommand, currentDir, logPrefix + prettyName,
                     GenObject.encodeNonAlphanumerics (name),
                     #name,
                     GenObject.encodeNonAlphanumerics (key))
        if options.precision:
            describeCmd += " --precision=" + options.precision
        if options.verbose:
            print("describing %s" % name)
        if options.verboseDebug:
            print(describeCmd, '\n')
        returnCode = os.system (describeCmd)
        if returnCode:
            # return codes are shifted by 8 bits:
            if returnCode == GenObject.uselessReturnCode << 8:
                useless += 1
            else:
                print("Error trying to describe '%s'.  Continuing.\n" % \
                      (name))
                failed += 1
    if options.describeOnly:
        print("Total: %3d  Skipped: %3d   Failed: %3d  Useless: %3d" % \
              (total, skipped, failed, useless))
        if not options.noQueue:
            print("Note: Failed not recorded when using queuing system.")
        sys.exit()

    ##################################
    ## Run edmOneToOneComparison.py ##
    ##################################
    for key, value in sorted (six.iteritems(collection)):
        #print "%-40s" % key,
        for obj in value:
            # print "  ", obj.label(),
            name = obj.name
            prettyName = nonAlphaRE.sub('', name)
            scriptName = 'edmOneToOneComparison.py'
            if prettyName in ['int', 'double']:
                scriptName = 'simpleEdmComparison.py'
            prettyLabel = commaRE.sub ('_', obj.label())
            compareCmd = '%s %s %s %s --compare --label=reco^%s^%s' \
                          % (scriptName,
                             prettyName + '.txt',
                             filename1,
                             filename2,
                             prettyName,
                             obj.label())
            fullCompareCmd = '%s %s %s %s' \
                             % (fullCommand, currentDir,
                                logPrefix + prettyName + '_' + prettyLabel,
                                compareCmd)
            if options.relative:
                fullCompareCmd += ' --relative'
            elif options.absolute:
                fullCompareCmd += ' --absolute'
            if options.compRoot:
                compRootName = compRootPrefix + prettyName \
                               + '_' + prettyLabel + '.root'
                fullCompareCmd += ' --compRoot=%s' % compRootName
            if options.strictPairing:
                fullCompareCmd += ' --strictPairing'
            if options.verbose:
                print("comparing branch %s %s" % (name, obj.label()))
            if options.verboseDebug:
                print(fullCompareCmd,'\n')
            os.system (fullCompareCmd)

    ##################################
    ## Print Summary (if requested) ##
    ##################################
    if options.summary or options.summaryFull:
        if options.prefix:
            summaryMask = options.prefix + '_%_'
        else:
            summaryMask = '%_'
        if options.summaryFull:
            summaryOptions = '--diffTree'
        else:
            summaryOptions = '--counts'
        summaryCmd = 'summarizeEdmComparisonLogfiles.py %s %s logfiles' \
                     % (summaryOptions, summaryMask)
        print(summaryCmd)
        print(commands.getoutput (summaryCmd))

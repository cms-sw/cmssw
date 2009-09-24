#! /usr/bin/env python

import optparse
import commands
import pprint
import re
import os
import sys

piecesRE = re.compile (r'(.+?)\s+"(\S+)"\s+"(\S*)"\s+"(\S+)\."')
vectorRE = re.compile (r'^vector<(\S+)>')
colonRE  = re.compile (r':+')
commaRE  = re.compile (r',')
queueCommand = '/uscms/home/cplager/bin/clpQueue.pl addjob %s'

class EdmObject (object):

    def __init__ (self, tup):
        self.container, self.one, self.two, self.three = tup
        vecMatch = vectorRE.search( self.container )
        if vecMatch:
            self.bool = True
            self.name = vecMatch.group(1)
        else:
            self.bool = False

    def __str__ (self):        
        return pprint.pformat (self.__dict__)

    def __bool__ (self):
        return self.bool

    def label (self):
        return "%s,%s,%s" % (self.one, self.two, self.three)


if __name__ == "__main__":

    parser = optparse.OptionParser ("Usage: %prog edmFile.root")
    parser.add_option ("--forceDescribe", dest="forceDescribe",
                       action="store_true", default=False,
                       help="Run description step even if file already exists.")
    parser.add_option ("--describeOnly", dest="describeOnly",
                       action="store_true", default=False,
                       help="Run description step only and stop.")
    parser.add_option ("--verbose", dest="verbose",
                       action="store_true", default=False,
                       help="Verbose output.")

    options, args = parser.parse_args()
    if len (args) < 1:
        raise RuntimeError, "You must provide a root file"
    currentDir = os.getcwd()
    #filename = os.path.abspath( args[0] )
    filename = args[0]
    print "filename", filename
    output = commands.getoutput ("edmDumpEventContent %s" % filename)\
             .split("\n")

    collection = {}
    for line in output:
        match = piecesRE.search(line)
        if match:
            obj = EdmObject( match.group(1,2,3,4) )
            if obj.bool:
                collection.setdefault( obj.container, [] ).append(obj)
    #pprint.pprint(collection)
    for key, value in sorted (collection.iteritems()):
        name      = value[0].name
        prettyName = colonRE.sub('', name)
        descriptionName = prettyName + '.txt'
        if os.path.exists (descriptionName) and not options.forceDescribe:
            if options.verbose:
                print '%s exists.  Skipping' % descriptionName
            continue
        # print name, prettyName
        describeCmd = '/uscms/home/cplager/work/cmssw/CMSSW_3_3_0_pre3/src/Validation/Tools/scripts/runCMScommand.bash %s %s describeReflexForGenObject.py --index %s "\\\"--type=%s\\\""' \
                  % (currentDir, prettyName, name, key)
        os.system (queueCommand % describeCmd)
        print describeCmd, '\n'

    if options.describeOnly:
        sys.exit()

    for key, value in sorted (collection.iteritems()):
        #print "%-40s" % key,
        for obj in value:
            # print "  ", obj.label(),
            name = obj.name
            prettyName = colonRE.sub('', name)
            prettyLabel = commaRE.sub ('_', obj.label())
            compareCmd = 'edmOneToOneComparison.py --config=%s --file=%s --tuple=reco --compare --label=reco^%s^%s' \
                          % (prettyName + '.txt',
                             filename,
                             prettyName,
                             obj.label())
            fullCompareCmd = '/uscms/home/cplager/work/cmssw/CMSSW_3_3_0_pre3/src/Validation/Tools/scripts/runCMScommand.bash %s %s %s' \
                             % (currentDir, prettyName + '_' + prettyLabel,
                                compareCmd)
            #print fullCompareCmd,'\n'
            os.system (queueCommand % fullCompareCmd)

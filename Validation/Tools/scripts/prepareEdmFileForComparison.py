#! /usr/bin/env python

import optparse
import commands
import pprint
import re
import os

piecesRE = re.compile (r'(.+?)\s+"(\S+)"\s+"(\S*)"\s+"(\S+)\."')
vectorRE = re.compile (r'^vector<(\S+)>')

class EdmObject (object):

    def __init__ (self, tup):
        self.variable, self.one, self.two, self.three = tup
        vecMatch = vectorRE.search( self.variable )
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
        return "'%s,%s,%s'" % (self.one, self.two, self.three)


if __name__ == "__main__":

    parser = optparse.OptionParser ("Usage: %prog edmFile.root")
    options, args = parser.parse_args()
    if len (args) < 1:
        raise RuntimeError, "You must provide a root file"

    filename = os.path.abspath( args[0] )
    print "filename", filename
    output = commands.getoutput ("edmDumpEventContent %s" % filename).split("\n")

    collection = {}
    for line in output:
        match = piecesRE.search(line)
        if match:
            obj = EdmObject( match.group(1,2,3,4) )
            if obj.bool:
                collection.setdefault( obj.variable, [] ).append(obj)
    #pprint.pprint(collection)
    for key, value in sorted (collection.iteritems()):
        print "%-40s" % key,
        for obj in value:
            print "  ", obj.label(),
        print

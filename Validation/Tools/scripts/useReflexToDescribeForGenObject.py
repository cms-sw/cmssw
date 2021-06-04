#!/usr/bin/env python3

from __future__ import print_function
from builtins import range
import ROOT
import re
import pprint
import sys
import inspect
import optparse

defsDict = {
    'int'    : '%-40s : form=%%%%8d     type=int',
    'float'  : '%-40s : form=%%%%7.2f   prec=',
    'str'    : '%-40s : form=%%%%20s    type=string',
    'long'   : '%-40s : form=%%%%10d    type=long',
    }

root2GOtypeDict = {
    'int'                      : 'int',
    'float'                    : 'float',
    'double'                   : 'float',
    'long'                     : 'long',
    'long int'                 : 'long',
    'unsigned int'             : 'int',
    'bool'                     : 'int',
    'string'                   : 'str',
    'std::basic_string<char>'  : 'str',
    }

startString = """
# -*- sh -*- For Font lock mode

###########################
## GenObject Definitions ##
###########################

# GenObject 'event' definition
[runevent singleton]
run:   type=int
event: type=int
"""

defTemplate = """
#####################
## %(OBJS)s Definition ##
#####################

# Nickname and Tree
[%(objs)s:FWLite]

# 'reco'-tupe 'runevent' 'tofill' information
[runevent:%(objs)s:EventAuxiliary shortcut=eventAuxiliary()]
run:   run()
event: event()

"""

colonRE        = re.compile (r':')
dotRE          = re.compile (r'\.')
nonAlphaRE     = re.compile (r'\W')
alphaRE        = re.compile (r'(\w+)')
vetoedTypes    = set()

def getObjectList (objectName, base, verbose = False, memberData = False):
    """Get a list of interesting things from this object"""
    # The autoloader needs an object before it loads its dictionary.
    # So let's give it one.
    try:
        rootObjConstructor = getattr (ROOT, objectName)
    except AttributeError as missingAttr:
        if str(missingAttr) in ['double', 'int']:
            print("Do not need to describe doubles or ints")
            sys.exit(0)
        else:
            raise

    obj = rootObjConstructor()
    alreadySeenFunction = set()
    vetoedFunction = set()
    etaFound, phiFound = False, False
    global vetoedTypes
    retval = []
    # Put the current class on the queue and start the while loop
    classList = [ ROOT.TClass.GetClass(objectName) ]
    if verbose: print(classList)
    # Uses while because reflixList is really a stack
    while classList:
        alreadySeenFunction.update(vetoedFunction) # skip functions hidden by derived class
        vetoedFunction.clear()
        oneclass = classList.pop (0) # get first element
        print("Looking at %s" % oneclass.GetName ())
        bases = oneclass.GetListOfBases()
        funcs = oneclass.GetListOfMethods()
        if verbose:
            print("baseSize", bases.GetSize())
            print("FunctionMemberSize", funcs.GetSize())
        for baseIndex in range( bases.GetSize() ) :
            classList.append( bases.At(baseIndex).GetClassPointer() )
        for index in range( funcs.GetSize() ):
            funcMember = funcs.At (index)
            # if we've already seen this, don't bother again
            name = funcMember.GetName()
            if verbose:
                print("name", name)
            if name == 'eta':
                etaFound = True
            elif name == 'phi':
                phiFound = True
            if name in alreadySeenFunction:
                continue
            # make sure this is an allowed return type
            returnType = funcMember.GetReturnTypeName()
            goType     = root2GOtypeDict.get (returnType, None)
            if verbose:
                print("   type", returnType, goType)
            if not goType:
                vetoedTypes.add (returnType)
                if verbose:
                    print("     skipped")
                continue
            elif verbose:
                print("     good")
            # only bother printout out lines where it is a const function
            # and has no input parameters.
            if funcMember.Property() & ROOT.kIsConstMethod and not funcMember.GetNargs():
                retval.append( ("%s.%s()" % (base, name), goType))
                alreadySeenFunction.add( name )
                if verbose:
                    print("     added")
            else :
                vetoedFunction.add( name )
                if verbose:
                    print("      failed IsConst() and GetNargs()")
        if not memberData:
            continue
        dataList = oneclass.GetListOfDataMembers()
        for index in range( dataList.GetSize() ):
            data = dataList.At( index );
            name = data.GetName()
            dataType = data.GetTypeName()
            goType = root2GOtypeDict.get (dataType, None)
            if not goType:
                continue
            if verbose:
                print("name", name, "dataType", dataType, "goType", goType)
            retval.append ( ("%s.%s" % (base, name), goType) )
    retval.sort()
    return retval, etaFound and phiFound


def genObjNameDef (line):
    """Returns GenObject name and ntuple definition function"""
    words = dotRE.split (line)[1:]
    func = ".".join (words)
    name =  "_".join (words)
    name = nonAlphaRE.sub ('', name)
    return name, func


def genObjectDef (mylist, tuple, alias, label, type, etaPhiFound):
    """Does something, but I can't remembrer what... """
    print("tuple %s alias %s label %s type %s" % (tuple, alias, label, type))
    # first get the name of the object
    firstName = mylist[0][0]
    match = alphaRE.match (firstName)
    if not match:
        raise RuntimeError("firstName doesn't parse correctly. (%s)" \
              % firstName)
    genName = match.group (1)
    genDef =  " ## GenObject %s Definition ##\n[%s]\n" % \
             (genName, genName)
    if options.index or not etaPhiFound:
        # either we told it to always use index OR either eta or phi
        # is missing.
        genDef += "-equiv: index,0\n";
    else:
        genDef += "-equiv: eta,0.1 phi,0.1 index,100000\n";
    tupleDef = '[%s:%s:%s label=%s type=%s]\n' % \
               (genName, tuple, alias, label, type)

    for variable in mylist:
        name, func = genObjNameDef (variable[0])
        typeInfo   = variable[1]
        form = defsDict[ typeInfo ]
        genDef   += form % name + '\n'
        tupleDef += "%-40s : %s\n" % (name, func)
    return genDef, tupleDef


if __name__ == "__main__":
    # Setup options parser
    parser = optparse.OptionParser \
             ("usage: %prog [options]  objectName\n" \
              "Creates control file for GenObject.")
    parser.add_option ('--goName', dest='goName', type='string',
                       default='',
                       help='GenObject name')
    parser.add_option ('--index', dest='index', action='store_true',
                       help='use index for matching')
    parser.add_option ('--label', dest='label', type='string',
                       default = 'dummyLabel',
                       help="Tell GO to set an label")
    parser.add_option ('--output', dest='output', type='string',
                       default = '',
                       help="Output (Default 'objectName.txt')")
    parser.add_option ('--precision', dest='precision', type='string',
                       default = '1e-5',
                       help="precision to use for floats (default %default)")
    parser.add_option ('--privateMemberData', dest='privateMemberData',
                       action='store_true',
                       help='include private member data (NOT for comparisons)')
    parser.add_option ('--tupleName', dest='tupleName', type='string',
                       default = 'reco',
                       help="Tuple name (default '%default')")
    parser.add_option ('--type', dest='type', type='string',
                       default = 'dummyType',
                       help="Tell GO to set an type")
    parser.add_option ('--verbose', dest='verbose', action='store_true',
                       help='Verbose output')
    options, args = parser.parse_args()
    defsDict['float'] += options.precision
    from Validation.Tools.GenObject import GenObject
    options.type = GenObject.decodeNonAlphanumerics (options.type)
    if len (args) < 1:
        raise RuntimeError("Need to provide object name.")
    #
    objectName = GenObject.decodeNonAlphanumerics (args[0])
    goName     = options.goName or colonRE.sub ('', objectName)
    outputFile = options.output or goName + '.txt'
    ROOT.gROOT.SetBatch()
    # load the right libraries, etc.
    ROOT.gSystem.Load("libFWCoreFWLite")
    ROOT.gSystem.Load("libDataFormatsFWLite")
    #ROOT.gSystem.Load("libReflexDict")
    ROOT.FWLiteEnabler.enable()
    mylist, etaPhiFound = getObjectList (objectName, goName, options.verbose,
                                         options.privateMemberData)
    if not len (mylist):
        print("There are no member functions that are useful for comparison.")
        sys.exit (GenObject.uselessReturnCode)
    targetFile = open (outputFile, 'w')
    genDef, tupleDef = genObjectDef (mylist,
                                     options.tupleName,
                                     goName,
                                     options.label,
                                     options.type,
                                     etaPhiFound)
    targetFile.write (startString)
    targetFile.write (genDef)
    targetFile.write (defTemplate % {'objs':'reco', 'OBJS':'RECO'})
    targetFile.write (tupleDef)
    print("Vetoed types:")
    pprint.pprint ( sorted( list(vetoedTypes) ) )

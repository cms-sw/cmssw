#!/usr/bin/env python

import ROOT
import PhysicsTools.PythonAnalysis as cmstools
import re
import pprint
import sys
import inspect
import optparse

defsDict = {
    'int'    : '%-40s : form=%%%%8d   type=int',
    'float'  : '%-40s : form=%%%%7.2f prec=0.001',
    'str'    : '%-40s : form=%%%%20s  type=string',
    'long'   : '%-40s : form=%%%%10d  type=long',    
    }

dotRE   = re.compile (r'\.')
nonAlphaRE = re.compile (r'\W')
alreadySeenSet = set( )

def warn (*args):
    """print out warning with line number and rest of arguments"""
    frame = inspect.stack()[1]
    if len (args):
        print "%s (%s)" % (frame[1], frame[2]),
        for arg in args:
            print arg,
        print
    else:
        print "%s (%s)" % (frame[1], frame[2])


def getObjectList (baseObj, base):
    """Get a list of interesting things from this object"""
    print "In", base
    match = lastClassRegex.search (base)
    lastClass = ''
    if match:
        lastClass = match.group (1)
    retval = []
    todoDict = {}
    for objName in dir (baseObj):
        #print "  ", objName
        skip = False
        for regex in memberReList:            
            if regex.search (objName):
                skip = True
                break
        if skip: continue
        # is this simple?
        obj = getattr (baseObj, objName)
        foundSimple = False
        for possible in simpleClassList:
            if isinstance (obj, possible):
                retval.append( ("%s.%s" % (base, objName), possible.__name__) )
                foundSimple = True
        if foundSimple: continue
        if isinstance (obj, ROOT.MethodProxy):
            try:
                value = obj()
            except:
                continue
            knownType = False
            for possible in simpleClassList:
                if isinstance (value, possible):
                    retval.append( ("%s.%s()" % (base, objName),
                                    possible.__name__) )
                    knownType = True
                    break
            if knownType:
                continue
            # if we're here, then we don't know what we've got
            className = value.__class__.__name__
            if objName == lastClass:
                # don't fall into a recursive trap
                continue
            skipType = False
            for regex in typeReList:
                if regex.search (className):
                    #print "skipping type '%s'" % className
                    skipType = True
                    break
            for regex in memberReList:
                if regex.search (objName):
                    skipType = True
                    break
            if skipType:
                continue
            #name = "%s.%s()" % (base, className)
            name = "%s.%s()" % (base, objName)
            todoDict[name] = value
            #retval.append( ("%s.%s()" % (base, objName), className) )
    for name, obj in sorted (todoDict.iteritems()):
        retval.extend( getObjectList (obj, name) )
    return retval


def genObjNameDef (line):
    """Returns GenObject name and ntuple definition function"""
    words = dotRE.split (line)[1:]
    func = ".".join (words)
    name =  "_".join (words)
    name = nonAlphaRE.sub ('', name)
    return name, func
    
    
def genObjectDef (mylist, tuple, alias, full):
    """ """
    # first get the name of the object
    firstName = mylist[0][0]
    match = re.match (r'(\w+)\.', firstName)
    if not match:
        raise RuntimeError, "firstName doesn't parse correctly. (%s)" \
              % firstName
    genName = match.group (1)
    genDef =  " ## GenObject %s Definition ##\n[%s]\n" % \
             (genName, genName)
    genDef += "-equiv: eta,0.1 phi,0.1\n";
    tupleDef = '[%s:%s:%s alias=%s]\n' % \
               (genName, tuple, alias, full)
    
    for variable in mylist:
        name, func = genObjNameDef (variable[0])
        if name in alreadySeenSet:
            raise RuntineError, "Duplicate '%s'" % name
        alreadySeenSet.add (name)
        typeInfo   = variable[1]
        form = defsDict[ typeInfo ]
        genDef   += form % name + '\n'
        tupleDef += "%-40s : %s\n" % (name, func)
    return genDef, tupleDef


if __name__ == "__main__":
    # Setup options parser
    parser = optparse.OptionParser \
             ("usage: %prog [options] output.txt edmFile.root goName fullBranchName alias\n" \
              "Creates control file for GenObject.")
    parser.add_option ('--tupleName', dest='tupleName', type='string',
                       default = 'reco',
                       help="Tuple name (default '%default')")
    options, args = parser.parse_args()
    if len (args) < 5:
        print "Need to provide root file and branch name. Aborting."
        sys.exit(1)
    #
    outputFile = args[0]
    rootFile   = args[1]
    goName     = args[2]
    branchName = args[3]
    alias      = args[4]
    ROOT.gROOT.SetBatch()
    # load the right libraries, etc.
    print "Loading autoloader"
    ROOT.gSystem.Load("libFWCoreFWLite.so")
    ROOT.AutoLibraryLoader.enable()
    # setup my events.
    print "getting EventTree"
    events = cmstools.EventTree(rootFile)
    print "setting alias"
    events.SetAlias (alias, branchName)

    # setup lists of things to avoid by default.  I'm avoiding 'p4$' and
    # 'vertex$' because they cause Python/Root to crash.
    print "compiling regex"
    memberSkipList = [r'^__', r'p4$', r'vertex$', r'^set', r'^begin$', r'^end$',
                      r'^clone$', r'^IsA', r'unit', r'Unit']
    memberReList = []
    for skip in memberSkipList:
        memberReList.append( re.compile (skip, re.IGNORECASE) )

    lastClassRegex = re.compile (r'([^\.]+)\(\)$')

    # list of types and member functions to avoid.
    typeSkipList   = [r'edm::Ref', r'edm::Ptr', r'vector']    
    typeReList = []
    for skip in typeSkipList:
        typeReList.append( re.compile (skip) )

    # list of 'known' classes
    simpleClassList = [int, float, str, long]

    regex1 = re.compile (r'class')
    regex2 = re.compile (r'^__', re.IGNORECASE)
    obj = ''
    for event in events:
        objects = event.getProduct (alias)
        if len (objects):
            print "object"
            obj = objects[0]
            print obj
            break
    if not obj:
        print "Can't find any '%s'.  Aborting." % alias
        sys.exit()

    print "get %s attributes" % goName
    mylist = getObjectList (obj, goName)
    targetFile = open (outputFile, 'w')
    genDef, tupleDef = genObjectDef (mylist, options.tupleName, alias,
                                     branchName)
    targetFile.write ("# -*- sh -*- For Font lock mode\n# GenObject 'event' definition\n[runevent singleton]\nrun:   type=int\nevent: type=int\n\n")
    targetFile.write (genDef)
    targetFile.write ('\n\n# %s Definition\n# Nickname and Tree\n[%s:Events]\n'\
                      % (options.tupleName, options.tupleName));
    targetFile.write ('[runevent:%s:EventAuxiliary]\nrun:   id_.run()\nevent: id_.event()\n\n' % options.tupleName)
    targetFile.write (tupleDef)

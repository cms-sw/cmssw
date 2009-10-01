#! /usr/bin/env python

import optparse
import os
import re
from pprint import pprint

epsilon = 1.e-4

def getPieceFromObject (obj, description):
    """ """
    parsed = GenObject.parseVariableTofill (description)
    return GenObject.evaluateFunction (obj, parsed)

def getDictFromObject (obj, varDict, prefix = ''):
    """Given a object and a prefix, fills an return dictionary with the
    proper values"""
    if prefix:
        obj = getPieceFromObject (obj, prefix)
    retval = {}
    for key, description in varDict.iteritems():
        retval[key] = getPieceFromObject (obj, description)
    return retval


def format (objDict, label, spacing=8):
    '''return a formatted string for given object'''
    value = objDict[label]
    diff  = objDict['delta_' + label]
    problem = False
    if isinstance (diff, float):
        formatString = '%%%d.%df' % (spacing, spacing - 3)
        retval = formatString % value
        if abs(diff) > epsilon:
            retval += ' (' + formatString % (value + diff) + ')'
        else:
            retval += ' ' * (spacing + 3)
        return retval
    else:
        formatString = '%%%ds' % spacing
        retval = formatString % value
        if diff:
            if isinstance (diff, str):
                retval += ' (' + formatSting % diff + ')'
            else:
                retval += ' (' + formatString % (value + diff) + ')'
        else:
            retval += ' ' * (spacing + 3)
        return retval
    

if __name__ == "__main__":
    parser = optparse.OptionParser ("Usage: %prog bla.root lib.so var1 [var2]")
    options, args = parser.parse_args()
    from Validation.Tools.GenObject import GenObject
    if len (args) <= 2:
        raise RuntimeError, "Must provide root file, shlib location, "\
              "and at least one variable"
    rootFilename  = args.pop(0)
    shlib     = args.pop(0)
    variables = args
    # play with shlib and cFile names
    if not re.search (r'_C.so$', shlib) and not re.search (r'_C$', shlib):
        shlib += '_C'
    cFile = re.sub (r'_C$', r'.C', re.sub(r'\.so$','', shlib))
    if not os.path.exists (cFile):
        raise RuntimeError, "Can not find accompying C file '%s'."  % cFile
    if not os.path.exists (rootFilename):
        raise RuntimeError, "Can not find root file '%s'."  % rootFilename
    # regex
    diffContRE  = re.compile (r'^class goDiffCont_(\w+)')
    # diffRE      = re.compile (r'^class goDiff_(\w+)')
    variableREDict = {}
    for var in variables:
        variableREDict[var] = ( re.compile (r'\bdelta_%s' % var),
                                re.compile (r'\bother_%s' % var) ) 
    source = open (cFile, 'r')
    stringSet    = set()
    typeFoundSet = set()
    name         = ''

    
    for line in source:
        match = diffContRE.search (line)
        if match:
            if name:
                raise RuntimeError, "Currently only supported for a single"\
                      " class at a time."
            name = match.group(1)
            continue
        for key, regexTuple in variableREDict.iteritems():
            if regexTuple[0].search(line):
                typeFoundSet.add( key )
                continue
            if regexTuple[1].search(line):
                typeFoundSet.add( key )
                stringSet.add   ( key )
    if not name:
        raise RuntimeError, "Didn't find any Diff Container"
    for var in variables:
        if var not in typeFoundSet:
            raise RuntimeError, "Variable '%s' not found." % var
    import ROOT
    if ROOT.gSystem.Load (shlib):
        raise RuntimeError, "Can not load shilb '%s'." % shlib
    rootfile = ROOT.TFile.Open (rootFilename)
    if not rootfile:
        raise RuntimeError, "Failed to open root file '%s'" % rootFilename
    tree = rootfile.Get ('diffTree')
    if not tree:
        raise RuntimeError, "Failed to get 'diffTree'"
    size = tree.GetEntries()
    runeventDict = {'run':'run', 'event':'event'}
    indexDict = {'index':'index', 'delta_index':'delta_index'}
    infoDict = {}
    for var in variables:
        infoDict[var] = var;
        if var in stringSet:
            infoDict['delta_' + var] = 'other_' + var
        else:
            infoDict['delta_' + var] = 'delta_' + var
    for index in range (size):
        tree.GetEntry (index)
        runevent = getDictFromObject (tree, runeventDict, 'runevent')
        pprint (runevent)
        diffColl = getPieceFromObject (tree, name+'.diff')
        size = diffColl.size()
        if not size:
            continue
        print "index",
        for var in variables:
            print "%19s" % (var + '      '),
        print
        print '-' * (5 + 19 * size)
        for index in range (size):
            diff = diffColl[index]
            index = getDictFromObject (diff, indexDict)
            print '  ', format (index, 'index', 3),
            info = getDictFromObject (diff, infoDict)
            for var in variables:
                print '  ', format (info, var),
            print
        print

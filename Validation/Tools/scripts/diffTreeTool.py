#! /usr/bin/env python3

from __future__ import print_function
from builtins import range
import optparse
import os
import re
from pprint import pprint
import six

epsilon = 1.e-4

def getPieceFromObject (obj, description):
    """Returns piece from object """
    parsed = GenObject.parseVariableTofill (description)
    return GenObject.evaluateFunction (obj, parsed)

def getDictFromObject (obj, varDict, prefix = ''):
    """Given a object and a prefix, fills an return dictionary with the
    proper values"""
    if prefix:
        obj = getPieceFromObject (obj, prefix)
    retval = {}
    for key, description in six.iteritems(varDict):
        retval[key] = getPieceFromObject (obj, description)
    return retval


def format (objDict, label, spacing=9, firstOnly = False):
    '''return a formatted string for given object'''
    value = objDict[label]
    if firstOnly:
        diff = 0.
    else:
        diff  = objDict['delta_' + label]
        
    problem = False
    if isinstance (value, float):
        formatString = '%%%d.%df' % (spacing, spacing - 5)
        retval = formatString % value
        if abs(diff) > epsilon:
            if options.delta:
                retval += ' [' + formatString % (diff) + ']'
            else:
                retval += ' (' + formatString % (value + diff) + ')'
        elif not firstOnly:
            retval += ' ' * (spacing + 3)
        return retval
    else:
        formatString = '%%%ds' % spacing
        retval = formatString % value
        if diff:
            if isinstance (value, str):
                retval += ' (' + formatString % diff + ')'
            elif options.delta:
                retval += ' [' + formatString % diff + ']'
            else:
                retval += ' (' + formatString % (value + diff) + ')'
        elif not firstOnly:
            retval += ' ' * (spacing + 3)
        return retval
    

if __name__ == "__main__":
    parser = optparse.OptionParser ("Usage: %prog bla.root lib.so var1 [var2]")
    parser.add_option ("--delta", dest="delta",
                       action="store_true", default=False,
                       help="Show deltas when difference is large enough.")
    parser.add_option ("--skipUndefined", dest="skipUndefined",
                       action="store_true", default=False,
                       help="Skip undefined variables without warning.")
    options, args = parser.parse_args()
    from Validation.Tools.GenObject import GenObject
    if len (args) <= 2:
        raise RuntimeError("Must provide root file, shlib location, "\
              "and at least one variable")
    rootFilename  = args.pop(0)
    shlib     = args.pop(0)
    variables = args
    # play with shlib and cFile names
    if not re.search (r'_C.so$', shlib) and not re.search (r'_C$', shlib):
        shlib += '_C'
    cFile = re.sub (r'_C$', r'.C', re.sub(r'\.so$','', shlib))
    if not os.path.exists (cFile):
        raise RuntimeError("Can not find accompying C file '%s'."  % cFile)
    if not os.path.exists (rootFilename):
        raise RuntimeError("Can not find root file '%s'."  % rootFilename)
    # regex
    diffContRE  = re.compile (r'^class goDiffCont_(\w+)')
    # diffRE      = re.compile (r'^class goDiff_(\w+)')
    variableREDict = {}
    for var in variables:
        variableREDict[var] = ( re.compile (r'\bdelta_%s\b' % var),
                                re.compile (r'\bother_%s\b' % var) ) 
    source = open (cFile, 'r')
    stringSet    = set()
    typeFoundSet = set()
    name         = ''

    
    for line in source:
        match = diffContRE.search (line)
        if match:
            if name:
                raise RuntimeError("Currently only supported for a single"\
                      " class at a time.")
            name = match.group(1)
            continue
        for key, regexTuple in six.iteritems(variableREDict):
            if regexTuple[0].search(line):
                typeFoundSet.add( key )
                continue
            if regexTuple[1].search(line):
                typeFoundSet.add( key )
                stringSet.add   ( key )
    if not name:
        raise RuntimeError("Didn't find any Diff Container")
    working = []
    for var in variables:
        if var not in typeFoundSet:
            if not options.skipUndefined:
                raise RuntimeError("Variable '%s' not found." % var)
        else:
            working.append (var)
    variables = working
    import ROOT
    if ROOT.gSystem.Load (shlib):
        raise RuntimeError("Can not load shilb '%s'." % shlib)
    rootfile = ROOT.TFile.Open (rootFilename)
    if not rootfile:
        raise RuntimeError("Failed to open root file '%s'" % rootFilename)
    tree = rootfile.Get ('diffTree')
    if not tree:
        raise RuntimeError("Failed to get 'diffTree'")
    size = tree.GetEntries()
    runeventDict = {'Run':'run', 'Event':'event'}
    indexSingleDict = {'index':'index'}
    indexDoubleDict = {'index':'index', 'delta_index':'delta_index'}
    infoSingleDict = {}
    infoDoubleDict = {}
    for var in variables:
        infoSingleDict[var] = infoDoubleDict[var] = var;        
        if var in stringSet:
            infoDoubleDict['delta_' + var] = 'other_' + var
        else:
            infoDoubleDict['delta_' + var] = 'delta_' + var
    for index in range (size):
        tree.GetEntry (index)
        runevent = getDictFromObject (tree, runeventDict, 'runevent')
        pprint (runevent)
        # first only
        firstOnlyColl  = getPieceFromObject (tree, name + '.firstOnly')
        size = firstOnlyColl.size()
        if size:            
            print("First Only:\n   index    ", end=' ')
            for var in variables:
                print("%-12s" % (' ' + var), end=' ')
            print()
            print('-' * (12 + 11 * len(variables)))
        for index in range (size):
            firstOnly = firstOnlyColl[index]
            index = getDictFromObject (firstOnly, indexSingleDict)
            print('  ', format (index, 'index', 3, firstOnly = True), end=' ')
            info = getDictFromObject (firstOnly, infoSingleDict)
            for var in variables:
                print('  ', format (info, var, firstOnly = True), end=' ')
            print()
        print()
        # second only
        secondOnlyColl = getPieceFromObject (tree, name + '.secondOnly')
        size = secondOnlyColl.size()
        if size:            
            print("Second Only:\n   index    ", end=' ')
            for var in variables:
                print("%-12s" % (' ' + var), end=' ')
            print()
            print('-' * (12 + 11 * len(variables)))
        for index in range (size):
            secondOnly = secondOnlyColl[index]
            index = getDictFromObject (secondOnly, indexSingleDict)
            print('  ', format (index, 'index', 3, firstOnly = True), end=' ')
            info = getDictFromObject (secondOnly, infoSingleDict)
            for var in variables:
                print('  ', format (info, var, firstOnly = True), end=' ')
            print()
        print()
        # both
        diffColl = getPieceFromObject (tree, name+'.diff')
        size = diffColl.size()
        if size:            
            print("Both:\n   index", end=' ')
            for var in variables:
                print("%-24s" % ('          ' + var), end=' ')
            print()
            print('-' * (16 + 23 * len(variables)))
        for index in range (size):
            diff = diffColl[index]
            index = getDictFromObject (diff, indexDoubleDict)
            print('  ', format (index, 'index', 3), end=' ')
            info = getDictFromObject (diff, infoDoubleDict)
            for var in variables:
                print('  ', format (info, var), end=' ')
            print()
        print()
 

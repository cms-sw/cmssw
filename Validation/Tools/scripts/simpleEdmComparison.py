#! /usr/bin/env python

import inspect
import itertools
import logging
import optparse
import pprint
import random
import sys

import ROOT
from DataFormats.FWLite import Events, Handle

typeMap = { 'double' : ['double', 'vector<double>'],
            'int'    : ['int',    'vector<int>'],}

class ProductNotFoundError(RuntimeError):
    """
    Special exception for a product not in file
    """
    pass

def compareEvents(event1, event2, handleName, label, options):
    """
    Compare two events
    """

    # Is it a vector of objects or object (funky ROOT buffer for single value)
    isSimpleObject = (handleName.find('vector') == -1)

    # Compare run, lumi, event
    aux1 = event1.eventAuxiliary()
    aux2 = event2.eventAuxiliary()

    rle1 = (aux1.run(), aux1.luminosityBlock(), aux1.event())
    rle2 = (aux2.run(), aux2.luminosityBlock(), aux2.event())

    logging.debug("Comparing RLE #'s %s and %s" % (rle1, rle2))

    if rle1 != rle2:
        raise RuntimeError("Run/Lumi/Events don't match: %s vs %s" % (rle1, rle2))
    handle1 = Handle(handleName)
    handle2 = Handle(handleName)

    if event1.getByLabel(label, handle1) and event2.getByLabel(label, handle2):
        objects1 = handle1.product()
        objects2 = handle1.product()
    else:
        raise ProductNotFoundError("Product %s %s not found." % (handleName, label))

    if isSimpleObject:
        val1 = objects1[0]
        val2 = objects2[0]
        if options.blurRate and options.blur and random.random() < options.blurRate:
            # This is different than Charles's method, which makes no sense to me
            val1 += (random.random()-0.5) * options.blur
        if val1 != val2:
            logging.error("Mismatch %s and %s in %s" % (val1, val2, aux2.event()))
            return (1, 1)
        else:
            logging.debug("Match of %s in %s" % (objects1[0], aux2.event()))
            return (1, 0)
    else:
        count    = 0
        mismatch = 0
        for val1, val2 in itertools.izip_longest(objects1, objects2):
            count += 1
            if options.blurRate and options.blur and random.random() < options.blurRate:
                # This is different than Charles's method, which makes no sense to me
                val1 += (random.random()-0.5) * options.blur * val1
            if val1 != val2:
                mismatch += 1
                logging.error("Comparison problem %s != %s" % (val1, val2))
	logging.debug("Compared %s elements" % count)
        return (count, mismatch)

if __name__ == "__main__":

    ###################
    ## Setup Options ##
    ###################

    random.seed()
    logging.basicConfig(level=logging.INFO)

    parser = optparse.OptionParser("usage: %prog [options] config.txt file1.root file2.root\nVisit https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsToolsEdmOneToOneComparison\nfor full documentation.")
    modeGroup    = optparse.OptionGroup (parser, "Mode Conrols")
    tupleGroup   = optparse.OptionGroup (parser, "Tuple Controls")
    optionsGroup = optparse.OptionGroup (parser, "Options")

    modeGroup.add_option ('--compare', dest='compare', action='store_true',
                          help='Compare tuple1 to tuple2')

    tupleGroup.add_option ('--numEvents', dest='numEvents', type='int',
                           default=1e9,
                           help="number of events for first and second file")

    tupleGroup.add_option ('--label', dest='label', type='string',
                           action='append',
                           help="Change label ('tuple^object^label')")

    optionsGroup.add_option ('--blur1', dest='blur', type='float',
                             default=0.05,
                             help="Randomly changes values by 'BLUR'  " +\
                             "from tuple1.  For debugging only.")
    optionsGroup.add_option ('--blurRate', dest='blurRate', type='float',
                             default=0.00,
                             help="Rate at which objects will be changed. " + \
                             "(%default default)")

    parser.add_option_group (modeGroup)
    parser.add_option_group (tupleGroup)
    parser.add_option_group (optionsGroup)
    (options, args) = parser.parse_args()

    if len(args) != 3:
        parser.error("Too many or too few arguments")
    options.config = args[0]
    options.file1  = args[1]
    options.file2  = args[2]

    # Parse object name and label out of Charles format
    tName, objName, lName = options.label[0].split('^')
    label = lName.split(',')

    ROOT.gROOT.SetBatch()

    ROOT.gSystem.Load("libFWCoreFWLite.so")
    ROOT.gSystem.Load("libDataFormatsFWLite.so")
    ROOT.FWLiteEnabler::enable()

    chain1 = Events ([options.file1], forceEvent=True)
    chain2 = Events ([options.file2], forceEvent=True)

    if chain1.size() != chain1.size():
        raise RuntimeError("Files have different #'s of events")
    numEvents = min(options.numEvents, chain1.size())

    # Parameters to this script are the same regardless if the
    # product is double or vector<double> so have to try both
    productsCompared = 0
    totalCount = 0
    mismatches = 0
    for handleName in typeMap[objName]:
        try:
            chain1.toBegin()
            chain2.toBegin()
            logging.info("Testing identity for handle=%s, label=%s" % (handleName, label))
            # Use itertools to iterate over lists in ||
            for ev1, ev2, count in itertools.izip(chain1, chain2, xrange(numEvents)):
                evCount, evMismatch = compareEvents(event1=ev1, event2=ev2, handleName=handleName, label=label, options=options)
                totalCount += evCount
                mismatches += evMismatch
            logging.info("Compared %s events" % (count+1))
            productsCompared += 1
            # Try to reproduce the output that Charles's summary script is expecting
            plagerDict = {'eventsCompared' : count+1}
            plagerDict.update({'count_%s' % objName : totalCount})
            if mismatches:
                plagerDict.update({objName: {'_var' : {handleName:mismatches}}})
            print "Summary"
            pprint.pprint(plagerDict)
        except ProductNotFoundError:
            logging.info("No product found for handle=%s, label=%s" % (handleName, label))

    logging.info("Total products compared: %s, %s/%s" % (productsCompared, mismatches, totalCount))

    if not productsCompared:
        print "Plager compatible message: not able to get any products"
        sys.exit()

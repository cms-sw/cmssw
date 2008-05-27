#! /usr/bin/env python

import os, sys
import Configuration

from optparse import OptionParser

# Parsing command line option
parser = OptionParser()
parser.add_option("-r", "--release", dest="release", default=None, help="release that EDM files were created in")
parser.add_option("-d", "--datasets", dest="datasets", default=None,  help="Comma seperated list of datasets")
parser.add_option("-g", "--generator", dest="generator", default="Vista",  help="Generator to test files, if Vista use default")
(options, args) = parser.parse_args()

if options.release != None and options.datasets != None:
    sets = options.datasets.split(',')
    for dataset in sets:
        file = open(Configuration.variables['HomeDirectory']+'data/'+dataset+'-Vista__'+options.generator+'__'+options.release+'__DBS.cfi', 'w')
        file.write("Dummy cfi file for Vista subscription")
        file.close()


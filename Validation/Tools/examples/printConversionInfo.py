#! /usr/bin/env python3

from __future__ import print_function
import ROOT
import sys
from DataFormats.FWLite import Events, Handle

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('python')
options.parseArguments()

print("maxEvents", options.maxEvents)

events = Events (options)
conversionHandle  = Handle ('vector<reco::Conversion>')
conversionLabel   = ("conversions")

for event in events:
    aux = event.object().eventAuxiliary()
    print("run %6d event %d" % (aux.run(), aux.event()))
    event.getByLabel (conversionLabel, conversionHandle)
    conversionVector = conversionHandle.product()
    for index, conversion in enumerate (conversionVector):
        print("  %2d  %8.4f" % (index, conversion.EoverP()))

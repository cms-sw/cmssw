#!/usr/bin/env python
import sys
from ROOT import *
import ROOT

from optparse import OptionParser
parser = OptionParser(usage="usage: %prog [options] file directory")
parser.add_option("-r", "--reference-file", dest="ref", type="string", 
                  metavar="FILE", help="take reference plots from FILE")

(options, args) = parser.parse_args()

## === GLOBAL VARIABLES ===
fileIn  = ROOT.TFile(args[0])

fileOut = ROOT.TFile("skimSummaryFraction.root","RECREATE")

hFilterSelected = fileIn.Get("SkimSummary/filterSelected")
hTotalEvents = fileIn.Get("SkimSummary/totalEvents")

hFractionOfPD = hFilterSelected.Clone("filterSelectedFraction")
hFractionOfPD.Scale(1./hTotalEvents.GetBinContent(1))

fileOut.Write()
fileOut.Close()

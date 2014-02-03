import FWCore.ParameterSet.Config as cms

# File: HTMET.cfi
# Author: B. Scurlock
# Date: 03.04.2008
#
# Fill validation histograms for MET. Assumes htMetSC5, 
# htMetSC7, htMetIC5, htMetKT4, and htMetKT6 are in the event.
from Validation.RecoMET.HTMET_cfi import *
analyzeHTMET = cms.Sequence(htMetSC5Analyzer*htMetSC7Analyzer*htMetIC5Analyzer*htMetKT4Analyzer*htMetKT6Analyzer)


import FWCore.ParameterSet.Config as cms

# File: PFMET.cff
# Author: R. Remington
# Date: 11.14.2008
#
# Fill validation histograms for MET. Assumes tcMet is in the event.
from Validation.RecoMET.PFMET_cfi import *
analyzePFMET = cms.Sequence(pfMetAnalyzer)


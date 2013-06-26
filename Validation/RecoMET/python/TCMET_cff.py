import FWCore.ParameterSet.Config as cms

# File: TCMET.cff
# Author: R. Remington
# Date: 11.14.2008
#
# Fill validation histograms for MET. Assumes tcMet is in the event.
from Validation.RecoMET.TCMET_cfi import *
analyzeTCMET = cms.Sequence(tcMetAnalyzer)


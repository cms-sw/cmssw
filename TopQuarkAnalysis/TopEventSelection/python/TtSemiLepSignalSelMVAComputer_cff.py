import FWCore.ParameterSet.Config as cms

#
# produce mvaComputer with all necessary ingredients
#
from TopQuarkAnalysis.TopEventSelection.TtSemiLepSignalSelMVAComputer_cfi import *

## path for mva input file
TtSemiLepSignalSelMVAFileSource = cms.ESSource("TtSemiLepSignalSelMVAFileSource",
    ttSemiLepSignalSelMVA = cms.FileInPath('TopQuarkAnalysis/TopEventSelection/data/TtSemiLepSignalSel.mva')
)

import FWCore.ParameterSet.Config as cms

#
# produce mvaComputer with all necessary ingredients
#
from TopQuarkAnalysis.TopEventSelection.TtFullHadSignalSelMVAComputer_cfi import *

## path for mva input file
TtFullHadSignalSelMVAFileSource = cms.ESSource("TtFullHadSignalSelMVAFileSource",
    ttFullHadSignalSelMVA = cms.FileInPath('TopQuarkAnalysis/TopEventSelection/data/TtFullHadSignalSel.mva')
)

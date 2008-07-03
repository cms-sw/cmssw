import FWCore.ParameterSet.Config as cms

#
# produce mvaComputer with all necessary ingredients
#

## import MVA trainer cfi
from TopQuarkAnalysis.TopJetCombination.TtSemiJetCombMVATrainer_Muons_cfi import *

## define path for mva save file
mvaTtSemiJetCombSaveFile = cms.EDAnalyzer("TtSemiJetCombMVASaveFile",
    ttSemiJetCombMVA = cms.string('TopQuarkAnalysis/TopJetCombination/data/TtSemiJetComb_Muons.mva')
)

## ------------------------------------------------------------------------------------------
## configuration of event looper for mva taining; take care to make the 
## looper known to the process. The way to do that is to add the following
## lines to your cfg.py
##
## from TopQuarkAnalysis.TopJetCombination.TtSemiJetCombMVATrainer_Muons_cff import looper
## process.looper = looper
## ------------------------------------------------------------------------------------------ 
looper = cms.Looper("TtSemiJetCombMVATrainerLooper",
    trainers = cms.VPSet(cms.PSet(
        monitoring = cms.untracked.bool(True),
        loadState  = cms.untracked.bool(False),
        saveState  = cms.untracked.bool(True),
        calibrationRecord = cms.string('ttSemiJetCombMVA'),
        trainDescription = cms.untracked.string('TopQuarkAnalysis/TopJetCombination/data/TtSemiJetCombMVATrainer_Muons.xml')
    ))
)

makeMVATraining = cms.Sequence(trainTtSemiJetCombMVA*mvaTtSemiJetCombSaveFile)


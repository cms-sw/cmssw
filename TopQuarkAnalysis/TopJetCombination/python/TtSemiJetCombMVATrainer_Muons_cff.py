import FWCore.ParameterSet.Config as cms

#
# produce mvaComputer with all necessary ingredients
#

## include MVA trainer cfi
from TopQuarkAnalysis.TopJetCombination.TtSemiJetCombMVATrainer_Muons_cfi import *

## path for mva save file
mvaTtSemiJetCombSaveFile = cms.EDAnalyzer("TtSemiJetCombMVASaveFile",
    ttSemiJetCombMVA = cms.string('TtSemiJetComb_Muons.mva')
)

## configuration of event looper for mva taining
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


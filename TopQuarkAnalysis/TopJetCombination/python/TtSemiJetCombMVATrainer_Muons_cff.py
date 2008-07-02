# The following comments couldn't be translated into the new config version:

# add looper for different training processes

import FWCore.ParameterSet.Config as cms

# include MVA trainer cfi
from TopQuarkAnalysis.TopJetCombination.TtSemiJetCombMVATrainer_Muons_cfi import *
# add module for mva save file
mvaTtSemiJetCombSaveFile = cms.EDFilter("TtSemiJetCombMVASaveFile",
    ttSemiJetCombMVA = cms.string('TtSemiJetComb_Muons.mva')
)

looper = cms.Looper("TtSemiJetCombMVATrainerLooper",
    trainers = cms.VPSet(cms.PSet(
        monitoring = cms.untracked.bool(True),
        calibrationRecord = cms.string('ttSemiJetCombMVA'),
        saveState = cms.untracked.bool(True),
        trainDescription = cms.untracked.string('TopQuarkAnalysis/TopJetCombination/data/TtSemiJetCombMVATrainer_Muons.xml'),
        loadState = cms.untracked.bool(False)
    ))
)

makeMVATraining = cms.Sequence(trainTtSemiJetCombMVA*mvaTtSemiJetCombSaveFile)


import FWCore.ParameterSet.Config as cms

#
# produce mvaComputer with all necessary ingredients
#

## import MVA trainer cfi
from TopQuarkAnalysis.TopJetCombination.TtSemiLepJetCombMVATrainer_Muons_cfi import *

## ------------------------------------------------------------------------------------------
## configuration of event looper for mva taining; take care to make the 
## looper known to the process. The way to do that is to add the following
## lines to your cfg.py
##
## from TopQuarkAnalysis.TopJetCombination.TtSemiLepJetCombMVATrainer_Muons_cff import looper
## process.looper = looper
## ------------------------------------------------------------------------------------------ 
looper = cms.Looper("TtSemiLepJetCombMVATrainerLooper",
    trainers = cms.VPSet(cms.PSet(
        monitoring = cms.untracked.bool(False),
        loadState  = cms.untracked.bool(False),
        saveState  = cms.untracked.bool(True),
        calibrationRecord = cms.string('ttSemiLepJetCombMVA'),
        trainDescription = cms.untracked.string(
            'TopQuarkAnalysis/TopJetCombination/data/TtSemiLepJetCombMVATrainTreeSaver_Muons.xml')
    ))
)

## provide a sequence to save a tree for the training
saveTtSemiLepJetCombMVATrainTree = cms.Sequence(trainTtSemiLepJetCombMVA)


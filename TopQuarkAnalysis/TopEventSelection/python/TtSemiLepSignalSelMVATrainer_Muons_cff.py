import FWCore.ParameterSet.Config as cms

#
# produce mvaComputer with all necessary ingredients
#

## import MVA trainer cfi
from TopQuarkAnalysis.TopEventSelection.TtSemiLepSignalSelMVATrainer_Muons_cfi import *

## define path for mva save file
mvaTtSemiLepSignalSelSaveFile = cms.EDAnalyzer("TtSemiLepSignalSelMVASaveFile",
    ttSemiLepSignalSelMVA = cms.string('TopQuarkAnalysis/TopEventSelection/data/TtSemiLepSignalSel_Muons.mva')
)

## ------------------------------------------------------------------------------------------
## configuration of event looper for mva taining; take care to make the 
## looper known to the process. The way to do that is to add the following
## lines to your cfg.py
##
## from TopQuarkAnalysis.TopEventSelection.TtSemiLepSignalSelMVATrainer_Muons_cff import looper
## process.looper = looper
## ------------------------------------------------------------------------------------------ 
looper = cms.Looper("TtSemiLepSignalSelMVATrainerLooper",
    trainers = cms.VPSet(cms.PSet(
        monitoring = cms.untracked.bool(True),
        loadState  = cms.untracked.bool(False),
        saveState  = cms.untracked.bool(True),
        calibrationRecord = cms.string('ttSemiLepSignalSelMVA'),
        trainDescription = cms.untracked.string('TopQuarkAnalysis/TopEventSelection/data/TtSemiLepSignalSelMVATrainer_Muons.xml')
    ))
)

## provide a sequence for the training
## remark: do not use this sequence if you want to call your trainer after an event filter
##         since the SaveFile module should be called in an unfiltered path!
makeMVATraining = cms.Sequence(trainTtSemiLepSignalSelMVA*mvaTtSemiLepSignalSelSaveFile)

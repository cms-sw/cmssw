import FWCore.ParameterSet.Config as cms

#
# produce mvaComputer with all necessary ingredients
#

## import MVA trainer cfi
from TopQuarkAnalysis.TopEventSelection.TtSemiLepSignalSelMVATrainer_cfi import *

## ------------------------------------------------------------------------------------------
## configuration of event looper for mva taining; take care to make the 
## looper known to the process. The way to do that is to add the following
## lines to your cfg.py
##
## from TopQuarkAnalysis.TopEventSelection.TraintreeSaver_cff import looper
## process.looper = looper
## ------------------------------------------------------------------------------------------ 
looper = cms.Looper("TtSemiLepSignalSelMVATrainerLooper",
    trainers = cms.VPSet(cms.PSet(
        monitoring = cms.untracked.bool(False),
        loadState  = cms.untracked.bool(False),
        saveState  = cms.untracked.bool(True),
        calibrationRecord = cms.string('traintreeSaver'),
        trainDescription = cms.untracked.FileInPath(
            'TopQuarkAnalysis/TopEventSelection/data/TtSemiLepSignalSelMVATrainTreeSaver.xml')
    ))
)

## provide a sequence for the training
## remark: do not use this sequence if you want to call your trainer after an event filter
##         since the SaveFile module should be called in an unfiltered path!
saveTrainTree = cms.Sequence(buildTraintree)

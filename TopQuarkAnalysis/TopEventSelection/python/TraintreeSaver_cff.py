import FWCore.ParameterSet.Config as cms

#
# produce mvaComputer with all necessary ingredients
#

## import MVA trainer cfi
from TopQuarkAnalysis.TopEventSelection.TraintreeSaver_cfi import *

## define path for mva save file
traintreeSaverSaveFile = cms.EDAnalyzer("TtSemiLepSignalSelMVASaveFile",
    traintreeSaver = cms.string('TopQuarkAnalysis/TopEventSelection/data/TraintreeSaver.mva')
)

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
        monitoring = cms.untracked.bool(True),
        loadState  = cms.untracked.bool(False),
        saveState  = cms.untracked.bool(True),
        calibrationRecord = cms.string('traintreeSaver'),
        trainDescription = cms.untracked.string('TopQuarkAnalysis/TopEventSelection/data/TraintreeSaver.xml')
    ))
)

## provide a sequence for the training
## remark: do not use this sequence if you want to call your trainer after an event filter
##         since the SaveFile module should be called in an unfiltered path!
saveTrainTree = cms.Sequence(buildTraintree*traintreeSaverSaveFile)

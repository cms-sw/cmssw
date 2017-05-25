import FWCore.ParameterSet.Config as cms

hcalsimhitsClient = cms.EDProducer("HcalSimHitsClient", 
     DQMDirName = cms.string("/"), # root directory
     Verbosity  = cms.untracked.bool(False),
)

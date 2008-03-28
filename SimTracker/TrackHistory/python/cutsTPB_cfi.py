import FWCore.ParameterSet.Config as cms

cutsTPB = cms.EDFilter("BTrackSelection",
    src = cms.InputTag("trackingtruthprod")
)



import FWCore.ParameterSet.Config as cms

simHGCEEdigis = cms.EDProducer("HGCDigiProducer",
                               hitCollection     = cms.untracked.string("HGCHitsEE"),
                               maxSimHitsAccTime = cms.untracked.uint32(100),
                               doTrivialDigis    = cms.untracked.bool(True)
                               )

simHGCHEfrontDigis = simHGCEEdigis.clone( hitCollection = cms.untracked.string("HGCHitsHEfront") )

simHGCHEbackDigis = simHGCEEdigis.clone( hitCollection = cms.untracked.string("HGCHitsHEback") )

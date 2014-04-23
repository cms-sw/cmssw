import FWCore.ParameterSet.Config as cms

hgceeDigiBlock = cms.PSet( hitCollection     = cms.untracked.string("HGCHitsEE"),
                           maxSimHitsAccTime = cms.untracked.uint32(100),
                           doTrivialDigis    = cms.untracked.bool(True)
                           )

hgchefrontDigiBlock = hgceeDigiBlock.clone( hitCollection = cms.untracked.string("HGCHitsHEfront") )

hgchebackDigiBlock  = hgceeDigiBlock.clone( hitCollection = cms.untracked.string("HGCHitsHEback") )

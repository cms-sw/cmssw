import FWCore.ParameterSet.Config as cms

hgceeDigitizer = cms.PSet( accumulatorType  = cms.string("HGCDigiProducer"),
                           hitCollection     = cms.untracked.string("HGCHitsEE"),
                           maxSimHitsAccTime = cms.untracked.uint32(100),
                           doTrivialDigis    = cms.untracked.bool(True),
                           makeDigiSimLinks = cms.untracked.bool(False)
                           )

hgchefrontDigitizer = hgceeDigitizer.clone( hitCollection = cms.untracked.string("HGCHitsHEfront") )

hgchebackDigitizer = hgceeDigitizer.clone( hitCollection = cms.untracked.string("HGCHitsHEback") )



                           



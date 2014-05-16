import FWCore.ParameterSet.Config as cms

hgceeDigitizer = cms.PSet( accumulatorType   = cms.string("HGCDigiProducer"),
                           hitCollection     = cms.untracked.string("HGCHitsEE"),
                           digiCollection    = cms.untracked.string("HGCDigisEE"),
                           maxSimHitsAccTime = cms.untracked.uint32(100),
                           bxTime            = cms.untracked.int32(25),
                           doTrivialDigis    = cms.untracked.bool(True),
                           makeDigiSimLinks  = cms.untracked.bool(False),
                           digiCfg = cms.untracked.PSet( lsbInMeV   = cms.untracked.double(12.0),
                                                         noiseInMeV = cms.untracked.double(0)
                                                         )
                           )

hgchefrontDigitizer = hgceeDigitizer.clone()
hgchefrontDigitizer.hitCollection  = cms.untracked.string("HGCHitsHEfront")
hgchefrontDigitizer.digiCollection = cms.untracked.string("HGCDigisHEfront")
hgchefrontDigitizer.digiCfg.lsbInMev = cms.untracked.double(17.6)

hgchebackDigitizer = hgceeDigitizer.clone()
hgchebackDigitizer.hitCollection = cms.untracked.string("HGCHitsHEback")
hgchebackDigitizer.digiCollection = cms.untracked.string("HGCDigisHEback")
hgchebackDigitizer.digiCfg.lsbInMev = cms.untracked.double(310.8)



                           



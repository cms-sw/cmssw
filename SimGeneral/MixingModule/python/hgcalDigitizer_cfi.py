import FWCore.ParameterSet.Config as cms

hgceeDigitizer = cms.PSet( accumulatorType   = cms.string("HGCDigiProducer"),
                           hitCollection     = cms.untracked.string("HGCHitsEE"),
                           digiCollection    = cms.untracked.string("HGCDigisEE"),
                           maxSimHitsAccTime = cms.untracked.uint32(100),
                           bxTime            = cms.untracked.int32(25),
                           doTrivialDigis    = cms.untracked.bool(True),
                           makeDigiSimLinks  = cms.untracked.bool(False),
                           digiCfg = cms.untracked.PSet( mipInKeV      = cms.untracked.double(55.1),
                                                         lsbInMIP      = cms.untracked.double(0.25),
                                                         mip2noise     = cms.untracked.double(7.0),
                                                         adcThreshold  = cms.untracked.uint32(2),
                                                         doTimeSamples = cms.untracked.bool(True),
                                                         )
                           )

hgchefrontDigitizer = hgceeDigitizer.clone()
hgchefrontDigitizer.hitCollection  = cms.untracked.string("HGCHitsHEfront")
hgchefrontDigitizer.digiCollection = cms.untracked.string("HGCDigisHEfront")
hgchefrontDigitizer.digiCfg.mipInKeV = cms.untracked.double(85.0)
hgchefrontDigitizer.digiCfg.lsbInMIP = cms.untracked.double(0.25)
hgchefrontDigitizer.digiCfg.mip2noise = cms.untracked.double(7.0)


hgchebackDigitizer = hgceeDigitizer.clone()
hgchebackDigitizer.hitCollection = cms.untracked.string("HGCHitsHEback")
hgchebackDigitizer.digiCollection = cms.untracked.string("HGCDigisHEback")
hgchebackDigitizer.digiCfg.mipInKeV = cms.untracked.double(1498.4)
hgchebackDigitizer.digiCfg.lsbInMIP = cms.untracked.double(0.25)
hgchebackDigitizer.digiCfg.mip2noise = cms.untracked.double(5.0)




                           



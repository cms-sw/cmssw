import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *


seedingLayersEDProducer = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring(),
    BPix = cms.PSet(),
    FPix = cms.PSet(),
    TIB = cms.PSet(),
    TID = cms.PSet(),
    TOB = cms.PSet(),
    TEC = cms.PSet(),
    MTIB = cms.PSet(),
    MTID = cms.PSet(),
    MTOB = cms.PSet(),
    MTEC = cms.PSet(),
)

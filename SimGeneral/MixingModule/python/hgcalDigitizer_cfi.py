import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HGCSimProducers.hgcDigiProducers_cff import *

hgceeDigitizer = cms.PSet( digiBlock        = hgceeDigiBlock,
                           accumulatorType  = cms.string("HGCDigiProducer"),
                           makeDigiSimLinks = cms.untracked.bool(False)
                           )

hgchefrontDigitizer = cms.PSet( digiBlock        = hgchefrontDigiBlock,
                                accumulatorType  = cms.string("HGCDigiProducer"),
                                makeDigiSimLinks = cms.untracked.bool(False)
                                )

hgchebackDigitizer = cms.PSet( digiBlock        = hgchebackDigiBlock,
                               accumulatorType  = cms.string("HGCDigiProducer"),
                               makeDigiSimLinks = cms.untracked.bool(False)                               
                               )



                           



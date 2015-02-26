import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.SiPixelSimParameters_cfi import SiPixelSimBlock

pixelDigitizer = cms.PSet(
    SiPixelSimBlock,
    accumulatorType = cms.string("SiPixelDigitizer"),
    hitsProducer = cms.string('g4SimHits'),
    makeDigiSimLinks = cms.untracked.bool(True)
)

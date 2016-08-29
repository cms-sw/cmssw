import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.SiPixelSimParameters_cfi import SiPixelSimBlock
from SimTracker.SiPhase2Digitizer.phase2TrackerDigitizer_cfi import *

pixelDigitizer = cms.PSet(
    SiPixelSimBlock,
    accumulatorType = cms.string("SiPixelDigitizer"),
    hitsProducer = cms.string('g4SimHits'),
    makeDigiSimLinks = cms.untracked.bool(True)
)

from Configuration.StandardSequences.Eras import eras
eras.phase2_tracker.toModify( pixelDigitizer, 
                              pixel = phase2TrackerDigitizer)


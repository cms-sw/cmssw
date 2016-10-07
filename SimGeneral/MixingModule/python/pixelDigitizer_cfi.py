import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.SiPixelSimParameters_cfi import SiPixelSimBlock

pixelDigitizer = cms.PSet(
    SiPixelSimBlock,
    accumulatorType = cms.string("SiPixelDigitizer"),
    hitsProducer = cms.string('g4SimHits'),
    makeDigiSimLinks = cms.untracked.bool(True)
)

from Configuration.StandardSequences.Eras import eras
from SimTracker.SiPhase2Digitizer.phase2TrackerDigitizer_cfi import phase2TrackerDigitizer as _phase2TrackerDigitizer
eras.phase2_tracker.toReplaceWith(pixelDigitizer, _phase2TrackerDigitizer)


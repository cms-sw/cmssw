import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.SiStripGainSimESProducer_cfi import *
from SimGeneral.MixingModule.SiStripSimParameters_cfi import SiStripSimBlock

stripDigitizer = cms.PSet(
    SiStripSimBlock,
    accumulatorType = cms.string("SiStripDigitizer"),
    hitsProducer = cms.string('g4SimHits'),
    makeDigiSimLinks = cms.untracked.bool(True)
    )

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify( stripDigitizer, ROUList = ["g4SimHitsTrackerHitsPixelBarrelLowTof",
                                                         "g4SimHitsTrackerHitsPixelEndcapLowTof"]
)


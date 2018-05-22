import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.SiPixelSimParameters_cfi import SiPixelSimBlock

pixelDigitizer = cms.PSet(
    SiPixelSimBlock,
    accumulatorType = cms.string("SiPixelDigitizer"),
    hitsProducer = cms.string('g4SimHits'),
    makeDigiSimLinks = cms.untracked.bool(True)
)
from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
premix_stage1.toModify(pixelDigitizer, makeDigiSimLinks = False)

# Customize here instead of SiPixelSimBlock as the latter is imported
# also to DataMixer configuration, and the original version is needed
# there. Customize before phase2_tracker because this customization
# applies only to phase0/1 pixel, and at the moment it is unclear what
# needs to be done for phase2 tracker in premixing stage2.
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(pixelDigitizer,
    AddPixelInefficiency = False # will be added in DataMixer
)

from SimTracker.SiPhase2Digitizer.phase2TrackerDigitizer_cfi import phase2TrackerDigitizer as _phase2TrackerDigitizer

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toReplaceWith(pixelDigitizer, _phase2TrackerDigitizer)

from Configuration.Eras.Modifier_phase2_tracker_postTDR_cff import phase2_tracker_postTDR
phase2_tracker_postTDR.toReplaceWith(pixelDigitizer, _phase2TrackerDigitizer)


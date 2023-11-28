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

# ESProducer for SiPixelQuality with "forDigitizer" label
from CalibTracker.SiPixelESProducers.siPixelQualityForDigitizerESProducer_cfi import *

# Customize here instead of SiPixelSimBlock as the latter is imported
# also to DataMixer configuration, and the original version is needed
# there in stage2. Customize before phase2_tracker because this
# customization applies only to phase0/1 pixel.
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(pixelDigitizer,
    AddPixelInefficiency = False # will be added in DataMixer
)

from SimTracker.SiPhase2Digitizer.phase2TrackerDigitizer_cfi import phase2TrackerDigitizer as _phase2TrackerDigitizer, _premixStage1ModifyDict
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toReplaceWith(pixelDigitizer, _phase2TrackerDigitizer.clone()) # have to clone here in order to not change the original with further customizations

# Customize here instead of phase2TrackerDigitizer as the latter is
# imported also to DataMixer configuration, and the original version
# is needed there in stage2.
(premix_stage2 & phase2_tracker).toModify(pixelDigitizer, **_premixStage1ModifyDict)
from CalibTracker.SiPixelESProducers.PixelFEDChannelCollectionProducer_cfi import *

# Run-dependent MC
from Configuration.ProcessModifiers.runDependentForPixel_cff import runDependentForPixel
(runDependentForPixel & premix_stage1).toModify(pixelDigitizer, 
         UseReweighting = False,
         applyLateReweighting = False,
         store_SimHitEntryExitPoints = True,
         AdcFullScale = 1023,
         MissCalibrate = False
)


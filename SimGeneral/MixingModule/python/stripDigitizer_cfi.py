import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.SiStripGainSimESProducer_cfi import *
from SimGeneral.MixingModule.SiStripSimParameters_cfi import SiStripSimBlock

stripDigitizer = cms.PSet(
    SiStripSimBlock,
    accumulatorType = cms.string("SiStripDigitizer"),
    hitsProducer = cms.string('g4SimHits'),
    makeDigiSimLinks = cms.untracked.bool(True)
    )

from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
premix_stage1.toModify(stripDigitizer, makeDigiSimLinks = False)

# Customize here instead of SiStripSimBlock as the latter is imported
# also to DataMixer configuration, and the original version is needed
# there. Customize before phase2_tracker because this customization
# applies only to phase0 strips, and at the moment it is unclear what
# needs to be done for phase2 tracker in premixing stage2.
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(stripDigitizer,
    Noise = False, # will be added in DataMixer
    PreMixingMode = True, #Special mode to save all hit strips
    FedAlgorithm = 5, # special ZS mode: accept adc>0
    includeAPVSimulation = False  # APV simulation is off for the MixingModule configuration in premix stage2
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify( stripDigitizer, ROUList = ["g4SimHitsTrackerHitsPixelBarrelLowTof",
                                                         "g4SimHitsTrackerHitsPixelEndcapLowTof"]
)


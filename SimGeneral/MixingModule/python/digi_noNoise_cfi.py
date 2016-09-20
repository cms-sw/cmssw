import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
import SimGeneral.MixingModule.digitizers_cfi

theDigitizersNoNoise = SimGeneral.MixingModule.digitizers_cfi.theDigitizers.clone()

theDigitizersNoNoise.hcal.doNoise = cms.bool(False)
theDigitizersNoNoise.hcal.doEmpty = cms.bool(False)
theDigitizersNoNoise.hcal.doIonFeedback = cms.bool(False)
theDigitizersNoNoise.hcal.doThermalNoise = cms.bool(False)
theDigitizersNoNoise.hcal.doTimeSlew = cms.bool(False)
theDigitizersNoNoise.ecal.doENoise = cms.bool(False)
theDigitizersNoNoise.ecal.doESNoise = cms.bool(False)
theDigitizersNoNoise.ecal.applyConstantTerm = cms.bool(False)
theDigitizersNoNoise.ecal.EcalPreMixStage1 = cms.bool(True)
theDigitizersNoNoise.hcal.HcalPreMixStage1 = cms.bool(True)

# no pixel in fastsim era
if hasattr(theDigitizersNoNoise,"pixel"):
    theDigitizersNoNoise.pixel.AddNoise = cms.bool(True)
    theDigitizersNoNoise.pixel.addNoisyPixels = cms.bool(False)
    theDigitizersNoNoise.pixel.AddPixelInefficiency = cms.bool(False) #done in second step
    theDigitizersNoNoise.pixel.makeDigiSimLinks = cms.untracked.bool(False)
# no strip in fastsim era
if hasattr(theDigitizersNoNoise,"strip"):
    theDigitizersNoNoise.strip.Noise = cms.bool(False)
    theDigitizersNoNoise.strip.PreMixingMode = cms.bool(True)
    theDigitizersNoNoise.strip.FedAlgorithm = cms.int32(5) # special ZS mode: accept adc>0
    theDigitizersNoNoise.strip.makeDigiSimLinks = cms.untracked.bool(False)
theDigitizersNoNoiseValid = cms.PSet(
    theDigitizersNoNoise,
    mergedtruth = SimGeneral.MixingModule.digitizers_cfi.trackingParticles
    )
if hasattr(theDigitizersNoNoiseValid,"pixel"):
    theDigitizersNoNoiseValid.pixel.makeDigiSimLinks = cms.untracked.bool(True)
if hasattr(theDigitizersNoNoiseValid,"strip"):
    theDigitizersNoNoiseValid.strip.makeDigiSimLinks = cms.untracked.bool(True)

import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi import hcalSimBlock

hcalDigitizer = cms.PSet(
    hcalSimBlock,
    accumulatorType = cms.string("HcalDigiProducer"),
    makeDigiSimLinks = cms.untracked.bool(False))

_newFactors = cms.vdouble(
    210.55, 197.93, 186.12, 189.64, 189.63,
    189.96, 190.03, 190.11, 190.18, 190.25,
    190.32, 190.40, 190.47, 190.54, 190.61,
    190.69, 190.83, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94,
    190.94, 190.94, 190.94, 190.94, 190.94 )

from Configuration.StandardSequences.Eras import eras
eras.phase2_hgcal.toModify( hcalDigitizer,
    HBHEUpgradeQIE = cms.bool(True),
    HFUpgradeQIE = cms.bool(True),
    TestNumbering = cms.bool(True),
    hb = dict(
        photoelectronsToAnalog = cms.vdouble([57.5]*16),
        pixels = cms.int32(27370),
        sipmDarkCurrentuA = cms.double(0.055),
        sipmCrossTalk = cms.double(0.32)
    ),
    he = dict( samplingFactors = _newFactors,
        photoelectronsToAnalog = cms.vdouble([57.5]*len(_newFactors)),
        pixels = cms.int32(27370),
        sipmDarkCurrentuA = cms.double(0.055),
        sipmCrossTalk = cms.double(0.32)
    )
)

import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi import hcalSimBlock

hcalDigitizer = cms.PSet(
    hcalSimBlock,
    accumulatorType = cms.string("HcalDigiProducer"),
    makeDigiSimLinks = cms.untracked.bool(False))

def _modifyHcalDigitizerForHGCal( obj ):
    newFactors = cms.vdouble(
        210.55, 197.93, 186.12, 189.64, 189.63,
        189.96, 190.03, 190.11, 190.18, 190.25,
        190.32, 190.40, 190.47, 190.54, 190.61,
        190.69, 190.83, 190.94, 190.94, 190.94,
        190.94, 190.94, 190.94, 190.94, 190.94,
        190.94, 190.94, 190.94, 190.94, 190.94,
        190.94, 190.94, 190.94, 190.94, 190.94,
        190.94, 190.94, 190.94, 190.94, 190.94 )    
    obj.HBHEUpgradeQIE = True
    obj.hb.siPMCells = cms.vint32([1])
    obj.hb.photoelectronsToAnalog = cms.vdouble([10.]*16)
    obj.hb.pixels = cms.int32(4500*4*2)
    obj.he.samplingFactors = newFactors
    obj.he.photoelectronsToAnalog = cms.vdouble([10.]*len(newFactors))
    obj.he.pixels = cms.int32(4500*4*2)
    obj.HFUpgradeQIE = True
    obj.HcalReLabel.RelabelHits=cms.untracked.bool(True)

from Configuration.StandardSequences.Eras import eras
eras.phase2_hgcal.toModify( hcalDigitizer, func=_modifyHcalDigitizerForHGCal)

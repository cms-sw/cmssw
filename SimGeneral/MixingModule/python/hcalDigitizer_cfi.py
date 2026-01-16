import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi import hcalSimBlock

hcalDigitizer = cms.PSet(
    hcalSimBlock,
    accumulatorType = cms.string("HcalDigiProducer"),
    makeDigiSimLinks = cms.untracked.bool(False))

##
## Disable all noise for the tau embedding methods simulation step
##
from Configuration.ProcessModifiers.tau_embedding_sim_cff import tau_embedding_sim
tau_embedding_sim.toModify(
    hcalDigitizer,
    doNoise=False,
    doThermalNoise=False,
)

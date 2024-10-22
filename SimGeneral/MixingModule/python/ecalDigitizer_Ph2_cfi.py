import FWCore.ParameterSet.Config as cms

from SimCalorimetry.EcalSimProducers.ecalDigiParameters_Ph2_cff import *
from SimCalorimetry.EcalSimProducers.apdSimParameters_cff import *
from SimCalorimetry.EcalSimProducers.componentDigiParameters_cff import *
from SimCalorimetry.EcalSimProducers.ecalSimParameterMap_cff import *
from SimCalorimetry.EcalSimProducers.ecalElectronicsSim_Ph2_cff import *
from SimCalorimetry.EcalSimProducers.ecalNotContainmentSim_cff import *
from SimCalorimetry.EcalSimProducers.ecalCosmicsSim_cff import *


ecalDigitizer_Ph2 = cms.PSet(
    ecal_digi_parameters,
    apd_sim_parameters,
    component_digi_parameters,
    ecal_electronics_sim,
    ecal_cosmics_sim,
    ecal_sim_parameter_map_ph2,
    ecal_notCont_sim,
    hitsProducer = cms.string('g4SimHits'),
    accumulatorType = cms.string("EcalDigiProducer_Ph2"),
    makeDigiSimLinks = cms.untracked.bool(False),
    doEB = cms.bool(True)
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(ecalDigitizer_Ph2, hitsProducer = "fastSimProducer")


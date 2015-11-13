import FWCore.ParameterSet.Config as cms

from SimCalorimetry.EcalSimProducers.ecalDigiParameters_cff import *
from SimCalorimetry.EcalSimProducers.apdSimParameters_cff import *
from SimCalorimetry.EcalSimProducers.ecalSimParameterMap_cff import *
from SimCalorimetry.EcalSimProducers.ecalElectronicsSim_cff import *
from SimCalorimetry.EcalSimProducers.esElectronicsSim_cff import *
from SimCalorimetry.EcalSimProducers.ecalNotContainmentSim_cff import *
from SimCalorimetry.EcalSimProducers.ecalCosmicsSim_cff import *

ecalDigitizer = cms.PSet(
    ecal_digi_parameters,
    apd_sim_parameters,
    ecal_electronics_sim,
    ecal_cosmics_sim,
    ecal_sim_parameter_map,
    ecal_notCont_sim,
    es_electronics_sim,
    hitsProducer = cms.string('g4SimHits'),
    accumulatorType = cms.string("EcalDigiProducer"),
    makeDigiSimLinks = cms.untracked.bool(False)
)

from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    ecalDigitizer.hitsProducer = cms.string("famosSimHits")
    
ecalDigitizer.doEB = cms.bool(True)
ecalDigitizer.doEE = cms.bool(True)
ecalDigitizer.doES = cms.bool(True)


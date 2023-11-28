import FWCore.ParameterSet.Config as cms

from SimCalorimetry.EcalSimProducers.ecalDigiParameters_cff import *
from SimCalorimetry.EcalSimProducers.apdSimParameters_cff import *
from SimCalorimetry.EcalSimProducers.componentDigiParameters_cff import *
from SimCalorimetry.EcalSimProducers.ecalSimParameterMap_cff import *
from SimCalorimetry.EcalSimProducers.ecalElectronicsSim_cff import *
from SimCalorimetry.EcalSimProducers.esElectronicsSim_cff import *
from SimCalorimetry.EcalSimProducers.ecalNotContainmentSim_cff import *
from SimCalorimetry.EcalSimProducers.ecalCosmicsSim_cff import *


ecalDigitizer = cms.PSet(
    ecal_digi_parameters,
    apd_sim_parameters,
    component_digi_parameters,
    ecal_electronics_sim,
    ecal_cosmics_sim,
    ecal_sim_parameter_map,
    ecal_notCont_sim,
    es_electronics_sim,
    hitsProducer = cms.string('g4SimHits'),
    accumulatorType = cms.string("EcalDigiProducer"),
    makeDigiSimLinks = cms.untracked.bool(False)
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(ecalDigitizer, hitsProducer = "fastSimProducer")
    
ecalDigitizer.doEB = cms.bool(True)
ecalDigitizer.doEE = cms.bool(True)
ecalDigitizer.doES = cms.bool(True)



from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( ecalDigitizer, doEE = cms.bool(False) )

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify( ecalDigitizer, doES = cms.bool(False) )

#phase 2 digitization
from Configuration.Eras.Modifier_phase2_ecal_devel_cff import phase2_ecal_devel
from SimGeneral.MixingModule.ecalDigitizer_Ph2_cfi import ecalDigitizer_Ph2 as _ecalDigitizer_Ph2
phase2_ecal_devel.toReplaceWith(ecalDigitizer,_ecalDigitizer_Ph2)

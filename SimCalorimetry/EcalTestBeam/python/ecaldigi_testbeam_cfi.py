import FWCore.ParameterSet.Config as cms

from SimCalorimetry.EcalSimProducers.ecalDigiParameters_cff import *
from SimCalorimetry.EcalSimProducers.apdSimParameters_cff import *
from SimCalorimetry.EcalSimProducers.ecalSimParameterMap_cff import *
from SimCalorimetry.EcalSimProducers.ecalElectronicsSim_cff import *
from SimCalorimetry.EcalSimProducers.esElectronicsSim_cff import *
from SimCalorimetry.EcalSimProducers.ecalNotContainmentSim_cff import *
from SimCalorimetry.EcalSimProducers.ecalCosmicsSim_cff import *
from RecoTBCalo.EcalTBTDCReconstructor.EcalTBSimTDCRanges_cff import *
from SimGeneral.MixingModule.mixNoPU_cfi import *

simEcalUnsuppressedDigis = cms.EDProducer("EcalTBDigiProducer",
    ecal_digi_parameters,
    apd_sim_parameters,
    ecal_electronics_sim,
    ecal_cosmics_sim,
    ecal_sim_parameter_map,
    ecal_notCont_sim,
    es_electronics_sim,

    EcalTBSimTDCRanges,                                      

    use2004OffsetConvention = cms.untracked.bool(False),
    tunePhaseShift = cms.double(0.5),
    EcalTBInfoLabel = cms.untracked.string('SimEcalTBG4Object'),
    doReadout = cms.bool(True),
    EBdigiFinalCollection = cms.string('')
)

#simEcalUnsuppressedDigis.doESNoise = False
mix.digitizers.ecal.doESNoise = False

#simEcalUnsuppressedDigis.syncPhase = False
mix.digitizers.ecal.syncPhase = False

#simEcalUnsuppressedDigis.EBdigiCollection = cms.string('TEMP')
mix.digitizers.ecal.EBdigiCollection = cms.string('TEMP')

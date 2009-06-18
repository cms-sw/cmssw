import FWCore.ParameterSet.Config as cms

from SimCalorimetry.EcalSimProducers.ecalSimParameterMap_cff import *
from SimCalorimetry.EcalSimProducers.ecalElectronicsSim_cff import *
from SimCalorimetry.EcalSimProducers.ecalNotContainmentSim_cff import *
from RecoTBCalo.EcalTBTDCReconstructor.EcalTBSimTDCRanges_cff import *
simEcalUnsuppressedDigis = cms.EDProducer("EcalTBDigiProducer",
    ecal_electronics_sim,
    ecal_sim_parameter_map,
    ecal_notCont_sim,
    EcalTBSimTDCRanges,                                      
    hitsProducer = cms.string('g4SimHits'),
    use2004OffsetConvention = cms.untracked.bool(False),
    tunePhaseShift = cms.double(0.5),
    EBdigiCollection = cms.string(''),
    EEdigiCollection = cms.string(''),
    EcalTBInfoLabel = cms.untracked.string('SimEcalTBG4Object'),
    doReadout = cms.bool(True),
    #  vdouble CorrelatedNoiseMatrix = { 1.00, 0.67, 0.53, 0.44, 0.39, 0.36, 0.38, 0.35, 0.36, 0.32,
    #                                    0.67, 1.00, 0.67, 0.53, 0.44, 0.39, 0.36, 0.38, 0.35, 0.36,
    #                                    0.53, 0.67, 1.00, 0.67, 0.53, 0.44, 0.39, 0.36, 0.38, 0.35,
    #                                    0.44, 0.53, 0.67, 1.00, 0.67, 0.53, 0.44, 0.39, 0.36, 0.38,
    #                                    0.39, 0.44, 0.53, 0.67, 1.00, 0.67, 0.53, 0.44, 0.39, 0.36,
    #                                    0.36, 0.39, 0.44, 0.53, 0.67, 1.00, 0.67, 0.53, 0.44, 0.39,
    #                                    0.38, 0.36, 0.39, 0.44, 0.53, 0.67, 1.00, 0.67, 0.53, 0.44,
    #                                    0.35, 0.38, 0.36, 0.39, 0.44, 0.53, 0.67, 1.00, 0.67, 0.53,
    #                                    0.36, 0.35, 0.38, 0.36, 0.39, 0.44, 0.53, 0.67, 1.00, 0.67,
    #                                    0.32, 0.36, 0.35, 0.38, 0.36, 0.39, 0.44, 0.53, 0.67, 1.00 }
    CorrelatedNoiseMatrix = cms.vdouble(1.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 1.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 1.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 1.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 1.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        1.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 1.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 1.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 1.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 1.0)
)

simEcalUnsuppressedDigis.syncPhase = False


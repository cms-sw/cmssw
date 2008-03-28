import FWCore.ParameterSet.Config as cms

from SimCalorimetry.EcalSimProducers.ecalSimParameterMap_cff import *
from SimCalorimetry.EcalSimProducers.ecalElectronicsSim_cff import *
from SimCalorimetry.EcalSimProducers.ecalNotContainmentSim_cff import *
ecalUnsuppressedDigis = cms.EDProducer("EcalTBDigiProducer",
    ecal_electronics_sim,
    ecal_sim_parameter_map,
    ecal_notCont_sim,
    use2004OffsetConvention = cms.untracked.bool(False),
    tunePhaseShift = cms.double(0.5),
    EBdigiCollection = cms.string(''),
    EcalTBInfoLabel = cms.untracked.string('SimEcalTBG4Object'),
    doReadout = cms.bool(True),
    tdcRanges = cms.VPSet(cms.PSet(
        endRun = cms.int32(999999),
        tdcMax = cms.vdouble(1008.0, 927.0, 927.0, 927.0, 927.0),
        startRun = cms.int32(-1),
        tdcMin = cms.vdouble(748.0, 400.0, 400.0, 400.0, 400.0)
    )),
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
    CorrelatedNoiseMatrix = cms.vdouble(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
)

ecalUnsuppressedDigis.syncPhase = False


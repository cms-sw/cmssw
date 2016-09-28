
import FWCore.ParameterSet.Config as cms

# Module to create simulated RPC digis.

simMuonRPCDigis = cms.EDProducer("RPCDigiProducer",
    Noise = cms.bool(True),
    digiModelConfig = cms.PSet(
        signalPropagationSpeed = cms.double(0.66),
        timingRPCOffset = cms.double(50.0),
        Frate = cms.double(1.0),
        printOutDigitizer = cms.bool(False),
        cosmics = cms.bool(False),
        deltatimeAdjacentStrip = cms.double(3.0),
        linkGateWidth = cms.double(20.0),
        Rate = cms.double(0.0),
        timeResolution = cms.double(2.5),
        averageClusterSize = cms.double(1.5),
        Gate = cms.double(25.0),
        averageEfficiency = cms.double(0.95),
        Nbxing = cms.int32(9),
        timeJitter = cms.double(1.0),
        IRPC_time_resolution = cms.double(0.1),
        IRPC_electronics_jitter = cms.double(0.025)
    ),
    doBkgNoise = cms.bool(True), #False - no noise and bkg simulation
    Signal = cms.bool(True),
    mixLabel = cms.string('mix'),                                 
    InputCollection = cms.string('g4SimHitsMuonRPCHits'),
    digiModel = cms.string('RPCSimAsymmetricCls')
)

#the digitizer for PhaseII muon upgrade is RPCSimModelTiming and for the moment is based on  RPCSimAverageNoiseEffCls
from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen():
    simMuonRPCDigis.InputCollection = 'MuonSimHitsMuonRPCHits'
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
if phase2_muon.isChosen():
    simMuonRPCDigis = cms.EDProducer("RPCandIRPCDigiProducer",
                                     Noise = cms.bool(True),
                                     digiModelConfig = cms.PSet(
            signalPropagationSpeed = cms.double(0.66),
            timingRPCOffset = cms.double(50.0),
            Frate = cms.double(1.0),
            printOutDigitizer = cms.bool(False),
            cosmics = cms.bool(False),
            deltatimeAdjacentStrip = cms.double(3.0),
            linkGateWidth = cms.double(20.0),
            Rate = cms.double(0.0),
            timeResolution = cms.double(2.5),
            averageClusterSize = cms.double(1.5),
            Gate = cms.double(25.0),
            averageEfficiency = cms.double(0.95),
            Nbxing = cms.int32(9),
            timeJitter = cms.double(1.0),
            IRPC_time_resolution = cms.double(0.1),
            IRPC_electronics_jitter = cms.double(0.025)
            ),
                                     doBkgNoise = cms.bool(True), #False - no noise and bkg simulation
                                     Signal = cms.bool(True),
                                     mixLabel = cms.string('mix'),                                 
                                     InputCollection = cms.string('g4SimHitsMuonRPCHits'),
                                     digiModel = cms.string('RPCSimAverageNoiseEffCls'),
                                     digiIRPCModelConfig = cms.PSet(
            signalPropagationSpeed = cms.double(0.66),
            timingRPCOffset = cms.double(50.0),
            Frate = cms.double(1.0),
            printOutDigitizer = cms.bool(False),
            cosmics = cms.bool(False),
            deltatimeAdjacentStrip = cms.double(3.0),
            linkGateWidth = cms.double(20.0),
            Rate = cms.double(0.0),
            timeResolution = cms.double(2.5),
            averageClusterSize = cms.double(1.5),
            Gate = cms.double(25.0),
            averageEfficiency = cms.double(0.95),
            Nbxing = cms.int32(9),
            timeJitter = cms.double(1.0),
            IRPC_time_resolution = cms.double(0.1),
            IRPC_electronics_jitter = cms.double(0.025)
            ),       
                                     digiIRPCModel = cms.string('RPCSimModelTiming')
                                     )

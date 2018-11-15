
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
        BX_range = cms.int32(5),	
        timeJitter = cms.double(1.0),
        IRPC_time_resolution = cms.double(0.1),
        IRPC_electronics_jitter = cms.double(0.025),
        digitizeElectrons = cms.bool(False), # False - do not digitize electron hits (they are included in bkg simulation configured with doBkgNoise)
    ),
    doBkgNoise = cms.bool(True), #False - no noise and bkg simulation
    Signal = cms.bool(True),
    mixLabel = cms.string('mix'),                                 
    InputCollection = cms.string('g4SimHitsMuonRPCHits'),
    digiModel = cms.string('RPCSimAsymmetricCls')
)

#the digitizer for PhaseII muon upgrade is RPCSimModelTiming and for the moment is based on  RPCSimAverageNoiseEffCls
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(simMuonRPCDigis, InputCollection = 'MuonSimHitsMuonRPCHits')

_simMuonRPCDigisPhaseII = cms.EDProducer("RPCandIRPCDigiProducer",
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
        timeResolution = cms.double(1.5),
        averageClusterSize = cms.double(1.5),
        Gate = cms.double(25.0),
        averageEfficiency = cms.double(0.95),
        Nbxing = cms.int32(9),
        BX_range = cms.int32(5),
        timeJitter = cms.double(0.1),
        sigmaY = cms.double(2.), # resolution of 2 cm
        do_Y_coordinate = cms.bool(False),
        digitizeElectrons = cms.bool(True),
        IRPC_time_resolution = cms.double(1.5),# intrinsic time resolution of 1.5 ns
        IRPC_electronics_jitter = cms.double(0.1)# resolution of 100 ps
    ),
    doBkgNoise = cms.bool(False), #False - no noise and bkg simulation
    Signal = cms.bool(True),
    mixLabel = cms.string('mix'),
    InputCollection = cms.string('g4SimHitsMuonRPCHits'),
    digiModel = cms.string('RPCSimModelTiming'),
    digiIRPCModelConfig = cms.PSet(
        signalPropagationSpeed = cms.double(0.66),
        timingRPCOffset = cms.double(50.0),
        Frate = cms.double(1.0),
        printOutDigitizer = cms.bool(False),
        cosmics = cms.bool(False),
        deltatimeAdjacentStrip = cms.double(3.0),
        linkGateWidth = cms.double(20.0),
        Rate = cms.double(0.0),
        timeResolution = cms.double(1.0),
        averageClusterSize = cms.double(1.5),
        Gate = cms.double(25.0),
        averageEfficiency = cms.double(0.95),
        Nbxing = cms.int32(9),
        BX_range = cms.int32(5),
        timeJitter = cms.double(0.1),
        IRPC_time_resolution = cms.double(1),# resolution of 1 ns
        IRPC_electronics_jitter = cms.double(0.1),# resolution of 100 ps
        sigmaY = cms.double(2.), # resolution of 2 cm
        do_Y_coordinate = cms.bool(True),
        digitizeElectrons = cms.bool(True),
    ),
    digiIRPCModel = cms.string('RPCSimModelTiming')
)

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith( simMuonRPCDigis, _simMuonRPCDigisPhaseII )

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(simMuonRPCDigis, mixLabel = "mixData")

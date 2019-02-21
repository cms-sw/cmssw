import FWCore.ParameterSet.Config as cms

RPSiDetDigitizer = cms.EDProducer("RPDigiProducer",

    # all distances in [mm]
    # RPDigiProducer
    ROUList = cms.vstring('TotemHitsRP'),
    RPVerbosity = cms.int32(0),
    RPDigiSimHitRelationsPresistence = cms.bool(False), # save links betweend digi, clusters and OSCAR/Geant4 hits

    # RPDetDigitizer
    RPEquivalentNoiseCharge300um = cms.double(1000.0),
    RPNoNoise = cms.bool(False),

    # RPDisplacementGenerator
    RPDisplacementOn = cms.bool(False),

    # RPLinearChargeCollectionDrifter
    RPGeVPerElectron = cms.double(3.61e-09),
    RPInterStripSmearing = cms.vdouble(0.011),

    # RPLinearChargeDivider
    RPLandauFluctuations = cms.bool(True),
    RPChargeDivisionsPerStrip = cms.int32(15),
    RPChargeDivisionsPerThickness = cms.int32(5),
    RPDeltaProductionCut = cms.double(0.120425),    # [MeV]

    # RPLinearInduceChargeOnStrips
    RPInterStripCoupling = cms.double(1.0), # fraction of charge going to the strip, the missing part is taken by its neighbours

    # RPVFATSimulator
    RPVFATTriggerMode = cms.int32(2),
    RPVFATThreshold = cms.double(9000.0),
    RPDeadStripProbability = cms.double(0.001),
    RPDeadStripSimulationOn = cms.bool(False),

    # RPSimTopology
    RPSharingSigmas = cms.double(5.0), # how many sigmas taken into account for the edges and inter strips
    RPTopEdgeSmearing = cms.double(0.011),
    RPBottomEdgeSmearing = cms.double(0.011),
    RPActiveEdgeSmearing = cms.double(0.013),
    RPActiveEdgePosition = cms.double(0.034),   # from the physical edge
    RPTopEdgePosition = cms.double(1.5),
    RPBottomEdgePosition = cms.double(1.5),

    mixLabel = cms.string("mix"),
    InputCollection = cms.string("g4SimHitsTotemHitsRP")

)



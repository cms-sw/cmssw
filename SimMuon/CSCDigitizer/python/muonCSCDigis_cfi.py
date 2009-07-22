import FWCore.ParameterSet.Config as cms

simMuonCSCDigis = cms.EDProducer("CSCDigiProducer",
    strips = cms.PSet(
        peakTimeSigma = cms.double(3.0),
        timeBitForBxZero = cms.int32(6),
        doNoise = cms.bool(True),
        nScaBins = cms.int32(8),
        doCrosstalk = cms.bool(True),
        pedestal = cms.double(600.0),
        gainsConstant = cms.double(0.27),
        signalStartTime = cms.double(-250.0),
        shapingTime = cms.int32(100),
        comparatorTimeOffset = cms.double(15.0),
        bunchTimingOffsets = cms.vdouble(0.0, 20.0, 20.0, 45.0, 45.0, 
            45.0, 45.0, 45.0, 45.0, 45.0, 
            45.0),
        # parameters for tuning timing
        comparatorTimeBinOffset = cms.double(3.0),
        scaPeakBin = cms.int32(5),
        pedestalSigma = cms.double(1.5),
        signalStopTime = cms.double(500.0),
        readBadChannels = cms.bool(False),
        readBadChambers = cms.bool(True),
        gain = cms.double(2.0), ## counts per fC

        capacativeCrosstalk = cms.double(35.0),
        samplingTime = cms.double(25.0),
        resistiveCrosstalkScaling = cms.double(1.8),
        me11gain = cms.double(4.0),
        doSuppression = cms.bool(False),
        tailShaping = cms.int32(2),
        ampGainSigma = cms.double(0.03),
        doCorrelatedNoise = cms.bool(True)
    ),
    doNeutrons = cms.bool(False),
#    neutrons = cms.PSet(
#        luminosity = cms.double(0.1),
#        eventOccupancy = cms.vdouble(0.000709, 0.000782, 0.000162, 0.000162, 0.00238, 
#            0.000141, 0.00101, 0.000126, 0.000129),
#        startTime = cms.double(-400.0),
#        reader = cms.string('ROOT'),
#        input = cms.FileInPath('SimMuon/CSCDigitizer/data/CSCNeutronHits.root'),
#        endTime = cms.double(200.0)
#    ),
    wires = cms.PSet(
        signalStopTime = cms.double(150.0),
        wireTimingError = cms.double(0.0),
        signalStartTime = cms.double(-100.0),
        peakTimeSigma = cms.double(0.0),
        shapingTime = cms.int32(30),
        readBadChannels = cms.bool(False),
        timeBitForBxZero = cms.int32(6),
        samplingTime = cms.double(5.0),
        bunchTimingOffsets = cms.vdouble(0.0, 13.0, 13.0, 28.0, 28.0, 
            28.0, 28.0, 28.0, 28.0, 28.0, 
            28.0),
        tailShaping = cms.int32(2),
        doNoise = cms.bool(True)
    ),
    InputCollection = cms.string('g4SimHitsMuonCSCHits'),
    stripConditions = cms.string('Database'),
    GeometryType = cms.string('idealForDigi'),                            
    digitizeBadChambers = cms.bool(False),
    layersNeeded = cms.uint32(2)
)




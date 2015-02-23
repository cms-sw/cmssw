import FWCore.ParameterSet.Config as cms

# This object is used to customise for different running scenarios, e.g. run2
from Configuration.StandardSequences.Eras import eras

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
        # bunchTimingOffsets
        # Latest tuning by Vadim Khotilovich 16-Nov-2012 based on SingleMuPt10 relval sample.
        # Validation plots: http://khotilov.web.cern.ch/khotilov/csc/digiBunchTimingOffsets/
        # [Previous tuning by Chris Farrell
        # http://indico.cern.ch/getFile.py/access?contribId=5&resId=0&materialId=slides&confId=111101]
        bunchTimingOffsets = cms.vdouble(0.00, 40.52, 39.27, 57.28, 49.32, 56.27, 56.23, 54.73, 56.13, 53.65, 53.27),
        signalSpeed = cms.vdouble(0.0, -78, -76, -188, -262, -97, -99, -90, -99, -99, -113),
        timingCalibrationError = cms.vdouble(0., 4.2, 4.2, 0., 0., 0., 0., 0., 0., 0., 0.),
        # parameters for tuning timing
        scaTimingOffsets =  cms.vdouble(0.0, 10., 10., 0.,0.,0.,0.,0.,0.,0.,0.),
        comparatorTimeBinOffset = cms.double(3.0),
        comparatorSamplingTime = cms.double(25.0),
        scaPeakBin = cms.int32(5),
        pedestalSigma = cms.double(1.5),
        signalStopTime = cms.double(500.0),
        readBadChannels = cms.bool(False),
        readBadChambers = cms.bool(True),
        CSCUseTimingCorrections = cms.bool(False),
        CSCUseGasGainCorrections = cms.bool(False),
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
        signalStopTime = cms.double(300.0),
        # again, from http://indico.cern.ch/getFile.py/access?contribId=5&resId=0&materialId=slides&confId=111101
        timingCalibrationError = cms.vdouble(0., 6.2, 6.2, 0., 0., 0., 0., 0., 0., 0., 0.),
        signalStartTime = cms.double(-200.0),
        signalSpeed = cms.vdouble(0.0, -700, 900, 160, 146, 148, 117, 131, 107, 123, 123),
        peakTimeSigma = cms.double(0.0),
        shapingTime = cms.int32(30),
        readBadChannels = cms.bool(False),
        timeBitForBxZero = cms.int32(6),
        samplingTime = cms.double(5.0),
        # bunchTimingOffsets - comments for strips (above) also apply
        bunchTimingOffsets = cms.vdouble(0.00, 21.64, 21.64, 28.29, 29.36, 29.33, 28.57, 28.61, 28.83, 29.09, 28.22),
        tailShaping = cms.int32(2),
        doNoise = cms.bool(True)
    ),
   
    mixLabel = cms.string("mix"),
    InputCollection = cms.string("g4SimHitsMuonCSCHits"),

    stripConditions = cms.string('Database'),
    GeometryType = cms.string('idealForDigi'),                            
    digitizeBadChambers = cms.bool(False),
    layersNeeded = cms.uint32(3)
)

##
## Change the the bunch timing offsets if running in Run 2
##
eras.run2_common.toModify( simMuonCSCDigis.strips, bunchTimingOffsets=[0.0, 37.53, 37.66, 55.4, 48.2, 54.45, 53.78, 53.38, 54.12, 51.98, 51.28] )
eras.run2_common.toModify( simMuonCSCDigis.wires, bunchTimingOffsets=[0.0, 22.88, 22.55, 29.28, 30.0, 30.0, 30.5, 31.0, 29.5, 29.1, 29.88] )


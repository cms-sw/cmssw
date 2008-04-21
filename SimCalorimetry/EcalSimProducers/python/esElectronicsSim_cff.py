import FWCore.ParameterSet.Config as cms

es_electronics_sim = cms.PSet(
    doESNoise = cms.bool(True),
    ESGain = cms.int32(1),
    ESMIPkeV = cms.double(81.08),
    numESdetId = cms.int32(137216),
    ESNoiseSigma = cms.double(3.0),
    ESMIPADC = cms.double(9.0),
    zsThreshold = cms.double(2.98595),
    refHistosFile = cms.string('SimCalorimetry/EcalSimProducers/data/esRefHistosFile.txt'),
    ESBaseline = cms.int32(1000),
    doFast = cms.bool(False)
)


import FWCore.ParameterSet.Config as cms

es_electronics_sim = cms.PSet(
    doESNoise = cms.bool(True),
    numESdetId = cms.int32(137216),
    zsThreshold = cms.double(2.98595),
    refHistosFile = cms.string('SimCalorimetry/EcalSimProducers/data/esRefHistosFile.txt'),
    doFast = cms.bool(False)
)


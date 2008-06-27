import FWCore.ParameterSet.Config as cms

process = cms.Process("h4ValidData")
# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:./ECALH4TB_data_hits.root')
)

process.tbValidData = cms.EDFilter("EcalTBValidation",
    rootfile = cms.untracked.string('EcalTBValidationData.root'),
    eventHeaderProducer = cms.string('ecalTBunpack'),
    hitProducer = cms.string('ecal2006TBWeightUncalibRecHit'),
    digiCollection = cms.string(''),
    tdcRecInfoCollection = cms.string('EcalTBTDCRecInfo'),
    data_ = cms.untracked.int32(0),
    digiProducer = cms.string('ecalUnsuppressedDigis'),
    xtalInBeam = cms.untracked.int32(1104),
    hitCollection = cms.string('EcalUncalibRecHitsEB'),
    hodoRecInfoProducer = cms.string('ecal2006TBHodoscopeReconstructor'),
    eventHeaderCollection = cms.string(''),
    hodoRecInfoCollection = cms.string('EcalTBHodoscopeRecInfo'),
    tdcRecInfoProducer = cms.string('ecal2006TBTDCReconstructor')
)

process.p = cms.Path(process.tbValidData)


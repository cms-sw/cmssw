import FWCore.ParameterSet.Config as cms

process = cms.Process("h4ValidSimul")
# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:./ECALH4TB_detsim_hits.root')
)

process.tbValidSimul = cms.EDAnalyzer("EcalTBValidation",
    rootfile = cms.untracked.string('EcalTBValidationSimul.root'),
    eventHeaderProducer = cms.string('SimEcalEventHeader'),
    hitProducer = cms.string('ecalTBSimWeightUncalibRecHit'),
    digiCollection = cms.string(''),
    tdcRecInfoCollection = cms.string('EcalTBTDCRecInfo'),
    data_ = cms.untracked.int32(0),
    digiProducer = cms.string('simEcalUnsuppressedDigis'),
    xtalInBeam = cms.untracked.int32(1104),
    hitCollection = cms.string('EcalUncalibRecHitsEB'),
    hodoRecInfoProducer = cms.string('ecalTBSimHodoscopeReconstructor'),
    eventHeaderCollection = cms.string(''),
    hodoRecInfoCollection = cms.string('EcalTBHodoscopeRecInfo'),
    tdcRecInfoProducer = cms.string('ecalTBSimTDCReconstructor')
)

process.p = cms.Path(process.tbValidSimul)


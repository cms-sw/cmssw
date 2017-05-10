import FWCore.ParameterSet.Config as cms

process = cms.Process("TPInputAn")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:myfile.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.tpInputAnalyzer = cms.EDAnalyzer("EcalTPInputAnalyzer",
    ProducerEblabel = cms.InputTag('RecHits',''),
    ProducerEelabel = cms.InputTag('RecHits','')
)

process.p = cms.Path(process.tpInputAnalyzer)



import FWCore.ParameterSet.Config as cms

process = cms.Process("TPInputAn")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:myfile.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.tpInputAnalyzer = cms.EDAnalyzer("EcalTPInputAnalyzer",
    EBLabel = cms.string(''),
    EELabel = cms.string(''),
    Producer = cms.string('RecHits')
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('histos.root'),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

process.p = cms.Path(process.tpInputAnalyzer)



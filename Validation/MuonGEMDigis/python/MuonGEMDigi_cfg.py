import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        'file:../data/out_digi.root'
    )
)

process.demo = cms.EDAnalyzer('MuonGEMDigis',
	outputFile = cms.untracked.string('valid.root'),
	simMuonGEMDigis = cms.InputTag("simMuonGEMDigis")
)


process.p = cms.Path(process.demo)

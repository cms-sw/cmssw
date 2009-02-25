import FWCore.ParameterSet.Config as cms

process = cms.Process('demo')

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.MessageLogger.cerr.FwkReport.reportEvery=cms.untracked.int32(10)

# run over files
readfiles = cms.untracked.vstring()
readfiles.extend( ['file:test.root'] )
process.source = cms.Source ("PoolSource",
                             fileNames = readfiles)

# setup the analyzer
process.hcalnoiseinfoanalyzer = cms.EDAnalyzer(
    'HcalNoiseInfoAnalyzer',
    rbxCollName = cms.string('hcalnoiseinfoproducer'),
    rootHistFilename = cms.string('plots.root')
    )

process.p = cms.Path(process.hcalnoiseinfoanalyzer)

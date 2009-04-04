import FWCore.ParameterSet.Config as cms

process = cms.Process('demo')

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.MessageLogger.cerr.FwkReport.reportEvery=cms.untracked.int32(10)

# run over files
readfiles = cms.untracked.vstring()
readfiles.extend( ['file:noise_skim.root'] )
process.source = cms.Source ("PoolSource",
                             fileNames = readfiles)

# setup the analyzer
process.hcalnoiseinfoanalyzer = cms.EDAnalyzer(
    'HcalNoiseInfoAnalyzer',
    rbxCollName = cms.string('hcalnoise'),
    rootHistFilename = cms.string('plots.root'),
    noisetype = cms.int32(3)
    )

process.p = cms.Path(process.hcalnoiseinfoanalyzer)

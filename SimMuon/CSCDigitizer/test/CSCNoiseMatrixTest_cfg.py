import FWCore.ParameterSet.Config as cms

process = cms.Process('CSCNoiseMatrixTest')

process.source = cms.Source("EmptySource")

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.Geometry_cff')
process.load("SimMuon.CSCDigitizer.muonCSCDigis_cfi")
process.GlobalTag.globaltag = 'IDEAL_30X::All'

process.cscNoiseTest = cms.EDAnalyzer("CSCNoiseMatrixTest",
      readBadChannels = cms.bool(False),
      readBadChambers = cms.bool(True),
      doCrosstalk = cms.bool(True),
      gainsConstant = cms.double(0.27),
      capacativeCrosstalk = cms.double(35.0),
      resistiveCrosstalkScaling = cms.double(1.8),
      doCorrelatedNoise = cms.bool(True))

process.RandomNumberGeneratorService.cscNoiseTest = cms.PSet(
                   engineName = cms.untracked.string('HepJamesRandom'),
                   initialSeed = cms.untracked.uint32(1234)
)


process.path = cms.Path(process.cscNoiseTest)

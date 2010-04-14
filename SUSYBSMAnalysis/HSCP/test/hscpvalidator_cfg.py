import FWCore.ParameterSet.Config as cms

process = cms.Process("HSCPValidator")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    #duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    fileNames = cms.untracked.vstring(
      'rfio:/castor/cern.ch/user/s/scooper/hscp/354/MGStop130-GEN-SIM_10.root'
    )
)

process.load("PhysicsTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = 'hscpValidatorPlots.root'

process.hscpValidator = cms.EDAnalyzer('HSCPValidator',
  generatorLabel = cms.InputTag("generator"),
  particleIds = cms.vint32(
      # stop R-hadrons
      1000006,
      -1000006
     )

)


process.p = cms.Path(process.hscpValidator)

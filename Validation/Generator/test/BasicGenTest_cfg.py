import FWCore.ParameterSet.Config as cms

process = cms.Process('Validation')

process.load("Validation.Generator.BasicGenTest_cff")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.Validation = cms.EDFilter("BasicGenTest")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

### ADD YOUR FILES HERE ####
process.PoolSource.fileNames = [       
        'file:BasicGenTest_Minbias_pythia_V6_416.root'
       ]

process.p1 = cms.Path(process.Validation*process.dqmSaver)

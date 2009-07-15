import FWCore.ParameterSet.Config as cms

process = cms.Process('Validation')

process.load("Validation.Generator.BasicGenTest_cff")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.Validation = cms.EDFilter("BasicGenTest")

#process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring ('file:/uscms/home/jaz8600/CMSSW_3_1_0_pre11/src/Validation/RecoB/data/TTbar.root'))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.PoolSource.fileNames = [       
               '/store/relval/CMSSW_3_1_1/RelValProdMinBias/GEN-SIM-RAW/MC_31X_V2-v1/0002/EA036AF9-716B-DE11-8C20-000423D98844.root',
        '/store/relval/CMSSW_3_1_1/RelValProdMinBias/GEN-SIM-RAW/MC_31X_V2-v1/0002/DCDFC91A-736B-DE11-89B7-001D09F2514F.root',
        '/store/relval/CMSSW_3_1_1/RelValProdMinBias/GEN-SIM-RAW/MC_31X_V2-v1/0002/C64673CF-E16B-DE11-ABBA-000423D6B5C4.root'
       ]




process.p1 = cms.Path(process.Validation*process.dqmSaver)

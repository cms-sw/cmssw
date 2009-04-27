import FWCore.ParameterSet.Config as cms

process = cms.Process("Sim")

process.load("PhysicsTools.HepMCCandAlgos.genParticles_cfi")

process.load("Configuration.StandardSequences.Simulation_cff")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

process.load("Configuration.EventContent.EventContent_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(11),
        mix = cms.untracked.uint32(12345),
        VtxSmeared = cms.untracked.uint32(98765432)
    ),
    sourceSeed = cms.untracked.uint32(123456789)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    'file:/uscms_data/d2/kjsmith/CMSSW_3_1_0_pre3/src/tt_on.root',
    #'/store/mc/Winter09/Zjets-madgraph/GEN-SIM-DIGI-RECO/IDEAL_V11_FastSim_v1/0041/0405D546-3ED1-DD11-8CF9-003048322C3A.root'
    #'/store/mc/Winter09/Zjets-madgraph/GEN-SIM-DIGI-RECO/IDEAL_V11_FastSim_v1/0041/083C9E8B-36D1-DD11-817E-003048322CAA.root'
    #'/store/mc/Fall08/ZJets-madgraph/GEN-SIM-RECO/IDEAL_V9_v1_pre-production/0000/16E344F9-7CBC-DD11-A10E-00093D1142A9.root',
    #'/store/mc/Fall08/ZJets-madgraph/GEN-SIM-RECO/IDEAL_V9_v1_pre-production/0000/7272D65C-7EBC-DD11-A3D0-00093D13B582.root'

      )
)

process.myanalyzer = cms.EDAnalyzer("NewAnalyzer",
    OutputFilename = cms.untracked.string('ZJets_Spring.root')
)

process.p0 = cms.Path(process.genParticles)
process.p1 = cms.Path(process.myanalyzer)
process.schedule = cms.Schedule(process.p0, process.p1)




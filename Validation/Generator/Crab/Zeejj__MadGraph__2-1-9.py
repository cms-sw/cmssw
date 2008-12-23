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
    'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/0.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/1.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/2.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/3.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/4.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/5.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/6.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/7.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/8.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/9.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/10.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/11.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/12.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/13.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/14.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/15.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/16.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/17.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/18.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/19.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/20.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/21.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/22.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/23.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/24.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/25.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/26.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/27.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/28.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/29.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/30.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/31.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/32.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/33.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/34.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/35.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/36.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/37.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/38.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/39.root',
       'file:/uscmst1b_scratch/lpc1/3DayLifetime/kjsmith/ZJets/40.root'
      )
)

process.myanalyzer = cms.EDAnalyzer("NewAnalyzer",
    OutputFilename = cms.untracked.string('ZJets_MG.root')
)

process.p0 = cms.Path(process.genParticles)
process.p1 = cms.Path(process.myanalyzer)
process.schedule = cms.Schedule(process.p0, process.p1)




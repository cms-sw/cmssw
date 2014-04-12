import FWCore.ParameterSet.Config as cms

process = cms.Process("HecFP420Test")


process.load("Configuration.StandardSequences.Generator_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.load("Geometry.FP420CommonData.FP420GeometryXML_cfi")

process.load("SimTransport.HectorProducer.HectorTransport_cfi")

process.load("Configuration.EventContent.EventContent_cff")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(True),
    ignoreTotal = cms.untracked.int32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.load('Configuration/Generator/PythiaMinBias_cfi')
process.generator.comEnergy = cms.double(10000.0)

process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep edmHepMCProduct_*_*_*',
        'keep SimTracks_*_*_*',
        'keep SimVertexs_*_*_*',
        'keep PSimHitCrossingFrame_mix_FP420SI_*', 
        'keep PSimHits_*_FP420SI_*'),
    fileName = cms.untracked.string('HecMBDef.root')
)

process.Timing = cms.Service("Timing")
process.Tracer = cms.Service("Tracer")
process.p1 = cms.Path(process.generator*process.VtxSmeared*process.LHCTransport*process.g4SimHits)
process.outpath = cms.EndPath(process.o1)
process.schedule = cms.Schedule(process.p1,process.outpath)
process.g4SimHits.Physics.DefaultCutValue =  1000.
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Generator.ApplyEtaCuts = False
process.g4SimHits.Generator.HepMCProductLabel = 'LHCTransport'
process.g4SimHits.Generator.ApplyPhiCuts = cms.bool(False)
process.g4SimHits.SteppingAction.MaxTrackTime = cms.double(2000.0)
process.g4SimHits.StackingAction.MaxTrackTime = cms.double(2000.0)
#rocess.FP420Digi.ApplyTofCut = cms.bool(False)

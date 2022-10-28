import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD

process = cms.Process('Dump',Run3_DDD)

# import of standard configurations
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtended2021_cff') 
process.load('SimG4CMS.Muon.muonSimHitDump_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HitStudy=dict()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring('file:step1_ZMM_ddd.root'),
)

process.analysis_step   = cms.Path(process.muonSimHitDump)

process.muonSimHitDump.MaxEvent = 10

# Schedule definition
process.schedule = cms.Schedule(process.analysis_step)

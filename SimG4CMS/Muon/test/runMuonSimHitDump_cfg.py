import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep

process = cms.Process('Dump',Run3_dd4hep)

# import of standard configurations
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryDD4hepExtended2021_cff') 
process.load('SimG4CMS.Muon.muonSimHitDump_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HitStudy=dict()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring('file:step1_ZMM_dd4hep.root'),
)

process.analysis_step   = cms.Path(process.muonSimHitDump)

process.muonSimHitDump.MaxEvent = 10

# Schedule definition
process.schedule = cms.Schedule(process.analysis_step)

import FWCore.ParameterSet.Config as cms

process = cms.Process("TPGVERIF")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:TrigPrim_unsuppressed.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.tpgcheck = cms.EDAnalyzer("TPGCheck",
    Producer = cms.string(''),
    Label = cms.string('simEcalTriggerPrimitiveDigis')
)

process.Timing = cms.Service("Timing")

process.MessageLogger = cms.Service("MessageLogger")

process.p = cms.Path(process.tpgcheck)



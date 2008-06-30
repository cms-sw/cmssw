import FWCore.ParameterSet.Config as cms

process = cms.Process("TPGVERIF")
process.source = cms.Source("PoolSource",
    maxEvents = cms.untracked.int32(-1),
    fileNames = cms.untracked.vstring('file:TrigPrim.root')
)

process.tpgcheck = cms.EDFilter("TPGCheck",
    Producer = cms.string(''),
    Label = cms.string('ecalTriggerPrimitiveDigis')
)

process.Timing = cms.Service("Timing")

process.MessageLogger = cms.Service("MessageLogger")

process.p = cms.Path(process.tpgcheck)



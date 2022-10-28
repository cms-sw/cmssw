import FWCore.ParameterSet.Config as cms

process = cms.Process("FIRST")

process.source = cms.Source("EmptySource")

process.first = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string("stat_sender_first.root"))
process.b = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string("stat_sender_b.root"))
process.c = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string("stat_sender_c.root"))
process.d = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string("stat_sender_d.root"))
process.e = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string("stat_sender_e.root"))

process.Thing = cms.EDProducer("ThingProducer")
process.OtherThing = cms.EDProducer("OtherThingProducer")
process.EventNumber = cms.EDProducer("EventNumberIntProducer")


process.o = cms.EndPath(process.first+process.b+process.c+process.d+process.e, cms.Task(process.Thing, process.OtherThing, process.EventNumber))

process.maxEvents.input = 10

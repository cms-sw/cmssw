import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:stat_sender_first.root"))

process.b = cms.EDProducer("SecondaryProducer",
    seq = cms.untracked.bool(True),
    input = cms.SecSource("EmbeddedRootSource",
        sequential = cms.untracked.bool(True),
        fileNames = cms.untracked.vstring('file:stat_sender_b.root')
    )
)

process.c = cms.EDProducer("SecondaryProducer",
    seq = cms.untracked.bool(True),
    input = cms.SecSource("EmbeddedRootSource",
        sequential = cms.untracked.bool(True),
        fileNames = cms.untracked.vstring('file:stat_sender_c.root')
    )
)

process.d = cms.EDProducer("SecondaryProducer",
    seq = cms.untracked.bool(True),
    input = cms.SecSource("EmbeddedRootSource",
        sequential = cms.untracked.bool(True),
        fileNames = cms.untracked.vstring('file:stat_sender_d.root')
    )
)

process.e = cms.EDProducer("SecondaryProducer",
    seq = cms.untracked.bool(True),
    input = cms.SecSource("EmbeddedRootSource",
        sequential = cms.untracked.bool(True),
        fileNames = cms.untracked.vstring('file:stat_sender_e.root')
    )
)

process.pB = cms.Path(process.b)
process.pC = cms.Path(process.c)
process.pD = cms.Path(process.d)
process.pE = cms.Path(process.e)

process.add_(cms.Service("StatisticsSenderService", debug = cms.untracked.bool(True)))
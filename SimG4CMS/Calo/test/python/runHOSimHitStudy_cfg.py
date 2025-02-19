import FWCore.ParameterSet.Config as cms

process = cms.Process("SimHitStudy")

process.load("SimG4CMS.Calo.HOSimHitStudy_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('FwkJob', 'HitStudy'),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HitStudy = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    )
)

#process.Timing = cms.Service("Timing")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:simevent030eta13.root')
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('simHitStudy_30_13.root')
)

process.hoSimHitStudy.MaxEnergy = 60.0
process.hoSimHitStudy.ScaleEB   = 1.02
process.hoSimHitStudy.ScaleHB   = 104.4
process.hoSimHitStudy.ScaleHO   = 2.33

process.p1 = cms.Path(process.hoSimHitStudy)


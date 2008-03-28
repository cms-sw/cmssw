import FWCore.ParameterSet.Config as cms

roadPainter = cms.EDFilter("RoadPainter",
    # multi page PS, works only with PS, give picture name without extension
    PictureName = cms.untracked.string('output'),
    RingLabel = cms.untracked.string(''),
    RoadLabel = cms.untracked.string('')
)



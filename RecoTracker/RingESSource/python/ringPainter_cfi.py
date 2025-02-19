import FWCore.ParameterSet.Config as cms

ringPainter = cms.EDAnalyzer("RingPainter",
    PictureName = cms.untracked.string('rings.pdf'),
    RingLabel = cms.untracked.string('')
)



import FWCore.ParameterSet.Config as cms

ringTest = cms.EDAnalyzer("RingTest",
    RingLabel = cms.untracked.string(''),
    DumpRings = cms.untracked.bool(True),
    FileName = cms.untracked.string('rings.dat')
)



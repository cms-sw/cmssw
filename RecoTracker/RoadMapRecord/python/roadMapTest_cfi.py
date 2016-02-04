import FWCore.ParameterSet.Config as cms

roadMapTest = cms.EDAnalyzer("RoadMapTest",
    RoadLabel = cms.untracked.string(''),
    DumpRoads = cms.untracked.bool(True),
    FileName = cms.untracked.string('roads.dat')
)



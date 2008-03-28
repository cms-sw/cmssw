import FWCore.ParameterSet.Config as cms

roadMapTest = cms.EDFilter("RoadMapTest",
    RoadLabel = cms.untracked.string(''),
    DumpRoads = cms.untracked.bool(True),
    FileName = cms.untracked.string('roads.dat')
)



import FWCore.ParameterSet.Config as cms

roads = cms.ESSource("RoadMapESSource",
    ComponentName = cms.string(''),
    RingsLabel = cms.string(''),
    InputFileName = cms.FileInPath('RecoTracker/RoadMapESSource/data/roads-0010.dat')
)



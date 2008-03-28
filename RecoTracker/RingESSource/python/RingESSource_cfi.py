import FWCore.ParameterSet.Config as cms

rings = cms.ESSource("RingESSource",
    ComponentName = cms.string(''),
    InputFileName = cms.FileInPath('RecoTracker/RingESSource/data/rings-0004.dat')
)



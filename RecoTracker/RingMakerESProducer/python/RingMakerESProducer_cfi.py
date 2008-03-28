import FWCore.ParameterSet.Config as cms

rings = cms.ESProducer("RingMakerESProducer",
    DumpDetIds = cms.untracked.bool(False),
    # component name
    ComponentName = cms.string(''),
    RingAsciiFileName = cms.untracked.string('rings.dat'),
    DetIdsDumpFileName = cms.untracked.string('tracker_detids.dat'),
    # write out ascii dump of roads to file
    WriteOutRingsToAsciiFile = cms.untracked.bool(False),
    # configuration: FULL, TIFTOB
    Configuration = cms.untracked.string('FULL')
)



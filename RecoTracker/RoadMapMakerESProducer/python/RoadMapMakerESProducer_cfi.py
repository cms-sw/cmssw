import FWCore.ParameterSet.Config as cms

roads = cms.ESProducer("RoadMapMakerESProducer",
    # geometry structure, allowed values: FullDetector, MTCC, FullDetectorII
    GeometryStructure = cms.string('FullDetector'),
    # component name
    ComponentName = cms.string(''),
    # label for rings service
    RingsLabel = cms.string(''),
    # write out ascii dump of roads to file
    WriteOutRoadMapToAsciiFile = cms.untracked.bool(False),
    # seeding type, allowed values: FourRingSeeds, TwoRingSeeds
    SeedingType = cms.string('FourRingSeeds'),
    RoadMapAsciiFile = cms.untracked.string('roads.dat')
)



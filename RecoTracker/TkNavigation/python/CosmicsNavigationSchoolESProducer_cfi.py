#do not comment that line
import FWCore.ParameterSet.Config as cms

#cosmicsNavigationSchoolESProducer = cms.ESProducer("NavigationSchoolESProducer",
#    ComponentName = cms.string('CosmicNavigationSchool')
#)

cosmicsNavigationSchoolESProducer = cms.ESProducer("SkippingLayerCosmicNavigationSchoolESProducer",
                                                   ComponentName = cms.string('CosmicNavigationSchool'),
                                                   noPXB = cms.bool(False),
                                                   noPXF = cms.bool(False),
                                                   noTIB = cms.bool(False),
                                                   noTID = cms.bool(False),
                                                   noTOB = cms.bool(False),
                                                   noTEC = cms.bool(False),
                                                   selfSearch = cms.bool(True),
                                                   allSelf = cms.bool(True)
                                                   )



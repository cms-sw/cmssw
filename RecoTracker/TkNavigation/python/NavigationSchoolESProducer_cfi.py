import FWCore.ParameterSet.Config as cms

navigationSchoolESProducer = cms.ESProducer("NavigationSchoolESProducer",
    ComponentName = cms.string('SimpleNavigationSchool'),
    SimpleMagneticField = cms.string('')
#    SimpleMagneticField = cms.string('ParabolicMf')
)



import FWCore.ParameterSet.Config as cms

from RecoTracker.TkNavigation.navigationSchoolESProducer_cfi import navigationSchoolESProducer as _navigationSchoolESProducer
navigationSchoolESProducer = _navigationSchoolESProducer.clone(
    ComponentName = 'SimpleNavigationSchool',
    PluginName    = 'SimpleNavigationSchool',
    SimpleMagneticField = ''
#   SimpleMagneticField = 'ParabolicMf'
)

navigationSchoolESProducerParabolicMf = navigationSchoolESProducer.clone(
    ComponentName = 'SimpleNavigationSchoolParabolicMf',
    SimpleMagneticField = 'ParabolicMf'
)

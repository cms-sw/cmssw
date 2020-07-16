import FWCore.ParameterSet.Config as cms

from RecoTracker.TkNavigation.NavigationSchoolESProducer_cfi import *
beamHaloNavigationSchoolESProducer = navigationSchoolESProducer.clone(
    ComponentName       = 'BeamHaloNavigationSchool',
    SimpleMagneticField = '',
)

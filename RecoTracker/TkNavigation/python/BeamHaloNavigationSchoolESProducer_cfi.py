import FWCore.ParameterSet.Config as cms

from RecoTracker.TkNavigation.NavigationSchoolESProducer_cfi import *
beamHaloNavigationSchoolESProducer = navigationSchoolESProducer.clone()
beamHaloNavigationSchoolESProducer.ComponentName       = cms.string('BeamHaloNavigationSchool')
beamHaloNavigationSchoolESProducer.SimpleMagneticField = cms.string('')



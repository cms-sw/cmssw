import FWCore.ParameterSet.Config as cms
import RecoTracker.TrackProducer.TrackProducer_cfi

# for parabolic magnetic field
from Configuration.ProcessModifiers.trackingParabolicMf_cff import trackingParabolicMf

TrackProducer = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    useSimpleMF = cms.bool(True),
    SimpleMagneticField = cms.string("ParabolicMf"),
    Propagator = cms.string('PropagatorWithMaterialParabolicMf'),
    TTRHBuilder = cms.string('WithAngleAndTemplateWithoutProbQ')
)
trackingParabolicMf.toModify(TrackProducer, NavigationSchool = 'SimpleNavigationSchoolParabolicMf')

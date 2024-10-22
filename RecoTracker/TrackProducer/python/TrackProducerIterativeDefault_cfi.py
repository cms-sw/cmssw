import FWCore.ParameterSet.Config as cms
import RecoTracker.TrackProducer.TrackProducer_cfi

# for parabolic magnetic field
from Configuration.ProcessModifiers.trackingParabolicMf_cff import trackingParabolicMf

TrackProducerIterativeDefault = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    useSimpleMF = True,
    SimpleMagneticField = 'ParabolicMf',
    Propagator = 'PropagatorWithMaterialParabolicMf',
    TTRHBuilder = 'WithAngleAndTemplateWithoutProbQ'
)
trackingParabolicMf.toModify(TrackProducerIterativeDefault, NavigationSchool = 'SimpleNavigationSchoolParabolicMf')

import FWCore.ParameterSet.Config as cms
import RecoTracker.TrackProducer.TrackProducer_cfi

TrackProducer = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    useSimpleMF = cms.bool(True),
    SimpleMagneticField = cms.string("ParabolicMf"),
    Propagator = cms.string('PropagatorWithMaterialParabolicMf'),
    TTRHBuilder = cms.string('WithAngleAndTemplateWithoutProbQ')
)

import FWCore.ParameterSet.Config as cms




PropagatorWithMaterialForLoopers = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(4.0),      #default was 1.6
    ComponentName = cms.string('PropagatorWithMaterialForLoopers'),
    Mass = cms.double(0.1396),      #default was 0.105
    PropagationDirection = cms.string('alongMomentum'),
    useRungeKutta = cms.bool(False),
# If ptMin > 0, uncertainty in reconstructed momentum will be taken into account when estimating rms scattering angle.
# (By default, it is neglected). However, it will also be assumed that the track pt can't be below specified value,
# to prevent this scattering angle becoming too big.                                    
    ptMin = cms.double(-1),
    SimpleMagneticField = cms.string(''),
#    SimpleMagneticField = cms.string('ParabolicMf'),
    # Use new AnalyticalPropagator's logic for intersection between plane and helix (for loopers)
    useOldAnalPropLogic = cms.bool(False)
)


PropagatorWithMaterialForLoopersOpposite = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(4.0),     #default was 1.6
    ComponentName = cms.string('PropagatorWithMaterialForLoopersOpposite'),
    Mass = cms.double(0.1396),     #default was 0.105
    PropagationDirection = cms.string('oppositeToMomentum'),
    useRungeKutta = cms.bool(False),
# If ptMin > 0, uncertainty in reconstructed momentum will be taken into account when estimating rms scattering angle.
# (By default, it is neglected). However, it will also be assumed that the track pt can't be below specified value,
# to prevent this scattering angle becoming too big.                                    
    ptMin = cms.double(-1),
    SimpleMagneticField = cms.string(''),
#    SimpleMagneticField = cms.string('ParabolicMf'),
    # Use new AnalyticalPropagator's logic for intersection between plane and helix (for loopers)
    useOldAnalPropLogic = cms.bool(False)
)

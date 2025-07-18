import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.steppingHelixPropagatorESProducer_cfi import steppingHelixPropagatorESProducer as _steppingHelixPropagatorESProducer
SteppingHelixPropagator =  _steppingHelixPropagatorESProducer.clone(
    ComponentName = 'SteppingHelixPropagator',
    NoErrorPropagation = False,
    PropagationDirection = 'alongMomentum',
    useTuningForL2Speed = False,
    useIsYokeFlag = True,
    endcapShiftInZNeg = 0.0,
    SetVBFPointer = False,
    AssumeNoMaterial = False,
    endcapShiftInZPos = 0.0,
    useInTeslaFromMagField = False,
    VBFName = 'VolumeBasedMagneticField',
    useEndcapShiftsInZ = False,
    sendLogWarning = False,
    useMatVolumes = True,
    debug = False,
    #This sort of works but assumes a measurement at propagation origin  
    ApplyRadX0Correction = True,
    useMagVolumes = True,
    returnTangentPlane = True
)



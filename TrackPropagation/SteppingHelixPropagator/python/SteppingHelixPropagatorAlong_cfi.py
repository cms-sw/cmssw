import FWCore.ParameterSet.Config as cms

SteppingHelixPropagatorAlong = cms.ESProducer("SteppingHelixPropagatorESProducer",
    ComponentName = cms.string('SteppingHelixPropagatorAlong'),
    NoErrorPropagation = cms.bool(False),
    PropagationDirection = cms.string('alongMomentum'),
    useTuningForL2Speed = cms.bool(False),
    useIsYokeFlag = cms.bool(True),
    endcapShiftInZNeg = cms.double(0.0),
    SetVBFPointer = cms.bool(False),
    AssumeNoMaterial = cms.bool(False),
    endcapShiftInZPos = cms.double(0.0),
    useInTeslaFromMagField = cms.bool(False),
    VBFName = cms.string('VolumeBasedMagneticField'),
    useEndcapShiftsInZ = cms.bool(False),
    sendLogWarning = cms.bool(False),
    useMatVolumes = cms.bool(True),
    debug = cms.bool(False),
    #This sort of works but assumes a measurement at propagation origin  
    ApplyRadX0Correction = cms.bool(True),
    useMagVolumes = cms.bool(True),
    returnTangentPlane = cms.bool(True)
)



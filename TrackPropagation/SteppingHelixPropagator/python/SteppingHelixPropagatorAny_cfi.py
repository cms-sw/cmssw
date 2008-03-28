import FWCore.ParameterSet.Config as cms

SteppingHelixPropagatorAny = cms.ESProducer("SteppingHelixPropagatorESProducer",
    PropagationDirection = cms.string('anyDirection'),
    useTuningForL2Speed = cms.bool(False),
    useIsYokeFlag = cms.bool(True),
    NoErrorPropagation = cms.bool(False),
    SetVBFPointer = cms.bool(False),
    AssumeNoMaterial = cms.bool(False),
    returnTangentPlane = cms.bool(True),
    useInTeslaFromMagField = cms.bool(False),
    VBFName = cms.string('VolumeBasedMagneticField'),
    sendLogWarning = cms.bool(False),
    useMatVolumes = cms.bool(True),
    debug = cms.bool(False),
    #This sort of works but assumes a measurement at propagation origin  
    ApplyRadX0Correction = cms.bool(True),
    useMagVolumes = cms.bool(True),
    ComponentName = cms.string('SteppingHelixPropagatorAny')
)



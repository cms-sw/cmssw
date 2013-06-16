import FWCore.ParameterSet.Config as cms

def customise(process):

    process.g4SimHits.Physics.RusRoEcalGamma = cms.double(0.1)
    process.g4SimHits.Physics.RusRoEcalGammaLimit = cms.double(5.)
    process.g4SimHits.Physics.RusRoHcalGamma = cms.double(0.1)
    process.g4SimHits.Physics.RusRoHcalGammaLimit = cms.double(5.)

    process.g4SimHits.StackingAction.RusRoEcalNeutron = cms.double(0.1)
    process.g4SimHits.StackingAction.RusRoEcalNeutronLimit = cms.double(10.)
    process.g4SimHits.StackingAction.RusRoHcalNeutron = cms.double(0.1)
    process.g4SimHits.StackingAction.RusRoHcalNeutronLimit = cms.double(10.)

    return(process)

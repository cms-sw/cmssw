import FWCore.ParameterSet.Config as cms

def customiseRR(process):

    # Russian roulette enabled
  if hasattr(process,'g4SimHits'):
    process.g4SimHits.Physics.RusRoGammaEnergyLimit = cms.double(5.)
    process.g4SimHits.Physics.RusRoEcalGamma     = cms.double(1.0)
    process.g4SimHits.Physics.RusRoHcalGamma     = cms.double(0.3)
    process.g4SimHits.Physics.RusRoMuonIronGamma = cms.double(0.3)
    process.g4SimHits.Physics.RusRoPreShowerGamma= cms.double(1.0)
    process.g4SimHits.Physics.RusRoWorldGamma    = cms.double(0.3)

    process.g4SimHits.StackingAction.RusRoEcalNeutronLimit = cms.double(10.)
    process.g4SimHits.StackingAction.RusRoEcalNeutron     = cms.double(0.1)
    process.g4SimHits.StackingAction.RusRoHcalNeutron     = cms.double(0.1)
    process.g4SimHits.StackingAction.RusRoMuonIronNeutron = cms.double(0.1)
    process.g4SimHits.StackingAction.RusRoPreShowerNeutron= cms.double(0.1)
    process.g4SimHits.StackingAction.RusRoWorldNeutron    = cms.double(0.1)

    return(process)

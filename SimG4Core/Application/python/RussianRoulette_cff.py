import FWCore.ParameterSet.Config as cms

def customiseRR(process):

    # Russian roulette disabled - by default it is enabled
  if hasattr(process,'g4SimHits'):
    process.g4SimHits.StackingAction.RusRoGammaEnergyLimit = cms.double(0.0)
    process.g4SimHits.StackingAction.RusRoNeutronEnergyLimit = cms.double(0.0)

    return(process)

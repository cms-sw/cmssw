import FWCore.ParameterSet.Config as cms

def customise(process):

  # a fragment showing options for exotica simulation
  if hasattr(process,'g4SimHits'):
    process.g4SimHits.Physics.monopoleCharge       = cms.untracked.int32(1)
    process.g4SimHits.Physics.MonopoleDeltaRay     = cms.untracked.bool(True)
    process.g4SimHits.Physics.MonopoleMultiScatter = cms.untracked.bool(False)
    process.g4SimHits.Physics.MonopoleTransport    = cms.untracked.bool(True)
    process.g4SimHits.Physics.MonopoleMass         = cms.untracked.double(0) # GeV 
    process.g4SimHits.Physics.ExoticaTransport     = cms.untracked.bool(True)
    process.g4SimHits.Physics.ExoticaPhysicsSS     = cms.untracked.bool(False)
    process.g4SimHits.Physics.RhadronPhysics       = cms.bool(False)
    process.g4SimHits.Physics.DarkMPFactor         = cms.double(1.0)
    process.g4SimHits.Physics.particlesDef         = cms.FileInPath('')
    return(process)

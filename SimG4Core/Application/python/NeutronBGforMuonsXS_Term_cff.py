import FWCore.ParameterSet.Config as cms

def customise(process):

  # fragment allowing to simulate neutron background in muon system
  # using default neutron tracking and HP thermal neutron scattering

  from SimG4Core.Application.NeutronBGforMuons_cff import neutronBG

  process = neutronBG(process)

  if hasattr(process,'g4SimHits'):
    process.g4SimHits.Physics.ThermalNeutrons = cms.untracked.bool(True)

    return(process)


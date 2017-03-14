import FWCore.ParameterSet.Config as cms

def customise(process):

  # fragment allowing to simulate neutron background in muon system
  # using default neutron tracking 

  from SimG4Core.Application.NeutronBGforMuons_cff import neutronBG

  process = neutronBG(process)

  return(process)


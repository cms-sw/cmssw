# -*- coding: utf-8 -*-

import FWCore.ParameterSet.Config as cms
import os

def customise(process, inputProcess):

  process.particleFlowORG = process.particleFlow.clone()

  # Since CMSSW 4_4 the particleFlow reco works a bit differently. The step is
  # twofold, first particleFlowTmp is created and then the final particleFlow
  # collection. What we do in this case is that we merge the final ParticleFlow
  # collection. For the muon reconstruction, we also merge particleFlowTmp in
  # order to get PF-based isolation right.
  if hasattr(process, 'particleFlowTmp'):
    process.particleFlowTmpMixed = cms.EDProducer('PFCandidateMixer',
      col1 = cms.untracked.InputTag("cleanedParticleFlow"),
      col2 = cms.untracked.InputTag("particleFlowTmp", ""),
      trackCol = cms.untracked.InputTag("generalTracks"),

      # Don't produce value maps:
      muons = cms.untracked.InputTag(""),
      gsfElectrons = cms.untracked.InputTag("")
    )
    process.muons.PFCandidates = cms.InputTag("particleFlowTmpMixed")

    for p in process.paths:
      pth = getattr(process,p)
      if "particleFlow" in pth.moduleNames():
        pth.replace(process.particleFlow, process.particleFlowORG*process.particleFlow)
      if "muons" in pth.moduleNames():
        pth.replace(process.muons, process.particleFlowTmpMixed*process.muons)
  else:
    # CMSSW_4_2
    if hasattr(process,"famosParticleFlowSequence"):
      process.famosParticleFlowSequence.remove(process.pfPhotonTranslatorSequence)
      process.famosParticleFlowSequence.remove(process.pfElectronTranslatorSequence)
      process.famosParticleFlowSequence.remove(process.particleFlow)
      process.famosParticleFlowSequence.__iadd__(process.particleFlowORG)
      process.famosParticleFlowSequence.__iadd__(process.particleFlow)
      process.famosParticleFlowSequence.__iadd__(process.pfElectronTranslatorSequence)
      process.famosParticleFlowSequence.__iadd__(process.pfPhotonTranslatorSequence)
    elif hasattr(process,"particleFlowReco"):
      process.particleFlowReco.remove(process.pfPhotonTranslatorSequence)
      process.particleFlowReco.remove(process.pfElectronTranslatorSequence)
      process.particleFlowReco.remove(process.particleFlow)
      process.particleFlowReco.__iadd__(process.particleFlowORG)
      process.particleFlowReco.__iadd__(process.particleFlow)
      process.particleFlowReco.__iadd__(process.pfElectronTranslatorSequence)
      process.particleFlowReco.__iadd__(process.pfPhotonTranslatorSequence)
    else :
      raise "Cannot find particleFlow sequence"

    process.pfSelectedElectrons.src = cms.InputTag("particleFlowORG")
    process.pfSelectedPhotons.src   = cms.InputTag("particleFlowORG")

  process.particleFlow = cms.EDProducer('PFCandidateMixer',
    col1 = cms.untracked.InputTag("cleanedParticleFlow"),
    col2 = cms.untracked.InputTag("particleFlowORG", ""),
    trackCol = cms.untracked.InputTag("generalTracks"),
    muons = cms.untracked.InputTag("muons"),
    gsfElectrons = cms.untracked.InputTag("gsfElectrons")
    # TODO: photons ???
  )

  return process

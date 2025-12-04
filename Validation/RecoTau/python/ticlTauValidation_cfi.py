import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.ticlTauValidator_cfi import ticlTauValidator as _ticlTauValidator

# RECO: default
recoTiclTauValidator = _ticlTauValidator.clone(
    folder = cms.string("RecoTauV/ticlTauValidator")
)

# HLT
hltTiclTauValidator = _ticlTauValidator.clone(
    folder = cms.string("HLT/TICL/ticlTauValidator"),
    simTaus     = cms.InputTag("SimTauProducer"),
    TauProducer = cms.InputTag("hltHpsPFTauProducer"),
    pf          = cms.InputTag("hltParticleFlowTmp"),
    pfTmpBarrel = cms.InputTag("hltParticleFlowTmpBarrel"),
    jets        = cms.InputTag("hltAK4PFJets"),
    ticlCandidates = cms.InputTag("hltTiclCandidate"),
    simTracksters  = cms.InputTag("hltTiclSimTracksters","fromCPs"),
    allTrackstersToSimTrackstersAssociationsByLCs =
        cms.InputTag("hltAllTrackstersToSimTrackstersAssociationsByLCs",
                        "hltTiclSimTrackstersfromCPsTohltTiclCandidate"),
    genParticles = cms.InputTag("genParticles"),
    maxAssocScore = 0.6,
)
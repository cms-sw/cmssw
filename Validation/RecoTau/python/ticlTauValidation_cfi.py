import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.ticlTauValidator_cfi import ticlTauValidator as _ticlTauValidator

# RECO: default
recoTiclTauValidator = _ticlTauValidator.clone(
    TauProducer = cms.InputTag("hpsPFTauProducer"),
    folder = cms.string("Tau/ticlTauValidator"),
)

# HLT
hltTiclTauValidator = _ticlTauValidator.clone(
    folder = cms.string("HLT/Tau/ticlTauValidator"),
    simTaus     = cms.InputTag("SimTauProducer"),
    TauProducer = cms.InputTag("hltHpsPFTauProducer"),
    pf          = cms.InputTag("hltParticleFlowTmp"),
    pfTmpBarrel = cms.InputTag("hltParticleFlowTmpBarrel"),
    jets        = cms.InputTag("hltAK4PFJets"),
    simTICLCandidates = cms.InputTag("hltTiclSimTracksters"),
    simTracksters = cms.InputTag("hltTiclSimTracksters","fromCPs"),
    ticlCandidates = cms.InputTag("hltTiclCandidate"),
    simToRecoTracksterAssocByLCs = cms.InputTag(
        "hltAllTrackstersToSimTrackstersAssociationsByLCs",
        "hltTiclSimTrackstersfromCPsTohltTiclCandidate"
    ),
    recoToSimTracksterAssocByLCs = cms.InputTag(
        "hltAllTrackstersToSimTrackstersAssociationsByLCs",
        "hltTiclCandidateTohltTiclSimTrackstersfromCPs"
    ),
    genVisTaus = cms.InputTag("genVisTaus"),
    genParticles = cms.InputTag("genParticles"),
    hltProcessName = cms.string("HLT"),
    maxAssocScore = 0.6
)


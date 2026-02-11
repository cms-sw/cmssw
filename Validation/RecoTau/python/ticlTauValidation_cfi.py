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
    ticlCandidates = cms.InputTag("hltTiclTrackstersMerge"),
    simTracksters  = cms.InputTag("hltTiclSimTracksters","fromCPs"),
    simToRecoTracksterAssocByLCs =
        cms.InputTag("hltAllTrackstersToSimTrackstersAssociationsByLCs",
                        "hltTiclSimTrackstersfromCPsTohltTiclTrackstersMerge"),
    recoToSimTracksterAssocByLCs =
        cms.InputTag("hltAllTrackstersToSimTrackstersAssociationsByLCs",
                        "hltTiclTrackstersMergeTohltTiclSimTrackstersfromCPs"),
    genVisTaus = cms.InputTag("genVisTaus"),
    genParticles = cms.InputTag("genParticles"),
    maxAssocScore = 0.6,
)

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
ticl_v5.toModify(hltTiclTauValidator,
    ticlCandidates = cms.InputTag("hltTiclCandidate"),
    simToRecoTracksterAssocByLCs =
        cms.InputTag("hltAllTrackstersToSimTrackstersAssociationsByLCs",
                        "hltTiclSimTrackstersfromCPsTohltTiclCandidate"),
    recoToSimTracksterAssocByLCs =
        cms.InputTag("hltAllTrackstersToSimTrackstersAssociationsByLCs",
                        "hltTiclCandidateTohltTiclSimTrackstersfromCPs"),
)

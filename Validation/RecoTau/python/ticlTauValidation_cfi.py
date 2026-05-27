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
    simTICLCandidates = cms.InputTag("hltTiclSimTracksters"),
    simToRecoTracksterAssocByLCs =
        cms.InputTag("hltAllTrackstersToSimTrackstersAssociationsByLCs",
                        "hltTiclSimTrackstersTohltTiclTrackstersMerge"),
    recoToSimTracksterAssocByLCs =
        cms.InputTag("hltAllTrackstersToSimTrackstersAssociationsByLCs",
                        "hltTiclTrackstersMergeTohltTiclSimTracksters"),
    genVisTaus = cms.InputTag("genVisTaus"),
    genParticles = cms.InputTag("genParticles"),
    maxAssocScore = 0.6,
)

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
ticl_v5.toModify(hltTiclTauValidator,
    ticlCandidates = cms.InputTag("hltTiclCandidate"),
    simToRecoTracksterAssocByLCs =
        cms.InputTag("hltAllTrackstersToSimTrackstersAssociationsByLCs",
                        "hltTiclSimTrackstersTohltTiclCandidate"),
    recoToSimTracksterAssocByLCs =
        cms.InputTag("hltAllTrackstersToSimTrackstersAssociationsByLCs",
                        "hltTiclCandidateTohltTiclSimTracksters"),
)

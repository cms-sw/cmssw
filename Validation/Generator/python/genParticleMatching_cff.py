import FWCore.ParameterSet.Config as cms

## reco-generator matching for muons
electronMatch = cms.EDProducer("MCMatcher",       # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src         = cms.InputTag("gsfElectrons"), # RECO objects to match
    matched     = cms.InputTag("genParticles"), # mc-truth particle collection
    mcPdgId     = cms.vint32(11),               # one or more PDG ID (11 = electron); absolute values (see below)
    checkCharge = cms.bool(True),               # True = require RECO and MC objects to have the same charge
    mcStatus    = cms.vint32(3),                # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
    maxDeltaR   = cms.double(0.5),              # Minimum deltaR for the match
    maxDPtRel   = cms.double(0.5),              # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(True),     # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(False),    # False = just match input in order; True = pick lowest deltaR pair first
)

## reco-generator matching for electrons
muonMatch = cms.EDProducer("MCMatcher",           # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src     = cms.InputTag("muons"),            # RECO objects to match  
    matched = cms.InputTag("genParticles"),     # mc-truth particle collection
    mcPdgId     = cms.vint32(13),               # one or more PDG ID (13 = muon); absolute values (see below)
    checkCharge = cms.bool(True),               # True = require RECO and MC objects to have the same charge
    mcStatus = cms.vint32(3),                   # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
    maxDeltaR = cms.double(0.5),                # Minimum deltaR for the match
    maxDPtRel = cms.double(0.5),                # Minimum deltaPt/Pt for the match
    resolveAmbiguities = cms.bool(True),        # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(False),    # False = just match input in order; True = pick lowest deltaR pair first
)

## reco-generator(parton) matching for jets
jetPartonMatch = cms.EDProducer("MCMatcher",      # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src = cms.InputTag("antikt5CaloJets"),      # RECO objects to match
    matched = cms.InputTag("genParticles"),     # mc-truth particle collection
    mcPdgId  = cms.vint32(1, 2, 3, 4, 5),       # one or more PDG ID (quarks except top; gluons)
    mcStatus = cms.vint32(3),                   # PYTHIA status code (3 = hard scattering)
    checkCharge = cms.bool(False),              # False = any value of the charge of MC and RECO is ok
    maxDeltaR = cms.double(0.4),                # Minimum deltaR for the match
    maxDPtRel = cms.double(3.0),                # Minimum deltaPt/Pt for the match
    resolveAmbiguities = cms.bool(True),        # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(False),    # False = just match input in order; True = pick lowest deltaR pair first
)

## reco-generator(jets) matching for jets
jetGenJetMatch = cms.EDFilter("GenJetMatcher",  # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src      = cms.InputTag("antikt5CaloJets"), # RECO jets (any View<Jet> is ok)
    matched  = cms.InputTag("antikt5GenJets"),  # GEN jets  (must be GenJetCollection)
    mcPdgId  = cms.vint32(),                    # n/a
    mcStatus = cms.vint32(),                    # n/a
    checkCharge = cms.bool(False),              # n/a
    maxDeltaR = cms.double(0.4),                # Minimum deltaR for the match
    maxDPtRel = cms.double(3.0),                # Minimum deltaPt/Pt for the match
    resolveAmbiguities = cms.bool(True),        # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(False),    # False = just match input in order; True = pick lowest deltaR pair first
)


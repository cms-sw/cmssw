import FWCore.ParameterSet.Config as cms

# PATLayer0 Jet input
allLayer0Jets.jetSource = 'iterativeCone5CaloJets'
allLayer0Jets.removeOverlaps = cms.PSet(
    electrons = cms.PSet(
        deltaR = cms.double(0.3),
        collection = cms.InputTag("allLayer0Electrons")
    )
)
# PATLayer0 Jet matching
jetPartonMatch.src = 'allLayer0Jets'
jetPartonMatch.matched = 'genParticles'
jetPartonMatch.maxDeltaR = 0.4
jetPartonMatch.maxDPtRel = 3.0
jetPartonMatch.resolveAmbiguities = True
jetPartonMatch.resolveByMatchQuality = False
jetPartonMatch.checkCharge = False
jetPartonMatch.mcPdgId = [1, 2, 3, 4, 5, 
    21]
jetPartonMatch.mcStatus = [3]
jetGenJetMatch.src = 'allLayer0Jets'
jetGenJetMatch.matched = 'iterativeCone5GenJets'
jetGenJetMatch.maxDeltaR = 0.4
jetGenJetMatch.maxDPtRel = 3.0
jetGenJetMatch.resolveAmbiguities = True
jetGenJetMatch.resolveByMatchQuality = False
jetGenJetMatch.checkCharge = False
jetGenJetMatch.mcPdgId = []
jetGenJetMatch.mcStatus = []
# PATLayer0 Jet charge
layer0JetCharge.src = 'allLayer0Jets'
layer0JetCharge.jetTracksAssociation = 'layer0JetTracksAssociator'
layer0JetCharge.exp = 1.0
layer0JetCharge.var = 'Pt'


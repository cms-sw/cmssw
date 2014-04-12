import FWCore.ParameterSet.Config as cms

#
# module to make the MaxSumPtWMass jet combination
#
findTtSemiLepJetCombMaxSumPtWMass = cms.EDProducer("TtSemiLepJetCombMaxSumPtWMass",
    ## jet input 
    jets  = cms.InputTag("selectedPatJets"),
    ## lepton input 
    leps  = cms.InputTag("selectedPatMuons"),
    ## maximum number of jets to be considered
    maxNJets  = cms.int32(4),
    ## nominal WMass parameter (in GeV)
    wMass    = cms.double(80.4),
    ## use b-tagging two distinguish between light and b jets
    useBTagging = cms.bool(False),
    ## choose algorithm for b-tagging
    bTagAlgorithm = cms.string("trackCountingHighEffBJetTags"),
    ## minimum b discriminator value required for b jets and
    ## maximum b discriminator value allowed for non-b jets
    minBDiscBJets     = cms.double(1.0),
    maxBDiscLightJets = cms.double(3.0)
)

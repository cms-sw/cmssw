import FWCore.ParameterSet.Config as cms

#
# module to make the geom jet combination
#
findTtSemiLepJetCombGeom = cms.EDProducer("TtSemiLepJetCombGeom",
    ## jet input 
    jets  = cms.InputTag("selectedPatJets"),
    ## lepton input 
    leps  = cms.InputTag("selectedPatMuons"),
    ## maximum number of jets to be considered
    maxNJets  = cms.int32(4),
    ## use deltaR or deltaTheta
    useDeltaR = cms.bool(True),
    ## use b-tagging two distinguish between light and b jets
    useBTagging = cms.bool(False),
    ## choose algorithm for b-tagging
    bTagAlgorithm = cms.string("trackCountingHighEffBJetTags"),
    ## minimum b discriminator value required for b jets and
    ## maximum b discriminator value allowed for non-b jets
    minBDiscBJets     = cms.double(1.0),
    maxBDiscLightJets = cms.double(3.0)
)
# foo bar baz
# ffkqLQz6WRHD5
# Vi2DJVmaGxT8A

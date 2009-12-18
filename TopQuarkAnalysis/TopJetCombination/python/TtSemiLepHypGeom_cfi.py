import FWCore.ParameterSet.Config as cms

#
# module to make the geom hypothesis
#
ttSemiLepHypGeom = cms.EDProducer("TtSemiLepHypGeom",
    ## met input 
    mets  = cms.InputTag("layer1METs"),
    ## jet input 
    jets  = cms.InputTag("selectedLayer1Jets"),
    ## lepton input
    leps  = cms.InputTag("selectedLayer1Muons"),
    ## maximal number of jets to be considered
    maxNJets  = cms.int32(4),
    ## use a delta criterion or not
    useDeltaR = cms.bool(True)                               
)

import FWCore.ParameterSet.Config as cms

#pileupSummary = cms.EDProducer("PileupInformation",
addPileupInfo = cms.EDProducer("PileupInformation",
                               isPreMixed = cms.bool(True),
                               PileupSummaryInfoInputTag=cms.InputTag('mixData'),
                               BunchSpacingInputTag=cms.InputTag('mixData','bunchSpacing')
)

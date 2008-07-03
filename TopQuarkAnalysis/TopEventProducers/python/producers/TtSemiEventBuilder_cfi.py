import FWCore.ParameterSet.Config as cms

#
# module to fill the semi-leptonic ttbar event structure
#
ttSemiEvent = cms.EDProducer("TtSemiEventBuilder",
    ## choose semi-leptonic decay mode
    decay = cms.int32(2),  # 0: none
                           # 1: electron
                           # 2: muon
                           # 3: tau

    ## add genEvt (if available)
    genEvent = cms.InputTag("genEvt"),
                             
    ## considered event hypotheses                             
    hyps = cms.VInputTag(cms.InputTag("ttSemiHypothesisMaxSumPtWMass"),
                         cms.InputTag("ttSemiHypothesisGenMatch"),
                         cms.InputTag("ttSemiHypothesisMVADisc")
                         ),

    ## event hypotheses' keys for mapping
    keys = cms.VInputTag(cms.InputTag("ttSemiHypothesisMaxSumPtWMass","Key"),
                         cms.InputTag("ttSemiHypothesisGenMatch","Key"),
                         cms.InputTag("ttSemiHypothesisMVADisc","Key")
                         ),

    ## add extra information on genMatch
    genMatch = cms.PSet(
        sumPt = cms.InputTag("ttSemiJetPartonMatch","SumPt"),
        sumDR = cms.InputTag("ttSemiJetPartonMatch","SumDR"),
        match = cms.InputTag("ttSemiJetPartonMatch")
    ),

    mvaDisc = cms.PSet(
        meth = cms.InputTag("findTtSemiJetCombMVA","Meth"),
        disc = cms.InputTag("findTtSemiJetCombMVA","Disc")
    )
)



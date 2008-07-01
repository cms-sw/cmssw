import FWCore.ParameterSet.Config as cms

ttSemiEvent = cms.EDProducer("TtSemiEventBuilder",
    genMatch = cms.PSet(
        sumPt = cms.InputTag("ttSemiJetPartonMatch","SumPt"),
        sumDR = cms.InputTag("ttSemiJetPartonMatch","SumDR"),
        match = cms.InputTag("ttSemiJetPartonMatch")
    ),
    decay = cms.int32(1),
    keys = cms.VInputTag(cms.InputTag("ttSemiHypothesisMaxSumPtWMass","Key"),
                         cms.InputTag("ttSemiHypothesisGenMatch","Key"),
                         cms.InputTag("ttSemiHypothesisMVADisc","Key")),
    mvaDisc = cms.PSet(
        meth = cms.InputTag("findTtSemiJetCombMVA","Meth"),
        disc = cms.InputTag("findTtSemiJetCombMVA","Disc")
    ),
    hyps = cms.VInputTag(cms.InputTag("ttSemiHypothesisMaxSumPtWMass"),
                         cms.InputTag("ttSemiHypothesisGenMatch"),
                         cms.InputTag("ttSemiHypothesisMVADisc")),
    genEvent = cms.InputTag("genEvt")
)



import FWCore.ParameterSet.Config as cms

#
# module to fill the semi-leptonic ttbar event structure
#
ttSemiLepEvent = cms.EDProducer("TtSemiLepEvtBuilder",
    ## choose semi-leptonic decay mode
    decay = cms.int32(2),  # 0: none
                           # 1: electron
                           # 2: muon
                           # 3: tau

    ## add genEvt (if available)
    genEvent = cms.InputTag("genEvt"),
                             
    ## considered event hypotheses                             
    hyps = cms.VInputTag(cms.InputTag("ttSemiLepMaxSumPtWMass"),
                         cms.InputTag("ttSemiLepKinFit"),
                         cms.InputTag("ttSemiLepGenMatch"),
                         cms.InputTag("ttSemiLepMVADisc")
                         ),

    ## event hypotheses' keys for mapping
    keys = cms.VInputTag(cms.InputTag("ttSemiLepMaxSumPtWMass","Key"),
                         cms.InputTag("ttSemiLepGenMatch","Key"),
                         cms.InputTag("ttSemiLepKinFit","Key"),
                         cms.InputTag("ttSemiLepMVADisc","Key")
                         ),

    ## event hypotheses' jet parton association as meta information
    matches = cms.VInputTag(cms.InputTag("ttSemiLepMaxSumPtWMass","Match"),
                            cms.InputTag("ttSemiLepKinFit","Match"),
                            cms.InputTag("ttSemiLepGenMatch","Match"),
                            cms.InputTag("ttSemiLepMVADisc","Match")
                            ),

    ## add extra information on kinFit
    kinFit = cms.PSet(
        chi2 = cms.InputTag("kinFitTtSemiEvent","Chi2"),
        prob = cms.InputTag("kinFitTtSemiEvent","Prob"),
    ),

    ## add extra information on genMatch
    genMatch = cms.PSet(
        sumPt = cms.InputTag("ttSemiJetPartonMatch","SumPt"),
        sumDR = cms.InputTag("ttSemiJetPartonMatch","SumDR"),
    ),

    ## add extra information on mvaDisc
    mvaDisc = cms.PSet(
        meth = cms.InputTag("findTtSemiJetCombMVA","Meth"),
        disc = cms.InputTag("findTtSemiJetCombMVA","Disc")
    )
)



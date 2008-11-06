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

    ## set verbosity level
    verbosity = cms.int32(0),  # 0: no additional printout
                               # 1: print a summary for each event

    ## add genEvt (if available)
    genEvent = cms.InputTag("genEvt"),
                             
    ## considered event hypotheses                             
    hyps = cms.VInputTag(cms.InputTag("ttSemiLepHypGeom"),
                         cms.InputTag("ttSemiLepHypWMassMaxSumPt"),
                         cms.InputTag("ttSemiLepHypMaxSumPtWMass"),
                         cms.InputTag("ttSemiLepHypGenMatch"),
                         cms.InputTag("ttSemiLepHypKinFit"),
                         cms.InputTag("ttSemiLepHypMVADisc")
                         ),

    ## event hypotheses' keys for mapping
    keys = cms.VInputTag(cms.InputTag("ttSemiLepHypGeom","Key"),
                         cms.InputTag("ttSemiLepHypWMassMaxSumPt","Key"),
                         cms.InputTag("ttSemiLepHypMaxSumPtWMass","Key"),
                         cms.InputTag("ttSemiLepHypGenMatch","Key"),
                         cms.InputTag("ttSemiLepHypKinFit","Key"),
                         cms.InputTag("ttSemiLepHypMVADisc","Key")
                         ),

    ## event hypotheses' jet parton association as meta information
    matches = cms.VInputTag(cms.InputTag("ttSemiLepHypGeom","Match"),
                            cms.InputTag("ttSemiLepHypWMassMaxSumPt","Match"),
                            cms.InputTag("ttSemiLepHypMaxSumPtWMass","Match"),
                            cms.InputTag("ttSemiLepHypGenMatch","Match"),
                            cms.InputTag("ttSemiLepHypKinFit","Match"),
                            cms.InputTag("ttSemiLepHypMVADisc","Match")
                            ),

    ## add extra information on kinFit
    kinFit = cms.PSet(
        chi2 = cms.InputTag("kinFitTtSemiLepEventHypothesis","Chi2"),
        prob = cms.InputTag("kinFitTtSemiLepEventHypothesis","Prob"),
    ),

    ## add extra information on genMatch
    genMatch = cms.PSet(
        sumPt = cms.InputTag("ttSemiLepJetPartonMatch","SumPt"),
        sumDR = cms.InputTag("ttSemiLepJetPartonMatch","SumDR"),
    ),

    ## add extra information on mvaDisc
    mvaDisc = cms.PSet(
        meth = cms.InputTag("findTtSemiLepJetCombMVA","Meth"),
        disc = cms.InputTag("findTtSemiLepJetCombMVA","Disc")
    )
)

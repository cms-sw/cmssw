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

    ## maximum number of jets taken into account per event for each hypothesis
    ## (this parameter is used in the ttSemiLepEvtBuilder_cff to synchronize
    ## the individual maxNJets parameters)
    maxNJets = cms.int32(4),  # has to be >= 4
                              # can be set to -1 to take all jets

    ## labels for event hypotheses
    hyps = cms.vstring("ttSemiLepHypGeom",
                       "ttSemiLepHypWMassMaxSumPt",
                       "ttSemiLepHypMaxSumPtWMass",
                       "ttSemiLepHypGenMatch",
                       "ttSemiLepHypKinFit",
                       "ttSemiLepHypMVADisc"
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
        meth = cms.InputTag("findTtSemiLepJetCombMVA","Method"),
        disc = cms.InputTag("findTtSemiLepJetCombMVA","Discriminators")
    )
)

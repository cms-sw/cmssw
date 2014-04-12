import FWCore.ParameterSet.Config as cms

#
# module to fill the semi-leptonic ttbar event structure
#
ttSemiLepEvent = cms.EDProducer("TtSemiLepEvtBuilder",
    ## choose leptonic decay modes
    decayChannel1 = cms.int32(2),  # 0: none
                                   # 1: electron
                                   # 2: muon
                                   # 3: tau

    decayChannel2 = cms.int32(0),  # 0: none
                                   # 1: electron
                                   # 2: muon
                                   # 3: tau

    ## set verbosity level
    verbosity = cms.int32(0),  # 0: no additional printout
                               # 1: print a summary for each event

    ## add genEvt (if available)
    genEvent = cms.InputTag("genEvt"),

    ## labels for event hypotheses
    ## (this vector of strings can be modified using the functions
    ## addTtSemiLepHypotheses and removeTtSemiLepHypGenMatch in
    ## TopQuarkAnalysis.TopEventProducers.sequences.ttSemiLepEvtBuilder_cff)
    hypotheses = cms.VInputTag("ttSemiLepHypGenMatch"),  # "ttSemiLepHypGeom"
                                                         # "ttSemiLepHypWMassMaxSumPt"
                                                         # "ttSemiLepHypWMassDeltaTopMass"
                                                         # "ttSemiLepHypMaxSumPtWMass"
                                                         # "ttSemiLepHypKinFit"
                                                         # "ttSemiLepHypHitFit"                                
                                                         # "ttSemiLepHypMVADisc"
                                
    ## add extra information on kinFit
    kinFit = cms.PSet(
        chi2 = cms.InputTag("kinFitTtSemiLepEventHypothesis","Chi2"),
        prob = cms.InputTag("kinFitTtSemiLepEventHypothesis","Prob"),
    ),
    
    ## add extra information on hitFit
    hitFit = cms.PSet(
        chi2 = cms.InputTag("hitFitTtSemiLepEventHypothesis","Chi2"),
        prob = cms.InputTag("hitFitTtSemiLepEventHypothesis","Prob"),
        mt = cms.InputTag("hitFitTtSemiLepEventHypothesis","MT"),
        sigmt = cms.InputTag("hitFitTtSemiLepEventHypothesis","SigMT"),
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

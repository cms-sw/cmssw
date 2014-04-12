import FWCore.ParameterSet.Config as cms

#
# module to fill the full-leptonic ttbar event structure
#
ttFullLepEvent = cms.EDProducer("TtFullLepEvtBuilder",
    ## choose leptonic decay modes
    decayChannel1 = cms.int32(2),  # 0: none
                                  # 1: electron
                                  # 2: muon
                                  # 3: tau
    decayChannel2 = cms.int32(2),  # 0: none
                                  # 1: electron
                                  # 2: muon
                                  # 3: tau


    ## set verbosity level
    verbosity = cms.int32(0),  # 0: no additional printout
                               # 1: print a summary for each event

    ## add genEvt (if available)
    genEvent = cms.InputTag("genEvt"),

    ## maximum number of jets taken into account per event for each hypothesis
    ## (this parameter is used in the ttFullLepEvtBuilder_cff to synchronize
    ## the individual maxNJets parameters)
    maxNJets = cms.int32(4),  # has to be >= 2
                              # can be set to -1 to take all jets

    ## labels for event hypotheses
    hypotheses = cms.VInputTag("ttFullLepHypGenMatch",
                               "ttFullLepHypKinSolution"),
                                
    ## add extra information on kinSolution
    kinSolution = cms.PSet(
        solWeight   = cms.InputTag("kinSolutionTtFullLepEventHypothesis","solWeight"),
	wrongCharge = cms.InputTag("kinSolutionTtFullLepEventHypothesis","isWrongCharge"),
    ),

    ## add extra information on genMatch
    genMatch = cms.PSet(
        sumPt = cms.InputTag("ttFullLepJetPartonMatch","SumPt"),
        sumDR = cms.InputTag("ttFullLepJetPartonMatch","SumDR"),
    ),
)

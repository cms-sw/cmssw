import FWCore.ParameterSet.Config as cms

#
# module to fill the full-hadronic ttbar event structure
#
ttFullHadEvent = cms.EDProducer("TtFullHadEvtBuilder",
    ## choose leptonic decay modes
    decayChannel1 = cms.int32(0), # 0: none
                                  # 1: electron
                                  # 2: muon
                                  # 3: tau
    decayChannel2 = cms.int32(0), # 0: none
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
    ## addTtFullHadHypotheses and removeTtFullHadHypGenMatch in
    ## TopQuarkAnalysis.TopEventProducers.sequences.ttFullHadEvtBuilder_cff)
    hypotheses = cms.vstring("ttFullHadHypGenMatch"),  # "ttFullHadHypKinFit"
                                                       # "ttFullHadHypMVADisc"

    ## add extra information on kinFit
    kinFit = cms.PSet(
        chi2 = cms.InputTag("kinFitTtFullHadEventHypothesis","Chi2"),
        prob = cms.InputTag("kinFitTtFullHadEventHypothesis","Prob"),
    ),
                                
    ## add extra information on genMatch
    genMatch = cms.PSet(
        sumPt = cms.InputTag("ttFullHadJetPartonMatch","SumPt"),
        sumDR = cms.InputTag("ttFullHadJetPartonMatch","SumDR"),
    )
)

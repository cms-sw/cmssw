import FWCore.ParameterSet.Config as cms

#
# module to make jet-parton matches for semi-leptonic
# ttbar decays; the product will be a std::vector of
# matches, each match a std::vector<int> in the order
# (LightQ, LightQBar, HadB, LepB)
#
ttSemiLepJetPartonMatch = cms.EDProducer("TtSemiLepJetPartonMatch",
    ## sources
    jets = cms.InputTag("selectedPatJets"),

    #-------------------------------------------------
    # algorithms: totalMinDist
    #             minSumDist
    #             ptOrderedMinDist
    #             unambiguousOnly
    #-------------------------------------------------
    algorithm = cms.string("totalMinDist"),

    #-------------------------------------------------
    # use DeltaR (eta, phi) for calculating the
    # distance between jets and partons; the normal
    # space angle (theta, phi) is used otherwise
    #-------------------------------------------------
    useDeltaR = cms.bool(True),

    #-------------------------------------------------
    # do an outlier rejection based on an upper cut
    # on the distance between matched jet and parton
    # (useMaxDist = true is enforced for the
    #  unambiguousOnly algorithm)
    #-------------------------------------------------
    useMaxDist = cms.bool(False),
    maxDist    = cms.double(0.3),
                                    
    #-------------------------------------------------
    # number of jets to be considered in the matching
    # (has to be >= 4, can be set to -1 if you want to 
    # take all)
    #-------------------------------------------------
    maxNJets = cms.int32(4),

    #-------------------------------------------------
    # number of different combinations to be stored
    # (can be set to -1 if you want to take all,
    #  minSumDist is the only algorithm that provides
    #  more than only the best combination)
    #-------------------------------------------------
    maxNComb = cms.int32(1),

    #-------------------------------------------------
    # partons to be ignored in the matching;
    # "LightQ", "LightQBar", "HadB", "LepB"
    #-------------------------------------------------
    partonsToIgnore = cms.vstring(),

    #-------------------------------------------------
    # verbosity level: 0: no additional printout
    #                  1: print info for each event
    #-------------------------------------------------
    verbosity = cms.int32(0)
)



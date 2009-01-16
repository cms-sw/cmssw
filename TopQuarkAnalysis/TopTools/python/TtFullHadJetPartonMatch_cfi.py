import FWCore.ParameterSet.Config as cms

#
# module to make jet-parton matches for full-hadronic
# ttbar decays; the product will be a std::vector of
# matches, each match a std::vector<int> in the order
# (LightQTop, LightQBarTop, B, LightQTopBar,
# LightQBarTopBar, BBar)
#
ttFullHadJetPartonMatch = cms.EDProducer("TtFullHadJetPartonMatch",
    ## sources
    jets = cms.InputTag("selectedLayer1Jets"),

    #-------------------------------------------------
    # algorithms: 0 = totalMinDist
    #             1 = minSumDist
    #             2 = ptOrderedMinDist
    #             3 = unambiguousOnly
    #-------------------------------------------------
    algorithm = cms.int32(0),

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
    maxDist = cms.double(0.3),

    #-------------------------------------------------
    # number of jets to be considered in the matching
    # (has to be >= 6, can be set to -1 if you want to 
    # take all)
    #-------------------------------------------------
    maxNJets = cms.int32(6),

    #-------------------------------------------------
    # number of different combinations to be stored
    # (can be set to -1 if you want to take all,
    #  minSumDist is the only algorithm that provides
    #  more than only the best combination)
    #-------------------------------------------------
    maxNComb = cms.int32(1),

    #-------------------------------------------------
    # verbosity level: 0: no additional printout
    #                  1: print info for each event
    #-------------------------------------------------
    verbosity = cms.int32(0)
)



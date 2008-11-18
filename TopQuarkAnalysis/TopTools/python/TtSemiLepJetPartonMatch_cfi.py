import FWCore.ParameterSet.Config as cms

#
# module to make a jet parton match for semi-leptonic
# ttbar decays; the match will be a std::vector<int>
# in order  (LightQ, LightQBar, HadB, LepB)
#
ttSemiLepJetPartonMatch = cms.EDFilter("TtSemiLepJetPartonMatch",
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
    maxDist    = cms.double(0.3),
                                    
    #-------------------------------------------------
    # number of jets to be considered in the matching
    # (has to be >= 4, can be set to -1 if you want to 
    # take all)
    #-------------------------------------------------
    nJets = cms.int32(4)
)



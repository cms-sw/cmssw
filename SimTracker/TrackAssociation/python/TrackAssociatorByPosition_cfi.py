import FWCore.ParameterSet.Config as cms

TrackAssociatorByPosition = cms.ESProducer("TrackAssociatorByPositionESProducer",
    #use the delta eta-phi estimator on the momentum at a plane in the muon system
    #	string method = "momdr"
    #use the delta eta-phi estimator on the position at a plane in the muon system
    #	string method = "posdr"
    QminCut = cms.double(120.0),
    MinIfNoMatch = cms.bool(False),
    ComponentName = cms.string('TrackAssociatorByPosition'),
    propagator = cms.string('SteppingHelixPropagatorAlong'),
    # minimum distance from the origin to find a hit from a simulated particle and match it to reconstructed track
    positionMinimumDistance = cms.double(0.0),
    #use a chi2 estimation on the 5x5 local parameters and errors in a plane in the muon system
    #	string method = "chi2"
    #use the distance between state in a plane in the muon system
    method = cms.string('dist'),
    QCut = cms.double(10.0),
    # False is the old behavior, True will use also the muon simhits to do the matching.                                       
    ConsiderAllSimHits = cms.bool(False),
    simHitTpMapTag = cms.InputTag("simHitTPAssocProducer")
)



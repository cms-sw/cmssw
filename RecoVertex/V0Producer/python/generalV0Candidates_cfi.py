import FWCore.ParameterSet.Config as cms

generalV0Candidates = cms.EDProducer("V0Producer",
    # string that tells which TrackCollection to use for vertexing
    trackRecoAlgorithm = cms.untracked.string('generalTracks'),

    # set to true, uses tracks refit by the KVF for V0Candidate kinematics
    useSmoothing = cms.bool(True),

    # set to true, stores tracks refit by KVF in reco::Vertex object
    #  that is contained in the produced reco::V0Candidate 
    storeSmoothedTracksInRecoVertex = cms.bool(True),

    doPostFitCuts = cms.bool(True),
    doTrackQualityCuts = cms.bool(True),

    # The next parameters are cut values

    # Track quality cuts
    #   Normalized track Chi2:
    tkChi2Cut = cms.double(5.0),
    #   Number of valid hits on track:
    tkNhitsCut = cms.int32(6),

    # Vertex cuts
    vtxChi2Cut = cms.double(7.0),
    collinearityCut = cms.double(0.02),
    #  Setting this one to zero; significance cut is sufficient
    rVtxCut = cms.double(0.0),
    vtxSignificanceCut = cms.double(22.0),
    kShortMassCut = cms.double(0.07),
    lambdaMassCut = cms.double(0.05),
    impactParameterCut = cms.double(0.05),
    mPiPiCut = cms.double(0.7),

    # These parameters decide whether or not to reconstruct
    #  specific V0 particles
    selectKshorts = cms.bool(True),
    selectLambdas = cms.bool(True),

    vertexFitter = cms.untracked.string('kvf')

)



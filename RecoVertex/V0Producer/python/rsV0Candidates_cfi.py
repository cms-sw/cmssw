import FWCore.ParameterSet.Config as cms

rsV0Candidates = cms.EDProducer("V0Producer",
    collinearityCut = cms.double(0.02),
    #  Setting this one to zero; significance cut is sufficient
    rVtxCut = cms.double(0.0),
    doPostFitCuts = cms.bool(True),
    # string that tells which TrackCollection to use for vertexing
    trackRecoAlgorithm = cms.untracked.string('rsWithMaterialTracks'),
    doTrackQualityCuts = cms.bool(True),
    # set to 1, uses tracks refit by the KVF for V0Candidate kinematics
    useSmoothing = cms.bool(True),
    selectLambdas = cms.bool(True),
    # These next parameters decide whether or not to reconstruct
    #  specific V0 particles
    selectKshorts = cms.bool(True),
    lambdaMassCut = cms.double(0.25),
    kShortMassCut = cms.double(0.07),
    vtxSignificanceCut = cms.double(22.0),
    # The next parameters are cut values 
    # Track quality cuts
    #   Normalized track Chi2:
    tkChi2Cut = cms.double(5.0),
    # Vertex cuts
    vtxChi2Cut = cms.double(7.0),
    # set to 1, stores tracks refit by KVF in reco::Vertex object
    #  that is contained in the produced reco::V0Candidate
    storeSmoothedTracksInRecoVertex = cms.bool(True),
    vertexFitter = cms.untracked.string('kvf'),
    #   Number of valid hits on track:
    tkNhitsCut = cms.int32(6)
)



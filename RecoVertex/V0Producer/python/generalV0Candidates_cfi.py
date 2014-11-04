import FWCore.ParameterSet.Config as cms

generalV0Candidates = cms.EDProducer("V0Producer",
                                     
   # which TrackCollection to use for vertexing
   trackRecoAlgorithm = cms.InputTag('generalTracks'),

   # which V0s to reconstruct
   doKShorts = cms.bool(True),
   doLambdas = cms.bool(True),

   # which vertex fitting algorithm to use (we recommend using the KalmanVertexFitter)
   vertexFitter = cms.InputTag('KalmanVertexFitter'),

   # if set to True, uses tracks refit by the KVF for V0Candidate kinematics
   # this is automatically set to FALSE if using the AdaptiveVertexFitter (which is not recommended)
   useSmoothing = cms.bool(True),

   # -- cuts on initial track collection
   # Track normalized Chi2 <
   tkChi2Cut = cms.double(10.0),
   # Number of valid hits on track >=
   tkNHitsCut = cms.int32(7),
   # Pt of track >
   tkPtCut = cms.double(0.35),
   # Track impact parameter significance >
   tkIPSigCut = cms.double(2.0),

   # -- cuts on the vertex --
   # Vertex chi2 <
   vtxChi2Cut = cms.double(15.0),
   # Radial vertex significance >
   vtxDecayRSigCut = cms.double(10.0),

   # -- miscellaneous cuts --
   # POCA distance between tracks <
   tkDCACut = cms.double(2.0),
   # check if either track has a hit radially inside the vertex position minus this number times the sigma of the vertex fit
   # note: Set this to -1 to disable this cut, which MUST be done if you want to run V0Producer on the AOD track collection!
   innerHitPosCut = cms.double(4.0),
   # cos(angle) between x and p of V0 candidate >
   v0CosThetaCut = cms.double(0.9998),
   # V0 mass window +- pdg value
   kShortMassCut = cms.double(0.07),
   lambdaMassCut = cms.double(0.05)

)


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

   # The next parameters are cut selections (distances are in cm / energies are in GeV)

   # -- cuts on the input track collection --
   # select tracks using TrackBase::TrackQuality.
   # select ALL tracks by leaving this vstring empty, which is equivalent to using 'loose'
   # trackQualities = cms.vstring('highPurity', 'goodIterative'),
   trackQualities = cms.vstring('loose'),
   # Normalized track Chi2 <
   tkChi2Cut = cms.double(1000.0),
   # Pt of track >
   tkPtCut = cms.double(0.35),
   # Number of valid hits on track >=
   tkNhitsCut = cms.int32(7),
   # Track impact parameter significance >
   tkIPSigCut = cms.double(2.0),

   # -- cuts on the V0 vertex --
   #   Vertex chi2 <
   vtxChi2Cut = cms.double(15.0),
   #  Vertex radius cut >
   vtxRCut = cms.double(0.0),
   #  Radial vertex significance >
   vtxRSigCut = cms.double(10.0),

   # -- miscellaneous cuts after vertexing --
   # m_pipi calculated at PCA of tracks <
   #mPiPiCut = cms.double(0.6),
   # PCA distance between tracks <
   tkDCACut = cms.double(2.0),
   #  V0 mass window +- pdg value
   KShortMassCut = cms.double(0.07),
   LambdaMassCut = cms.double(0.05),
   # check if either track has a hit radially inside the vertex position minus this number times the sigma of the vertex fit
   # note: Set this to -1 to disable this cut, which MUST be done if you want to run V0Producer on the AOD track collection!
   innerHitPosCut = cms.double(4.0),
   # cos(angle) between x and p of V0 candidate >
   V0costhetaCut = cms.double(0.9998)

)


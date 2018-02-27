import FWCore.ParameterSet.Config as cms

L1TVertexAnalyzer = cms.EDAnalyzer('VertexAnalyzer',
  tpInputTag = cms.InputTag("mix", "MergedTrackTruth"),
  stubInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"),
  stubTruthInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
  clusterTruthInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
  l1TracksInputTag = cms.InputTag("TMTrackProducer", "TML1TracksSimpleLR"), # SFLR
  l1VerticesInputTag = cms.InputTag("VertexProducer", "l1vertices"),
  l1VerticesTDRInputTag = cms.InputTag("VertexProducer", "l1verticestdr"),

  #=== Cuts on MC truth particles (i.e., tracking particles) used for tracking efficiency measurements.
  GenCuts = cms.PSet(
     GenMinPt         = cms.double(3.0),
     GenMaxAbsEta     = cms.double(2.4),
     GenMaxVertR      = cms.double(1.0), # Maximum distance of particle production vertex from centre of CMS.
     GenMaxVertZ      = cms.double(30.0),
     GenPdgIds        = cms.vuint32(), # Only particles with these PDG codes used for efficiency measurement.


     # Additional cut on MC truth tracks used for algorithmic tracking efficiency measurements.
     # You should usually set this equal to value of L1TrackDef.MinStubLayers below, unless L1TrackDef.MinPtToReduceLayers
     # is < 10000, in which case, set it equal to (L1TrackDef.MinStubLayers - 1).
     GenMinStubLayers = cms.uint32(4)
  ),


  #=== Rules for deciding when the track finding has found an L1 track candidate
  L1TrackDef = cms.PSet(
     UseLayerID           = cms.bool(True),
     # Reduce this layer ID, so that it takes no more than 8 different values in any eta region (simplifies firmware).
     ReducedLayerID       = cms.bool(True)
  ),

  #=== Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).
  TrackMatchDef = cms.PSet(
     #--- Three different ways to define if a tracking particle matches a reco track candidate. (Usually, set two of them to ultra loose).
     # Min. fraction of matched stubs relative to number of stubs on reco track.
     MinFracMatchStubsOnReco  = cms.double(-99.),
     # Min. fraction of matched stubs relative to number of stubs on tracking particle.
     MinFracMatchStubsOnTP    = cms.double(-99.),
     # Min. number of matched layers.
     MinNumMatchLayers        = cms.uint32(4),
     # Min. number of matched PS layers.
     MinNumMatchPSLayers      = cms.uint32(0),
     # Associate stub to TP only if the TP contributed to both its clusters? (If False, then associate even if only one cluster was made by TP).
     StubMatchStrict          = cms.bool(False)
  ),


  # === Vertex Reconstruction configuration
  VertexReconstruction=cms.PSet(
        # Vertex Reconstruction Id (
        # 0: GapClustering,
        # 1: AgglomerativeClustering,
        # 2: DBSCAN,
        # 3: PVR,
        # 4: AdaptiveVertexReconstruction,
        # 5: HPV,
        # 6: Kmeans,
        # 7: Technical Proposal)
        AlgorithmId = cms.uint32(7),
        # Vertex distance
        VertexDistance = cms.double(.15),
        # Assumed Vertex Resolution
        VertexResolution = cms.double(.10),
        # Distance Type for agglomerative algorithm (0: MaxDistance, 1: MinDistance, 2: MeanDistance, 3: CentralDistance)
        DistanceType  = cms.uint32(0),
        # Keep only PV from hard interaction (highest pT)
        KeepOnlyPV   = cms.bool(True),
        # Minimum number of tracks to accept vertex
        MinTracks   = cms.uint32(2),
        # Compute the z0 position of the vertex with a mean weighted with track momenta
        WeightedMean = cms.bool(True),
        # Chi2 cut for the Adaptive Vertex Reconstruction Algorithm
        AVR_chi2cut = cms.double(5.),
        # TDR algorithm assumed vertex width [cm]
        TDR_VertexWidth = cms.double(.15),
        # Run locally the vertex reconstruction algorithms
        VertexLocal  = cms.bool(False),
        # Kmeans number of iterations
        KmeansIterations = cms.uint32(10),
        # Kmeans number of clusters
        KmeansNumClusters  = cms.uint32(18),
        # DBSCAN pt threshold
        DBSCANPtThreshold = cms.double(2.),
        # DBSCAN min density tracks
        DBSCANMinDensityTracks = cms.uint32(1),
        # Maximum distance to merge two vertices (only if VertexLocal is true)
        VxMergeDistance = cms.double(.15),
        VxInHTsector = cms.bool(True),
        VxMergeByTracks = cms.bool(False),
        VxMinTrackPt   = cms.double(2.)
    ),

  # Debug printout
  Debug  = cms.uint32(0), #(0=none, 1=print tracks/sec, 2=show filled cells in HT array in each sector of each event, 3=print all HT cells each TP is found in, to look for duplicates, 4=print missed tracking particles by r-z filters, 5 = show debug info about duplicate track removal, 6 = show debug info about fitters)
  printResults = cms.bool(False)
)

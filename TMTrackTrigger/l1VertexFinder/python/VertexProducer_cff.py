import FWCore.ParameterSet.Config as cms

VertexProducer = cms.EDProducer('VertexProducer',

  tpInputTag = cms.InputTag("mix", "MergedTrackTruth"),
  stubInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"),
  stubTruthInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
  clusterTruthInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
  l1TracksInputTag = cms.InputTag("TMTrackProducer", "TML1TracksKF4ParamsComb"),

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

  #=== Cuts applied to stubs before arriving in L1 track finding board.

  StubCuts = cms.PSet(
     # Reduce number of bits used by front-end chips to store stub bend info. (CMS is considering this option proposed by Seb. Viret).
     BendResReduced = cms.bool(True),
     # Don't use stubs with eta beyond this cut, since the tracker geometry makes it impossible to reconstruct tracks with them.
     MaxStubEta     = cms.double(2.4),
     # Don't use stubs whose measured Pt from bend info is significantly below HTArraySpec.HoughMinPt, where "significantly" means allowing for resolution in q/Pt derived from stub bend resolution specified below.
     KillLowPtStubs = cms.bool(True),
     # Bend resolution assumed by bend filter in units of strip pitch. Also used when assigning stubs to sectors if EtaPhiSectors.CalcPhiTrkRes=True. And by the bend filter if HTFillingRphi.UseBendFilter=True.
     # Suggested value: 1.19 if BendResReduced = false, or 1.30 if it is true.
     BendResolution = cms.double(1.25),
     # Additional contribution to bend resolution from its encoding into a reduced number of bits.
     # This number is the assumed resolution relative to the naive guess of its value.
     # It is ignored in BendResReduced = False.
     BendResolutionExtra = cms.double(0.0),
     # Order stubs by bend in DTC, such that highest Pt stubs are transmitted first.
     OrderStubsByBend = cms.bool(True)
  ),

  #=== Optional Stub digitization.

  StubDigitize = cms.PSet(
     EnableDigitize  = cms.bool(False),  # Digitize stub coords? If not, use floating point coords.
     FirmwareType    = cms.uint32(1),    # 0 = Old Thomas 2-cbin data format, 1 = new Thomas data format used for daisy chain, 2-98 = reserved for demonstrator daisy chain use, 99 = Systolic array data format.
     #
     #--- Parameters available in MP board.
     #
     PhiSectorBits   = cms.uint32(6),    # Bits used to store phi sector number
     PhiSBits        = cms.uint32(14),   # Bits used to store phiS coord. (13 enough?)
     PhiSRange       = cms.double(0.78539816340),  # Range phiS coord. covers in radians.
     RtBits          = cms.uint32(10),   # Bits used to store Rt coord.
     RtRange         = cms.double(103.0382), # Range Rt coord. covers in units of cm.
     ZBits           = cms.uint32(12),   # Bits used to store z coord.
     ZRange          = cms.double(640.), # Range z coord. covers in units of cm.
     # The following four parameters do not need to be specified if FirmwareType = 1 (i.e., daisy-chain firmware)
     DPhiBits        = cms.untracked.uint32(8),    # Bits used to store Delta(phi) track angle.
     DPhiRange       = cms.untracked.double(1.),   # Range Delta(phi) covers in radians.
     RhoBits         = cms.untracked.uint32(6),    # Bits used to store rho parameter.
     RhoRange        = cms.untracked.double(0.25), # Range rho parameter covers.
     #
     #--- Parameters available in GP board (excluding any in common with MP specified above).
     #
     PhiOBits        = cms.uint32(15),      # Bits used to store PhiO parameter.
     PhiORange       = cms.double(1.5707963268), # Range PhiO parameter covers.
     BendBits        = cms.uint32(6)        # Bits used to store stub bend.
  ),

  #=== Division of Tracker into phi sectors.

  PhiSectors = cms.PSet(
     NumPhiSectors      = cms.uint32(32),  # IRT - 32 or 64 are reasonable choices.
     ChosenRofPhi       = cms.double(58.), # Use phi of track at this radius for assignment of stubs to phi sectors & also for one of the axes of the r-phi HT. If ChosenRofPhi=0, then use track phi0.
     #--- You can set one or both the following parameters to True.
     UseStubPhi         = cms.bool(True),  # Require stub phi to be consistent with track of Pt > HTArraySpec.HoughMinPt that crosses HT phi axis?
     UseStubPhiTrk      = cms.bool(True),  # Require stub phi0 (or phi65 etc.) as estimated from stub bend, to lie within HT phi axis, allowing tolerance(s) specified below?
     AssumedPhiTrkRes   = cms.double(0.5), # Tolerance in stub phi0 (or phi65) assumed to be this fraction of phi sector width. (N.B. If > 0.5, then stubs can be shared by more than 2 phi sectors).
     CalcPhiTrkRes      = cms.bool(True),  # If true, tolerance in stub phi0 (or phi65 etc.) will be reduced below AssumedPhiTrkRes if stub bend resolution specified in StubCuts.BendResolution suggests it is safe to do so.
     HandleStripsPhiSec = cms.bool(False)  # If True, adjust algorithm to allow for uncertainty in stub (r,z) coordinate caused by length of 2S module strips when assigning stubs to phi sectors.
  ),

  #=== Division of Tracker into eta sectors

  EtaSectors = cms.PSet(
     EtaRegions = cms.vdouble(-2.4, -2.0, -1.53, -0.98, -0.37, 0.37, 0.98, 1.53, 2.0, 2.4), # Eta boundaries.
     ChosenRofZ  = cms.double(45.),        # Use z of track at this radius for assignment of tracks to eta sectors & also for one of the axes of the r-z HT. Do not set to zero!
     BeamWindowZ = cms.double(15),         # Half-width of window assumed to contain beam-spot in z.
     HandleStripsEtaSec = cms.bool(False), # If True, adjust algorithm to allow for uncertainty in stub (r,z) coordinate caused by length of 2S module strips when assigning stubs to eta sectors.
     AllowOver2EtaSecs = cms.bool(False)   # If True, the code will not throw an error if a stub is assigned to 3 or more eta sectors.
  ),

  #=== r-phi Hough transform array specifications.

  HTArraySpecRphi = cms.PSet(
     HoughMinPt      = cms.double(3.0), # Min track Pt that Hough Transform must find. Also used by StubCuts.KillLowPtStubs and by EtaPhiSectors.UseStubPhi.
     HoughNbinsPt    = cms.uint32(32),  # HT array dimension in track q/Pt. Ignored if HoughNcellsRphi > 0.
     HoughNbinsPhi   = cms.uint32(32),  # HT array dimension in track phi0 (or phi65 or any other track phi angle. Ignored if HoughNcellsRphi > 0.
     HoughNcellsRphi = cms.int32(-1),   # If > 0, then parameters HoughNbinsPt and HoughNbinsPhi will be calculated from the constraints that their product should equal HoughNcellsRphi and their ratio should make the maximum |gradient|" of stub lines in the HT array equal to 1. If <= 0, then HoughNbinsPt and HoughNbinsPhi will be taken from the values configured above.
     EnableMerge2x2  = cms.bool(False), # Groups of neighbouring 2x2 cells in HT will be treated as if they are a single large cell? N.B. You can only enable this option if your HT array has even numbers of bins in both dimensions.
     MaxPtToMerge2x2 = cms.double(6.),  # but only cells with pt < MaxPtToMerge2x2 will be merged in this way (irrelevant if EnableMerge2x2 = false).
     NumSubSecsEta   = cms.uint32(1),    # Subdivide each sector into this number of subsectors in eta within r-phi HT.
     Shape           = cms.uint32(0)    # cell shape: 0 for square, 1 for diamond, 2 hexagon (with vertical sides), 3 square with alternate rows shifted by 0.5*cell_width.
  ),

  #=== r-z Hough transform array specifications.

  HTArraySpecRz = cms.PSet(
     EnableRzHT    = cms.bool(False),  # If true, find tracks with both r-phi & r-z HTs. If false, use only r-phi HT. If false, other parameters in this sector irrelevant.
     HoughNbinsZ0  = cms.uint32(0),   # HT array dimension in track z0. Ignored if HoughNcellsRz > 0.
     HoughNbinsZ65 = cms.uint32(0),   # HT array dimension in track z65 (or any other r-z track related variable). Ignored if HoughNcellsRz > 0.
     HoughNcellsRz = cms.int32(1024)  # If > 0, then parameters HoughNbinsZ0 and HoughNbinsZ65 will be calculated from the constraints that their product should equal HoughNcellsRz and their ratio should make the maximum |gradient|" of stub lines in the HT array equal to 1. If <= 0, then HoughNbinsZ0 and HoughNbinsRz will be taken from the values configured above.
  ),

  #=== Rules governing how stubs are filled into the r-phi Hough Transform array.

  HTFillingRphi = cms.PSet(
     # If True, adjust algorithm to allow for uncertainty in stub (r,z) coordinate caused by length of 2S module strips when filling r-phi HT with stubs.
     HandleStripsRphiHT   = cms.bool(False),
     # Take all cells in r-phi HT array crossed by line corresponding to each stub (= 0) or take only some to reduce rate at cost
     # of efficiency ( > 0). If this option is > 0, it can be 1 or 2, corresponding to different algorithms for rejecting
     # some of the cells. "1" is an algorithm invented by Ian, whereas "2" corresponds to Thomas' 1st firmware implementation which only handled 1 cell per HT column.
     # Suggest setting KillSomeHTCellsRphi=1 (=0) if HTArraySpec.ChosenRofPhi=0 (>0)
     KillSomeHTCellsRphi  = cms.uint32(0),
     # Use filter in each r-phi HT cell, filling it only with stubs that have consistent bend information?
     # The assumed bend resolution is specified in StubCuts.BendResolution.
     UseBendFilter        = cms.bool(True),
     # Use filter in each HT cell, preventing more than the specified number of stubs being stored in the cell. (Reflecting memory limit of hardware). N.B. Results depend on assumed order of stubs.
     #MaxStubsInCell       = cms.uint32(99999), # Setting this to anything more than 99 disables this option
     MaxStubsInCell      = cms.uint32(16),    # set it equal to value used in hardware.
     # If BusySectorKill = True, and more than BusySectorNumStubs stubs are assigned to tracks by an r-phi HT array, then the excess tracks are killed, with lowest Pt ones killed first. This is because HT hardware has finite readout time.
     BusySectorKill       = cms.bool(False),
     BusySectorNumStubs   = cms.uint32(210),
     # If this is True, then the BusySectorNumStubs cut is applied to +ve and -ve charge track seperately. (Irrelevant if BusySectorKill = False). This option is ignored if BusySectorQoverPtRanges is a non-empty vector.
     BusySectorEachCharge = cms.bool(False),
     # Alternatively, apply the BusySectorNumStubs cut to the subset of tracks appearing in the following m bin (q/Pt) ranges in the HT array. The sum of the entries in the vector should equal the number of m bins in the HT, although the entries will be rescaled if this is not the case. If the vector is empty, this option is disabled. (P.S. If you  set EnableMerge2x2 = True, then the m bin ranges specified here should correspond to the bins before merging).
     BusySectorMbinRanges = cms.vuint32(),
     #BusySectorMbinRanges = cms.vuint32(6,5,5,5,5,6),
     #BusySectorMbinRanges = cms.vuint32(3,3,3,3,2,2,3,3,3,3,2,2),
     # If BusySecMbinOrder is empty, then the groupings specified in BusySectorMbinRanges are applied to the m bins in the order
     # 0,1,2,3,4,5 ... . If it is not empty, then they are grouped in the order specified here.
     BusySectorMbinOrder   = cms.vuint32(),
     #BusySectorMbinOrder  = cms.vuint32(16,21,26, 17,22,27, 18,23,28, 19,24,29, 20,30, 25,31, 15,10,5, 14,9,4, 13,8,3, 12,7,2, 11,1, 6,0),
     # If BusyInputSectorKill = True, and more than BusyInputSectorNumStubs are input to the HT array from the GP, then
     # the excess stubs are killed. This is because HT hardware has finite readin time.
     # Results unreliable as depend on assumed order of stubs.
     BusyInputSectorKill  = cms.bool(False),
     BusyInputSectorNumStubs   = cms.uint32(210),
     # Multiplex the outputs from several HTs onto a single pair of output optical links. N.B. This option is irrelevant unless BusySectorKill = True.
     # (The mux algorithm is hard-wired in class MuxHToutputs, and currently only works if option BusySectorMbinRanges is being used).
     MuxOutputsHT = cms.bool(False),
     # Do this mux using the new design that gives more load balancing to help the KF fit. This algorithm is hard-wired in class MuxHToutputs. To use this option, you must also set MuxOutputsHT=True.
     MuxOutputsHTforKF = cms.bool(False),
     # If this is non-empty, then only the specified eta sectors are enabled, to study them individually.
     EtaRegWhitelist = cms.vuint32()

  ),

  #=== Rules governing how stubs are filled into the r-z Hough Transform array. (Irrelevant if HTArraySpecRz.enableRzHT = false)

  HTFillingRz = cms.PSet(
     # If True, adjust algorithm to allow for uncertainty in stub (r,z) coordinate caused by length of 2S module strips when filling r-z HT with stubs.
     HandleStripsRzHT  = cms.bool(True),
     # Take all cells in r-z HT array crossed by line corresponding to each stub (= 0) or take only some to reduce rate at cost
     # of efficiency ( > 0). If this option is > 0, it can be 1 or 2, corresponding to different algorithms for rejecting
     # some of the cells.
     KillSomeHTCellsRz = cms.uint32(0)
  ),

  #=== Options controlling r-z track filters (or any other track filters run after the Hough transform, as opposed to inside it).

  RZfilterOpts = cms.PSet(
     # Use filter in each r-phi HT cell, filling it only with stubs that have consistent rapidity?
     UseEtaFilter        = cms.bool(False),
     # Use filter in each HT cell using only stubs which have consistent value of ZTrk
     UseZTrkFilter       = cms.bool(False),
     # Filter Stubs in each HT cell using a tracklet seed algorithm
     UseSeedFilter       = cms.bool(False),
     #--- Options for Ztrk filter, (so only relevant if UseZtrkFilter=true).
     # Use z of track at this radius for ZTrkFilter.
     ChosenRofZFilter    = cms.double(23.),
     #--- Options relevant for Seed filter, (so only relevant if useSeedFilter=true).
     # Added resolution for a tracklet-like filter algorithm, beyond that estimated from hit resolution.
     SeedResolution      = cms.double(0.),
     # Store stubs compatible with all possible good seed.
     KeepAllSeed         = cms.bool(False),
     # Maximum number of seed combinations to bother checking per track candidate.
     #MaxSeedCombinations = cms.uint32(999),
     MaxSeedCombinations = cms.uint32(15),
     # Maximum number of seed combinations consistent with (z0,eta) sector constraints to bother checking per track candidate.
     #MaxGoodSeedCombinations = cms.uint32(13),
     MaxGoodSeedCombinations = cms.uint32(10),
     # Maximum number of seeds that a single stub can be included in.
     MaxSeedsPerStub     = cms.uint32(4),
     # Reject tracks whose estimated rapidity from seed filter is inconsistent range of with eta sector. (Kills some duplicate tracks).
     zTrkSectorCheck     = cms.bool(True),
     # Min. number of layers in rz track that must have stubs for track to be declared found.
     MinFilterLayers     = cms.uint32(4),
  ),

  #=== Rules for deciding when the track finding has found an L1 track candidate

  L1TrackDef = cms.PSet(
     # Min. number of layers the track must have stubs in.
     MinStubLayers        = cms.uint32(5),
     # Change min. number of layers cut to (MinStubLayers - 1) for tracks with Pt exceeding this cut.
     # If this is set to > 10000 , this option is disabled.
     MinPtToReduceLayers  = cms.double(99999.),
     # Change min. number of layers cut to (MinStubLayers - 1) for tracks in these rapidity sectors.
     # (Histogram "AlgEffVsEtaSec" will help you identify which sectors to declare).
     EtaSecsReduceLayers  = cms.vuint32(),
     #EtaSecsReduceLayers  = cms.vuint32(0,5,12,17),
     # Define layers using layer ID (true) or by bins in radius of 5 cm width (false).
     UseLayerID           = cms.bool(True),
     # Reduce this layer ID, so that it takes no more than 8 different values in any eta region (simplifies firmware).
     ReducedLayerID       = cms.bool(True)
  ),

  #=== Specification of algorithm to eliminate duplicate tracks.

  DupTrkRemoval = cms.PSet(
    #--- Specify which duplicate removal algorithm(s) to run:  option 0 means disable duplicate track removal, whilst > 0 runs a specific algorithm.
    # Algorithm used for duplicate removal of 2D tracks produced by r-phi HT. Assumed to run before tracks are output from HT board.
    DupTrkAlgRphi = cms.uint32(0),
    #DupTrkAlgRphi = cms.uint32(12),
    # Algorithm used for duplicate removal run on 2D tracks produced by r-z HT corresponding to a single r-phi HT track. Assumed to run before tracks are output from HT board.
    DupTrkAlgRz   = cms.uint32(0),
    # Algorithm run on all 3D tracks within each sector after r-z HT or r-z seed filter.
    #DupTrkAlgRzSeg = cms.uint32(0),
    DupTrkAlgRzSeg = cms.uint32(0),
    # Algorithm run on tracks after the track helix fit has been done.
    DupTrkAlgFit   = cms.uint32(50),
    #DupTrkAlgFit   = cms.uint32(50),
    #--- Options used by individual algorithms.
    # Parameter for OSU duplicate-removal algorithm
    # Specifies minimum number of independent stubs to keep candidate in comparison in Algo 3
    DupTrkMinIndependent = cms.uint32(3),
    # Parameter for "inverse" OSU duplicate-removal algorithm
    # Specifies minimum number of common stubs in same number of layers to keep smaller candidate in comparison in Algos 5-9 + 15-16.
    DupTrkMinCommonHitsLayers = cms.uint32(5),
    # Reduced ChiSq cut in linear RZ fit in Algo 4
    DupTrkChiSqCut = cms.double(99999.),
    # Max diff in qOverPt of 2 tracks in Algo15
    DupMaxQOverPtScan = cms.double(0.025),
    # Max diff in phi0 range of 2 tracks in Algo15
    DupMaxPhi0Scan = cms.double(0.01),
    # Max diff in z0 of 2 tracks in Algo15
    DupMaxZ0Scan = cms.double(0.2),
    # Max diff in tanLambda of 2 tracks in Algo 15
    DupMaxTanLambdaScan = cms.double(0.01)
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

  #=== Track Fitting Algorithm Settings.

  TrackFitSettings = cms.PSet(
     #
     #--- Options applicable to all track fitters ---
     #
     # Track Fitting algortihms to use. You can run several in parallel.
     # (TrackFitLinearAlgo & ChiSquared* are chi2 fits, KF* is a Kalman filter fit, and LinearRegression is simplified 4
     # parameter fit that neglects the hit uncertainties. The number 4 or 5 indicates if 4 or 5 helix parameters are fitted).
     # WARNING: KF5ParamsComb crashes so don't use it!
     # NOTE: globalLinearRegression is usually superior to LinearRegression, except the latter is slightly superior if you are using the seed filter.
     # NOTE: KF4ParamsComb corresponds to the firmware. KF4ParamsCombIV used to be superior, but is currently messed up.
     # NOTE: globalLinearRegression2 is a test version that doesnt corresponds to current LR firmware.
     TrackFitters = cms.vstring(
                                # "TrackFitLinearAlgo4",
                                #"TrackFitLinearAlgo5",
                                #"ChiSquared4ParamsApprox",
                                #"ChiSquared5ParamsApprox",
                                #"ChiSquared4ParamsTrackletStyle",
                                 "KF4ParamsComb",
                                #"KF5ParamsComb",
                                #"KF4ParamsCombIV",
                                #"LinearRegression",
                                # "globalLinearRegression"
                                #"globalLinearRegression2"
                                #"SimpleLR"
                                ),
     # Cut on chi2/dof of fitted track when making histograms.
     Chi2OverNdfCut = cms.double(999999.),
     # Print detailed summary of track fit performance at end of job (as opposed to a brief one).
     DetailedFitOutput = cms.bool(False),
     #
     #--- Options for chi2 track fitter ---
     #
     # Number of fitting iterations to undertake. (15 is not realistic in hardware, but is necessary to kill bad hits)
     NumTrackFitIterations = cms.uint32(15),
     # Optionally kill hit with biggest residuals in track fit (occurs after the first fit, so three iterations would have two killings). (Only used by chi2 track fit).
     KillTrackFitWorstHit  = cms.bool(True),
     # Cuts in standard deviations used to kill hits with big residuals during fit. If the residual exceeds the "General" cut, the hit is killed providing it leaves the track with enough hits to survive. If the residual exceeds the "Killing" cut, the hit is killed even if that kills the track.
     GeneralResidualCut = cms.double(3.0),
     KillingResidualCut = cms.double(20.0),
     #
     #--- Additional options for Linear Regression track fitter ---
     #
     # Maximum allowed number of iterations of LR fitter.
     MaxIterationsLR                 = cms.uint32( 8 ),
     # Internal histograms are filled if it is True
     LRFillInternalHists             = cms.bool(False),
     # If False: residual of a stub is the max of its r-phi & r-z residuals.
     # If True: the residual is the mean of these residuals.
     CombineResiduals                = cms.bool( True ),
     # Correct stub phi coordinate for higher orders in circle expansion, so that a trajectory is straight in r-phi.
     LineariseStubPosition           = cms.bool( True ),
     # Checks if the fitted track is consistent with the sector, if not it will be not accepted.
     CheckSectorConsistency          = cms.bool( False ),
     # Checks if the fitted track r phi parameter  are consistent with the HT candidate parameter within in range of +- 2 cells.
     CheckHTCellConsistency          = cms.bool( False ),
     # Tracks must have stubs in at least this number of PS layers.
     MinPSLayers                     = cms.uint32( 2 ),
     # Digitization
     DigitizeLR      = cms.bool( False ),
     PhiPrecision    = cms.double( 0.009 / 108. ),
     RPrecision      = cms.double( 0.14 ),
     ZPrecision      = cms.double( 0.28 ),
     ZSlopeWidth     = cms.uint32( 11 ),
     ZInterceptWidth = cms.uint32( 11 ),
     #
     #--- Options for Kalman filter track fitters ---
     #
     # larger number has more debugging outputs.
     KalmanDebugLevel                = cms.uint32(0),
     # Internal histograms are filled if it is True
     KalmanFillInternalHists         = cms.bool(False),
     # Multiple scattering factor.  (If you set this to 1, you will get better results with KF4ParamsCombIV).
     KalmanMultipleScatteringFactor  = cms.double(0.0),
     # A stub which is inconsistent with the state is not processed for the state. Cut value on chisquare from the forecast is set. Unused!
     KalmanValidationGateCutValue    = cms.double(999),
     # Best candidate is selected from the candidates with the more number of stubs if this is Ture. Chi2 only selection is better to remove combinatorial hits.
     KalmanSelectMostNumStubState    = cms.bool(False),
     # After this number of stubs in the state, the rest of the stubs are not processed and go to the next state.
     KalmanMaxNumNextStubs           = cms.uint32(999),
     # Allowed number of virtual stubs. (CAUTION: kalmanState::nVirtualStubs() return the # of virtual stubs including the seed. The number on this option does not include the seed.)
     KalmanMaxNumVirtualStubs        = cms.uint32(2),
     # The number of states for each virtual stub list is restricted to this number.  The lists are truncated to this number of states. Unused!
     KalmanMaxNumStatesCutValue      = cms.uint32(999),
     # The state is removed if the reduced chisquare is more than this number. Unused.
     KalmanStateReducedChi2CutValue  = cms.double(999),
     # Following two options implement configurable chi2 cuts for KF4ParamsComb only.
     KalmanBarrelChi2Dof = cms.vdouble(999.,999.,20.,30.,40.,45.),
     KalmanEndcapChi2Dof = cms.vdouble(999.,25.,25.,25.,25.,25.,25.,30.,30.,30.,30.,30.,30.,30.,30.),
        #
     #--- Options for Simple LR track fitters ---
     #
     # Digitize Simple Linear Regression variables and calculation
     DigitizeSLR         = cms.bool(False),
     # Number of bits to be used in hardware to compute the division needed to calculate the helix parameters
     DividerBitsHelix    = cms.uint32(16),
     # Number of bits to reduce the helix parameter calculation weight
     ShiftingBits        = cms.uint32(12),
     # Number of bits to reduce the qOverPt parameter calculation weight
     ShiftingBitsPt      = cms.uint32(5),
     # Number of bits to reduce the tanLambda parameter calculation weight
     ShiftingBitsLambda  = cms.uint32(0),
     # Number of bits to reduce the z0 parameter calculation weight
     ShiftingBitsZ0      = cms.uint32(9),
     # ChiSquare Cut
     SLR_chi2cut         = cms.double(9999.),
     # Cut on Rphi Residuals
     ResidualCut         = cms.double(999.),
     # Minimum number of stubs in output fitted track
     SLR_minstubs        = cms.uint32(4),
  ),

  #=== Treatment of dead modules.

  DeadModuleOpts = cms.PSet(
     # In (eta,phi) sectors containing dead modules, reduce the min. number of layers cut on tracks to (MinStubLayers - 1)?
     # The sectors affected are hard-wired in DeadModuleDB::defineDeadTrackerRegions().
     DeadReduceLayers  = cms.bool( False ),
     # Emulate dead modules by killing fraction of stubs given by DeadSimulateFrac in certain layers & angular regions of
     # the tracker that are hard-wired in DeadModuleDB::defineDeadSectors(). Disable by setting <= 0. Fully enable by setting to 1.
     DeadSimulateFrac = cms.double(-999.)
  ),
# === Vertex Reconstruction configuration
  VertexReconstruction=cms.PSet(
        # Vertex Reconstruction Id (0: GapClustering, 1: SimpleMergeClustering, 2: DBSCAN, 3: PVR, 4: AdaptiveVertexReconstruction, 5: HPV)
        AlgorithmId = cms.uint32(2),
        # Minimum distance of tracks to belong to same recovertex [cm]
        VertexResolution = cms.double(.15),
        # Minimum number of tracks to accept vertex
        MinTracks   = cms.uint32(2),
        # Chi2 cut for the Adaptive Vertex Reconstruction Algorithm
        AVR_chi2cut = cms.double(5.),
        # TDR algorithm assumed vertex width [cm]
        TDR_VertexWidth = cms.double(.15),
        # Maximum distance between reconstructed and generated vertex, in order to consider the vertex as correctly reconstructed
        RecoVertexDistance = cms.double(.15),
        # Minimum number of high pT (pT > 10 GeV) tracks that the vertex has to contain to be a good hard interaction vertex candidate
        MinHighPtTracks = cms.uint32(1),
    ),
  #=== Fitted track digitisation.

  TrackDigi=cms.PSet(
     #======= SimpleLR digi parameters ========
     SLR_skipTrackDigi = cms.bool( False ), # Optionally skip track digitisation if done internally inside fitting code.
     SLR_oneOver2rBits = cms.uint32(11),
     SLR_oneOver2rRange = cms.double(0.0076223979397932),
     SLR_phi0Bits = cms.uint32(14),
     SLR_phi0Range = cms.double(0.78539816340), # phi0 is actually only digitised relative to centre of sector.
     #SLR_z0Bits = cms.uint32(11),
     #SLR_z0Range  = cms.double(40),
     SLR_z0Bits = cms.uint32(12),
     SLR_z0Range  = cms.double(80),
     SLR_tanlambdaBits = cms.uint32(17),
     SLR_tanlambdaRange = cms.double(49.6903090310196),
     SLR_chisquaredBits = cms.uint32(10),
     SLR_chisquaredRange = cms.double(256.),

     # ====== Kalman Filter Digi paramters ========
     KF_skipTrackDigi = cms.bool( False ), # Optionally skip track digitisation if done internally inside fitting code.
     KF_oneOver2rBits = cms.uint32(18),
     KF_oneOver2rRange = cms.double(0.06097882386778998),
     KF_phi0Bits = cms.uint32(18),
     KF_phi0Range = cms.double(0.7855158),  # phi0 is actually only digitised relative to centre of sector.
     KF_z0Bits = cms.uint32(18),
     KF_z0Range  = cms.double(51.5194204),
     KF_tanlambdaBits = cms.uint32(18),
     KF_tanlambdaRange = cms.double(32.),
     KF_chisquaredBits = cms.uint32(17),
     KF_chisquaredRange = cms.double(1024.),
     #====== Other track fitter Digi params.
     # Currently equal to those for KF, although you can skip track digitisation for them with following.
     Other_skipTrackDigi = cms.bool( True ),
  ),

  # Debug printout
  Debug  = cms.uint32(0), #(0=none, 1=print tracks/sec, 2=show filled cells in HT array in each sector of each event, 3=print all HT cells each TP is found in, to look for duplicates, 4=print missed tracking particles by r-z filters, 5 = show debug info about duplicate track removal, 6 = show debug info about fitters)
  # Specify sector for which debug histos for hexagonal HT will be made.
  iPhiPlot = cms.uint32(0),
  iEtaPlot = cms.uint32(9)
)

#include <TMTrackTrigger/VertexFinder/interface/Settings.h>
#include "FWCore/Utilities/interface/Exception.h"
 

namespace vertexFinder {

///=== Get configuration parameters

Settings::Settings(const edm::ParameterSet& iConfig) :
 
  // See either Analyze_Defaults_cfi.py or Settings.h for description of these parameters.
 
  // Parameter sets for differents types of configuration parameter.
  genCuts_                ( iConfig.getParameter< edm::ParameterSet >         ( "GenCuts"                ) ),
  stubCuts_               ( iConfig.getParameter< edm::ParameterSet >         ( "StubCuts"               ) ),
  stubDigitize_           ( iConfig.getParameter< edm::ParameterSet >         ( "StubDigitize"           ) ),
  phiSectors_             ( iConfig.getParameter< edm::ParameterSet >         ( "PhiSectors"             ) ),
  etaSectors_             ( iConfig.getParameter< edm::ParameterSet >         ( "EtaSectors"             ) ),
  htArraySpecRphi_        ( iConfig.getParameter< edm::ParameterSet >         ( "HTArraySpecRphi"        ) ),
  htArraySpecRz_          ( iConfig.getParameter< edm::ParameterSet >         ( "HTArraySpecRz"          ) ),
  htFillingRphi_          ( iConfig.getParameter< edm::ParameterSet >         ( "HTFillingRphi"          ) ),
  htFillingRz_            ( iConfig.getParameter< edm::ParameterSet >         ( "HTFillingRz"            ) ),
  rzFilterOpts_           ( iConfig.getParameter< edm::ParameterSet >         ( "RZfilterOpts"           ) ),
  l1TrackDef_             ( iConfig.getParameter< edm::ParameterSet >         ( "L1TrackDef"             ) ),
  dupTrkRemoval_          ( iConfig.getParameter< edm::ParameterSet >         ( "DupTrkRemoval"          ) ),
  trackMatchDef_          ( iConfig.getParameter< edm::ParameterSet >         ( "TrackMatchDef"          ) ),
  trackFitSettings_       ( iConfig.getParameter< edm::ParameterSet >         ( "TrackFitSettings"       ) ),
  deadModuleOpts_         ( iConfig.getParameter< edm::ParameterSet >         ( "DeadModuleOpts"         ) ),
  trackDigi_              ( iConfig.getParameter< edm::ParameterSet >         ( "TrackDigi"              ) ),
  vertex_                 ( iConfig.getParameter< edm::ParameterSet >         ( "VertexReconstruction" )), 

  //=== Cuts on MC truth tracks used for tracking efficiency measurements.
  genMinPt_               ( genCuts_.getParameter<double>                     ( "GenMinPt"               ) ),
  genMaxAbsEta_           ( genCuts_.getParameter<double>                     ( "GenMaxAbsEta"           ) ),
  genMaxVertR_            ( genCuts_.getParameter<double>                     ( "GenMaxVertR"            ) ),
  genMaxVertZ_            ( genCuts_.getParameter<double>                     ( "GenMaxVertZ"            ) ),
  genMinStubLayers_       ( genCuts_.getParameter<unsigned int>               ( "GenMinStubLayers"       ) ),

  //=== Cuts applied to stubs before arriving in L1 track finding board.
 
  bendResReduced_         ( stubCuts_.getParameter<bool>                      ( "BendResReduced"         ) ),
  maxStubEta_             ( stubCuts_.getParameter<double>                    ( "MaxStubEta"             ) ),
  killLowPtStubs_         ( stubCuts_.getParameter<bool>                      ( "KillLowPtStubs"         ) ),
  bendResolution_         ( stubCuts_.getParameter<double>                    ( "BendResolution"         ) ),
  bendResolutionExtra_    ( stubCuts_.getParameter<double>                    ( "BendResolutionExtra"    ) ),
  orderStubsByBend_       ( stubCuts_.getParameter<bool>                      ( "OrderStubsByBend"       ) ),

  //=== Optional stub digitization.
  enableDigitize_         ( stubDigitize_.getParameter<bool>                  ( "EnableDigitize"         ) ),
  firmwareType_           ( stubDigitize_.getParameter<unsigned int>          ( "FirmwareType"           ) ),
  //--- Parameters available in MP board.
  phiSectorBits_          ( stubDigitize_.getParameter<unsigned int>          ( "PhiSectorBits"          ) ),
  phiSBits_               ( stubDigitize_.getParameter<unsigned int>          ( "PhiSBits"               ) ),
  phiSRange_              ( stubDigitize_.getParameter<double>                ( "PhiSRange"              ) ),
  rtBits_                 ( stubDigitize_.getParameter<unsigned int>          ( "RtBits"                 ) ),
  rtRange_                ( stubDigitize_.getParameter<double>                ( "RtRange"                ) ),
  zBits_                  ( stubDigitize_.getParameter<unsigned int>          ( "ZBits"                  ) ),
  zRange_                 ( stubDigitize_.getParameter<double>                ( "ZRange"                 ) ),
  // These parameters are not needed with daisy chain firmware, so make them untracked, so they don't
  // need to be specified.
  dPhiBits_               ( stubDigitize_.getUntrackedParameter<unsigned int> ( "DPhiBits"          ,16  ) ),
  dPhiRange_              ( stubDigitize_.getUntrackedParameter<double>       ( "DPhiRange"         ,10. ) ),
  rhoBits_                ( stubDigitize_.getUntrackedParameter<unsigned int> ( "RhoBits"           ,16  ) ),
  rhoRange_               ( stubDigitize_.getUntrackedParameter<double>       ( "RhoRange"          ,1.  ) ),
  //--- Parameters available in GP board (excluding any in common with MP specified above).
  phiOBits_               ( stubDigitize_.getParameter<unsigned int>          ( "PhiOBits"               ) ),
  phiORange_              ( stubDigitize_.getParameter<double>                ( "PhiORange"              ) ),
  bendBits_               ( stubDigitize_.getParameter<unsigned int>          ( "BendBits"               ) ),
 
  //=== Division of Tracker into phi sectors.
  numPhiSectors_          ( phiSectors_.getParameter<unsigned int>            ( "NumPhiSectors"          ) ),
  chosenRofPhi_           ( phiSectors_.getParameter<double>                  ( "ChosenRofPhi"           ) ),
  useStubPhi_             ( phiSectors_.getParameter<bool>                    ( "UseStubPhi"             ) ),
  useStubPhiTrk_          ( phiSectors_.getParameter<bool>                    ( "UseStubPhiTrk"          ) ),
  assumedPhiTrkRes_       ( phiSectors_.getParameter<double>                  ( "AssumedPhiTrkRes"       ) ),
  calcPhiTrkRes_          ( phiSectors_.getParameter<bool>                    ( "CalcPhiTrkRes"          ) ),
  handleStripsPhiSec_     ( phiSectors_.getParameter<bool>                    ( "HandleStripsPhiSec"     ) ),
 
  //=== Division of Tracker into eta sectors.
  etaRegions_             ( etaSectors_.getParameter<std::vector<double> >    ( "EtaRegions"             ) ),
  chosenRofZ_             ( etaSectors_.getParameter<double>                  ( "ChosenRofZ"             ) ),
  beamWindowZ_            ( etaSectors_.getParameter<double>                  ( "BeamWindowZ"            ) ),  
  handleStripsEtaSec_     ( etaSectors_.getParameter<bool>                    ( "HandleStripsEtaSec"     ) ),
  allowOver2EtaSecs_      ( etaSectors_.getParameter<bool>                    ( "AllowOver2EtaSecs"      ) ),
                               
  //=== r-phi Hough transform array specifications.
  houghMinPt_             ( htArraySpecRphi_.getParameter<double>             ( "HoughMinPt"             ) ),
  houghNbinsPt_           ( htArraySpecRphi_.getParameter<unsigned int>       ( "HoughNbinsPt"           ) ),
  houghNbinsPhi_          ( htArraySpecRphi_.getParameter<unsigned int>       ( "HoughNbinsPhi"          ) ),
  houghNcellsRphi_        ( htArraySpecRphi_.getParameter<int>                ( "HoughNcellsRphi"        ) ),
  enableMerge2x2_         ( htArraySpecRphi_.getParameter<bool>               ( "EnableMerge2x2"         ) ),
  maxPtToMerge2x2_        ( htArraySpecRphi_.getParameter<double>             ( "MaxPtToMerge2x2"        ) ),
  numSubSecsEta_          ( htArraySpecRphi_.getParameter<unsigned int>       ( "NumSubSecsEta"          ) ),
  shape_                  ( htArraySpecRphi_.getParameter<unsigned int>       ( "Shape"                  ) ),

  //=== r-z Hough transform array specifications.
  enableRzHT_             ( htArraySpecRz_.getParameter<bool>                 ( "EnableRzHT"             ) ),
  houghNbinsZ0_           ( htArraySpecRz_.getParameter<unsigned int>         ( "HoughNbinsZ0"           ) ),
  houghNbinsZ65_          ( htArraySpecRz_.getParameter<unsigned int>         ( "HoughNbinsZ65"          ) ),
  houghNcellsRz_          ( htArraySpecRz_.getParameter<int>                  ( "HoughNcellsRz"          ) ),
                               
  //=== Rules governing how stubs are filled into the r-phi Hough Transform array.
  handleStripsRphiHT_     ( htFillingRphi_.getParameter<bool>                 ( "HandleStripsRphiHT"     ) ),
  killSomeHTCellsRphi_    ( htFillingRphi_.getParameter<unsigned int>         ( "KillSomeHTCellsRphi"    ) ),
  useBendFilter_          ( htFillingRphi_.getParameter<bool>                 ( "UseBendFilter"          ) ),
  maxStubsInCell_         ( htFillingRphi_.getParameter<unsigned int>         ( "MaxStubsInCell"         ) ),
  busySectorKill_         ( htFillingRphi_.getParameter<bool>                 ( "BusySectorKill"         ) ),
  busySectorNumStubs_     ( htFillingRphi_.getParameter<unsigned int>         ( "BusySectorNumStubs"     ) ),
  busySectorEachCharge_   ( htFillingRphi_.getParameter<bool>                 ( "BusySectorEachCharge"   ) ),
  busySectorMbinRanges_   ( htFillingRphi_.getParameter<std::vector<unsigned int>> ( "BusySectorMbinRanges" ) ),
  busySectorMbinOrder_    ( htFillingRphi_.getParameter<std::vector<unsigned int>> ( "BusySectorMbinOrder" ) ),
  busyInputSectorKill_    ( htFillingRphi_.getParameter<bool>                 ( "BusyInputSectorKill"    ) ),
  busyInputSectorNumStubs_( htFillingRphi_.getParameter<unsigned int>         ( "BusyInputSectorNumStubs") ),
  muxOutputsHT_           ( htFillingRphi_.getParameter<bool>                 ( "MuxOutputsHT"           ) ),
  muxOutputsHTforKF_      ( htFillingRphi_.getParameter<bool>                 ( "MuxOutputsHTforKF"      ) ),
  etaRegWhitelist_        ( htFillingRphi_.getParameter<std::vector<unsigned int> > ( "EtaRegWhitelist"  ) ),

  //=== Rules governing how stubs are filled into the r-z Hough Transform array. (Irrelevant if enableRzHT = false.)
  handleStripsRzHT_       ( htFillingRz_.getParameter<bool>                   ( "HandleStripsRzHT"       ) ),
  killSomeHTCellsRz_      ( htFillingRz_.getParameter<unsigned int>           ( "KillSomeHTCellsRz"      ) ),
 
  //=== Options controlling r-z track filters (or any other track filters run after the Hough transform, as opposed to inside it).
  useEtaFilter_           ( rzFilterOpts_.getParameter<bool>                  ( "UseEtaFilter"           ) ),
  useZTrkFilter_          ( rzFilterOpts_.getParameter<bool>                  ( "UseZTrkFilter"          ) ),
  useSeedFilter_          ( rzFilterOpts_.getParameter<bool>                  ( "UseSeedFilter"          ) ),
  chosenRofZFilter_       ( rzFilterOpts_.getParameter<double>                ( "ChosenRofZFilter"       ) ),
  seedResolution_         ( rzFilterOpts_.getParameter<double>                ( "SeedResolution"         ) ),
  keepAllSeed_            ( rzFilterOpts_.getParameter<bool>                  ( "KeepAllSeed"            ) ),
  maxSeedCombinations_    ( rzFilterOpts_.getParameter<unsigned int>          ( "MaxSeedCombinations"    ) ),
  maxGoodSeedCombinations_( rzFilterOpts_.getParameter<unsigned int>          ( "MaxGoodSeedCombinations") ),
  maxSeedsPerStub_        ( rzFilterOpts_.getParameter<unsigned int>          ( "MaxSeedsPerStub"        ) ),
  zTrkSectorCheck_        ( rzFilterOpts_.getParameter<bool>                  ( "zTrkSectorCheck"        ) ),
  minFilterLayers_        ( rzFilterOpts_.getParameter<unsigned int>          ( "MinFilterLayers"        ) ),
 
  //=== Rules for deciding when the track finding has found an L1 track candidate
 
  minStubLayers_          ( l1TrackDef_.getParameter<unsigned int>            ( "MinStubLayers"          ) ),
  minPtToReduceLayers_    ( l1TrackDef_.getParameter<double>                  ( "MinPtToReduceLayers"    ) ),
  etaSecsReduceLayers_    ( l1TrackDef_.getParameter<std::vector<unsigned int>> ( "EtaSecsReduceLayers"  ) ),
  useLayerID_             ( l1TrackDef_.getParameter<bool>                    ( "UseLayerID"             ) ),
  reduceLayerID_          ( l1TrackDef_.getParameter<bool>                    ( "ReducedLayerID"         ) ),
 
  //=== Specification of algorithm to eliminate duplicate tracks.
  dupTrkAlgRphi_          ( dupTrkRemoval_.getParameter<unsigned int>         ( "DupTrkAlgRphi"          ) ),
  dupTrkAlgRz_            ( dupTrkRemoval_.getParameter<unsigned int>         ( "DupTrkAlgRz"            ) ),
  dupTrkAlgRzSeg_         ( dupTrkRemoval_.getParameter<unsigned int>         ( "DupTrkAlgRzSeg"         ) ),
  dupTrkAlgFit_           ( dupTrkRemoval_.getParameter<unsigned int>         ( "DupTrkAlgFit"           ) ),
  dupTrkMinIndependent_   ( dupTrkRemoval_.getParameter<unsigned int>         ( "DupTrkMinIndependent"   ) ),
  dupTrkMinCommonHitsLayers_   ( dupTrkRemoval_.getParameter<unsigned int>    ( "DupTrkMinCommonHitsLayers"   ) ),
  dupTrkChiSqCut_         ( dupTrkRemoval_.getParameter<double>               ( "DupTrkChiSqCut"         ) ),
  dupMaxQOverPtScan_      ( dupTrkRemoval_.getParameter<double>               ( "DupMaxQOverPtScan"      ) ),
  dupMaxPhi0Scan_         ( dupTrkRemoval_.getParameter<double>               ( "DupMaxPhi0Scan"         ) ),
  dupMaxZ0Scan_           ( dupTrkRemoval_.getParameter<double>               ( "DupMaxZ0Scan"           ) ),
  dupMaxTanLambdaScan_    ( dupTrkRemoval_.getParameter<double>               ( "DupMaxTanLambdaScan"    ) ),

  //=== Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).
 
  minFracMatchStubsOnReco_( trackMatchDef_.getParameter<double>               ( "MinFracMatchStubsOnReco") ),
  minFracMatchStubsOnTP_  ( trackMatchDef_.getParameter<double>               ( "MinFracMatchStubsOnTP"  ) ),
  minNumMatchLayers_      ( trackMatchDef_.getParameter<unsigned int>         ( "MinNumMatchLayers"      ) ),
  minNumMatchPSLayers_    ( trackMatchDef_.getParameter<unsigned int>         ( "MinNumMatchPSLayers"    ) ),
  stubMatchStrict_        ( trackMatchDef_.getParameter<bool>                 ( "StubMatchStrict"        ) ),

    //=== Track Fitting Settings
 
  trackFitters_   ( trackFitSettings_.getParameter<std::vector<std::string>>  ( "TrackFitters"           ) ),
  chi2OverNdfCut_         ( trackFitSettings_.getParameter<double>            ( "Chi2OverNdfCut"         ) ),
  detailedFitOutput_      ( trackFitSettings_.getParameter < bool >           ( "DetailedFitOutput"      ) ),
  numTrackFitIterations_  ( trackFitSettings_.getParameter<unsigned int>      ( "NumTrackFitIterations"  ) ),
  killTrackFitWorstHit_   ( trackFitSettings_.getParameter <bool>             ( "KillTrackFitWorstHit"   ) ),
  generalResidualCut_     ( trackFitSettings_.getParameter<double>            ( "GeneralResidualCut"     ) ),
  killingResidualCut_     ( trackFitSettings_.getParameter<double>            ( "KillingResidualCut"     ) ),
  maxIterationsLR_        ( trackFitSettings_.getParameter<unsigned int>      ( "MaxIterationsLR"        ) ),
  LRFillInternalHists_    ( trackFitSettings_.getParameter<bool>              ( "LRFillInternalHists"    ) ),
  combineResiduals_       ( trackFitSettings_.getParameter< bool >            ( "CombineResiduals"       ) ),
  lineariseStubPosition_  ( trackFitSettings_.getParameter< bool >            ( "LineariseStubPosition"  ) ),
  checkSectorConsistency_ ( trackFitSettings_.getParameter< bool >            ( "CheckSectorConsistency" ) ),
  checkHTCellConsistency_ ( trackFitSettings_.getParameter< bool >            ( "CheckHTCellConsistency" ) ),
  minPSLayers_            ( trackFitSettings_.getParameter< unsigned int >    ( "MinPSLayers"            ) ),
  digitizeLR_             ( trackFitSettings_.getParameter< bool >            ( "DigitizeLR"             ) ),
  PhiPrecision_           ( trackFitSettings_.getParameter< double >          ( "PhiPrecision"           ) ),
  RPrecision_             ( trackFitSettings_.getParameter< double >          ( "RPrecision"             ) ),
  ZPrecision_             ( trackFitSettings_.getParameter< double >          ( "ZPrecision"             ) ),
  ZSlopeWidth_            ( trackFitSettings_.getParameter< unsigned int >    ( "ZSlopeWidth"            ) ),
  ZInterceptWidth_        ( trackFitSettings_.getParameter< unsigned int >    ( "ZInterceptWidth"        ) ),
  kalmanDebugLevel_              ( trackFitSettings_.getParameter<unsigned>   ( "KalmanDebugLevel"               ) ),
  kalmanFillInternalHists_       ( trackFitSettings_.getParameter<bool>       ( "KalmanFillInternalHists"        ) ),
  kalmanMultiScattFactor_        ( trackFitSettings_.getParameter<double>     ( "KalmanMultipleScatteringFactor" ) ),
  kalmanValidationGateCutValue_  ( trackFitSettings_.getParameter<double>     ( "KalmanValidationGateCutValue"   ) ),
  kalmanSelectMostNumStubState_  ( trackFitSettings_.getParameter<bool>       ( "KalmanSelectMostNumStubState"   ) ),
  kalmanMaxNumNextStubs_         ( trackFitSettings_.getParameter<unsigned>   ( "KalmanMaxNumNextStubs"          ) ),
  kalmanMaxNumVirtualStubs_      ( trackFitSettings_.getParameter<unsigned>   ( "KalmanMaxNumVirtualStubs"       ) ),
  kalmanMaxNumStatesCutValue_    ( trackFitSettings_.getParameter<unsigned>   ( "KalmanMaxNumStatesCutValue"     ) ),
  kalmanStateReducedChi2CutValue_( trackFitSettings_.getParameter<double>     ( "KalmanStateReducedChi2CutValue" ) ),
  kalmanBarrelChi2Dof_           ( trackFitSettings_.getParameter<std::vector<double> >  ( "KalmanBarrelChi2Dof" ) ),
  kalmanEndcapChi2Dof_           ( trackFitSettings_.getParameter<std::vector<double> >  ( "KalmanEndcapChi2Dof" ) ),

  digitizeSLR_ (trackFitSettings_.getParameter<bool>                    ( "DigitizeSLR"      ) ),
  dividerBitsHelix_(trackFitSettings_.getParameter<unsigned int>                    ( "DividerBitsHelix"      ) ),
  ShiftingBits_(trackFitSettings_.getParameter<unsigned int>                    ( "ShiftingBits"      ) ), 
  ShiftingBitsPt_(trackFitSettings_.getParameter<unsigned int>                    ( "ShiftingBitsPt"      ) ),    
  ShiftingBitsLambda_(trackFitSettings_.getParameter<unsigned int>                    ( "ShiftingBitsLambda"      ) ),
  ShiftingBitsZ0_(trackFitSettings_.getParameter<unsigned int>            ("ShiftingBitsZ0") ),
  slr_chi2cut_(trackFitSettings_.getParameter<double>                    ( "SLR_chi2cut"      ) ),   
  residualCut_(trackFitSettings_.getParameter<double>                     ("ResidualCut") ) , 
  slr_minstubs_(trackFitSettings_.getParameter<unsigned int>            ("SLR_minstubs") ),
  //=== Treatment of dead modules.

  deadReduceLayers_       (deadModuleOpts_.getParameter<bool>                 ( "DeadReduceLayers"       ) ),
  deadSimulateFrac_       (deadModuleOpts_.getParameter<double>               ( "DeadSimulateFrac"       ) ),

  //=== Track digitisation configuration for various track fitters
  slr_skipTrackDigi_      (trackDigi_.getParameter<bool>                      ( 
    "SLR_skipTrackDigi"      ) ),
  slr_oneOver2rBits_      (trackDigi_.getParameter<unsigned int>              ( 
    "SLR_oneOver2rBits"      ) ),
  slr_oneOver2rRange_     (trackDigi_.getParameter<double>                    ( 
    "SLR_oneOver2rRange"     ) ),
  slr_phi0Bits_           (trackDigi_.getParameter<unsigned int>              ( 
    "SLR_phi0Bits"           ) ),
  slr_phi0Range_          (trackDigi_.getParameter<double>                    ( 
    "SLR_phi0Range"          ) ),
  slr_z0Bits_             (trackDigi_.getParameter<unsigned int>              ( 
    "SLR_z0Bits"             ) ),
  slr_z0Range_            (trackDigi_.getParameter<double>                    ( 
    "SLR_z0Range"            ) ),
  slr_tanlambdaBits_      (trackDigi_.getParameter<unsigned int>              ( 
    "SLR_tanlambdaBits"      ) ),
  slr_tanlambdaRange_     (trackDigi_.getParameter<double>                    ( 
    "SLR_tanlambdaRange"     ) ),
  slr_chisquaredBits_     (trackDigi_.getParameter<unsigned int>              ( 
    "SLR_chisquaredBits"     ) ),
  slr_chisquaredRange_    (trackDigi_.getParameter<double>                    ( 
    "SLR_chisquaredRange"    ) ),
//
  kf_skipTrackDigi_       (trackDigi_.getParameter<bool>                      ( 
    "KF_skipTrackDigi"       ) ),
  kf_oneOver2rBits_       (trackDigi_.getParameter<unsigned int>              ( 
    "KF_oneOver2rBits"       ) ),
  kf_oneOver2rRange_      (trackDigi_.getParameter<double>                    ( 
    "KF_oneOver2rRange"      ) ),
  kf_phi0Bits_            (trackDigi_.getParameter<unsigned int>              ( 
    "KF_phi0Bits"            ) ),
  kf_phi0Range_           (trackDigi_.getParameter<double>                    ( 
    "KF_phi0Range"           ) ),
  kf_z0Bits_              (trackDigi_.getParameter<unsigned int>              ( 
    "KF_z0Bits"              ) ),
  kf_z0Range_             (trackDigi_.getParameter<double>                    ( 
    "KF_z0Range"             ) ),
  kf_tanlambdaBits_       (trackDigi_.getParameter<unsigned int>              ( 
    "KF_tanlambdaBits"       ) ),
  kf_tanlambdaRange_      (trackDigi_.getParameter<double>                    ( 
    "KF_tanlambdaRange"      ) ),
  kf_chisquaredBits_      (trackDigi_.getParameter<unsigned int>              ( 
    "KF_chisquaredBits"      ) ),
  kf_chisquaredRange_     (trackDigi_.getParameter<double>                    ( 
    "KF_chisquaredRange"     ) ),
//
  other_skipTrackDigi_    (trackDigi_.getParameter<bool>                      ( 
    "Other_skipTrackDigi"    ) ),
  
  //=== Vertex Reconstruction configuration
  vx_algoId_              (vertex_.getParameter<unsigned int>                 ( "AlgorithmId")),
  vx_distance_            (vertex_.getParameter<double>                 ( "VertexResolution")),
  vx_minTracks_           (vertex_.getParameter<unsigned int>                 ( "MinTracks")),
  vx_chi2cut_             (vertex_.getParameter<double>               ("AVR_chi2cut")),
  tdr_vx_width_           (vertex_.getParameter<double>               ("TDR_VertexWidth")),
  vx_recodistance_        (vertex_.getParameter<double>               ("RecoVertexDistance")),
  vx_minHighPtTracks_     (vertex_.getParameter<unsigned int>         ("MinHighPtTracks")),
  // Debug printout
  debug_                  ( iConfig.getParameter<unsigned int>                ( "Debug"                  ) ),
  iPhiPlot_               ( iConfig.getParameter<unsigned int>                ( "iPhiPlot"               ) ),
  iEtaPlot_               ( iConfig.getParameter<unsigned int>                ( "iEtaPlot"               ) ),

  // Name of output EDM file if any.
  // N.B. This parameter does not appear inside TMTrackProducer_Defaults_cfi.py . It is created inside
  // tmtt_tf_analysis_cfg.py .
  writeOutEdmFile_        ( iConfig.getUntrackedParameter<bool>               ( "WriteOutEdmFile", true) ),

  // Bfield in Tesla. (Unknown at job initiation. Set to true value for each event
  bField_                 (0.)
 
{
  // If user didn't specify any PDG codes, use e,mu,pi,K,p, to avoid picking up unstable particles like Xi-.
  vector<unsigned int> genPdgIdsUnsigned( genCuts_.getParameter<std::vector<unsigned int> >   ( "GenPdgIds" ) ); 
  if (genPdgIdsUnsigned.empty()) {
    genPdgIdsUnsigned = {11, 13, 211, 321, 2212};  
  }
   
  // For simplicity, user need not distinguish particles from antiparticles in configuration file.
  // But here we must store both explicitely in Settings, since TrackingParticleSelector expects them.
  for (unsigned int i = 0; i < genPdgIdsUnsigned.size(); i++) {
    genPdgIds_.push_back(  genPdgIdsUnsigned[i] );
    genPdgIds_.push_back( -genPdgIdsUnsigned[i] );
  }
 
  //--- Sanity checks
 
  if ( ! (useStubPhi_ || useStubPhiTrk_) ) throw cms::Exception("Settings.cc: Invalid cfg parameters - You cant set both UseStubPhi & useStubPhiTrk to false.");
 
  if (minNumMatchLayers_ > minStubLayers_)    throw cms::Exception("Settings.cc: Invalid cfg parameters - You are setting the minimum number of layers incorrectly : type A.");
  if (genMinStubLayers_  > minStubLayers_)    throw cms::Exception("Settings.cc: Invalid cfg parameters - You are setting the minimum number of layers incorrectly : type B.");
  if (minNumMatchLayers_ > genMinStubLayers_) throw cms::Exception("Settings.cc: Invalid cfg parameters - You are setting the minimum number of layers incorrectly : type C.");
 
  // If reducing number of required layers for high Pt tracks, then above checks must be redone.
  bool doReduceLayers = (minPtToReduceLayers_ < 10000. || etaSecsReduceLayers_.size() > 0) ;
  if (doReduceLayers) {
    if (minNumMatchLayers_ > minStubLayers_ - 1) throw cms::Exception("Settings.cc: Invalid cfg parameters - You are setting the minimum number of layers incorrectly : type D.");
    if (genMinStubLayers_  > minStubLayers_ - 1) throw cms::Exception("Settings.cc: Invalid cfg parameters - You are setting the minimum number of layers incorrectly : type E.");
  }
 
  for (const unsigned int& iEtaReg : etaSecsReduceLayers_) {
    if (iEtaReg >= etaRegions_.size()) throw cms::Exception("Settings.cc: You specified an eta sector number in EtaSecsReduceLayers which exceeds the total number of eta sectors!")<<iEtaReg<<" "<<etaRegions_.size()<<endl;
  }
 
  // Duplicate track removal algorithm 50 must not be run in parallel with any other.
  if (dupTrkAlgFit_ == 50) {
    if (dupTrkAlgRphi_ != 0 || dupTrkAlgRz_ != 0 || dupTrkAlgRzSeg_ != 0) throw cms::Exception("Settings.c: Invalid cfg parameters -- If using DupTrkAlgFit = 50, you must disable all other duplicate track removal algorithms.");
  }
 
  for (const unsigned int& iEtaReg : etaSecsReduceLayers_) {
    if (iEtaReg >= etaRegions_.size()) throw cms::Exception("Settings.cc: You specified an eta sector number in EtaSecsReduceLayers which exceeds the total number of eta sectors!")<<iEtaReg<<" "<<etaRegions_.size()<<endl;
  }

  // Duplicate track removal algorithm 50 must not be run in parallel with any other.
  if (dupTrkAlgFit_ == 50) {
    if (dupTrkAlgRphi_ != 0 || dupTrkAlgRz_ != 0 || dupTrkAlgRzSeg_ != 0) throw cms::Exception("Settings.c: Invalid cfg parameters -- If using DupTrkAlgFit = 50, you must disable all other duplicate track removal algorithms.");
  }

  // Assunme user will only enable r-z Hough transform & r-z track filters simultaneously by mistake.
  if (enableRzHT_ && (useEtaFilter_ || useSeedFilter_) ) throw cms::Exception("Settings.cc: Invalid cfg parameters - You are trying to use r-z Hough transform & r-z track filters simultaneously");
  // Chains of m bin ranges for output of HT.
  if ( ! busySectorMbinOrder_.empty() ) {
    // User has specified an order in which the m bins should be chained together. Check if it makes sense.
    if (busySectorMbinOrder_.size() != houghNbinsPt_) throw cms::Exception("Settings.cc: Invalid cfg parameters - BusySectorMbinOrder contains wrong number of elements");
    unsigned int sum_nr = 0;
    for (unsigned int nr : busySectorMbinRanges_) {
      sum_nr += nr;
    }
    if (sum_nr != houghNbinsPt_) throw cms::Exception("Settings.cc: Invalid cfg parameters - Sum of entries in BusySectorMbinRanges is incorrect.");
  }

  // Chains of m bin ranges for output of HT.
  if ( ! busySectorMbinOrder_.empty() ) {
    // User has specified an order in which the m bins should be chained together. Check if it makes sense.
    if (busySectorMbinOrder_.size() != houghNbinsPt_) throw cms::Exception("Settings.cc: Invalid cfg parameters - BusySectorMbinOrder used by HT MUX contains wrong number of elements. Unless you are optimising the MUX, suggest you configure it to an empty vector.");
    unsigned int sum_nr = 0;
    for (unsigned int nr : busySectorMbinRanges_) {
      sum_nr += nr;
    }
    if (sum_nr != houghNbinsPt_) throw cms::Exception("Settings.cc: Invalid cfg parameters - Sum of entries in BusySectorMbinRanges is incorrect.");
  }

  // Merging of HT cells has not yet been implemented for diamond or hexagonal HT cell shape.
  if (enableMerge2x2_ && shape_ != 0) throw cms::Exception("Settings.cc: Invalid cfg parameters - merging only allowed for square-shaped cells.");
}

bool Settings::isHTRPhiEtaRegWhitelisted(unsigned const iEtaReg) const
{
  bool whitelisted = true;

  bool const whitelist_enabled = ( ! etaRegWhitelist_.empty() );
  if (whitelist_enabled) {
    whitelisted = (std::count(etaRegWhitelist_.begin(), etaRegWhitelist_.end(), iEtaReg) > 0);
  }

  return whitelisted;
}

} // end namespace vertexFinder

#include <TMTrackTrigger/VertexFinder/interface/Settings.h>
#include "FWCore/Utilities/interface/Exception.h"
 

namespace vertexFinder {

///=== Get configuration parameters

Settings::Settings(const edm::ParameterSet& iConfig) :
 
  // See either Analyze_Defaults_cfi.py or Settings.h for description of these parameters.
 
  // Parameter sets for differents types of configuration parameter.
  genCuts_                ( iConfig.getParameter< edm::ParameterSet >         ( "GenCuts"                ) ),
  l1TrackDef_             ( iConfig.getParameter< edm::ParameterSet >         ( "L1TrackDef"             ) ),
  trackMatchDef_          ( iConfig.getParameter< edm::ParameterSet >         ( "TrackMatchDef"          ) ),
  vertex_                 ( iConfig.getParameter< edm::ParameterSet >         ( "VertexReconstruction" )), 

  //=== Cuts on MC truth tracks used for tracking efficiency measurements.
  genMinPt_               ( genCuts_.getParameter<double>                     ( "GenMinPt"               ) ),
  genMaxAbsEta_           ( genCuts_.getParameter<double>                     ( "GenMaxAbsEta"           ) ),
  genMaxVertR_            ( genCuts_.getParameter<double>                     ( "GenMaxVertR"            ) ),
  genMaxVertZ_            ( genCuts_.getParameter<double>                     ( "GenMaxVertZ"            ) ),
  genMinStubLayers_       ( genCuts_.getParameter<unsigned int>               ( "GenMinStubLayers"       ) ),

  //=== Rules for deciding when the track finding has found an L1 track candidate
 
  useLayerID_             ( l1TrackDef_.getParameter<bool>                    ( "UseLayerID"             ) ),
  reduceLayerID_          ( l1TrackDef_.getParameter<bool>                    ( "ReducedLayerID"         ) ),
 
  //=== Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).
 
  minFracMatchStubsOnReco_( trackMatchDef_.getParameter<double>               ( "MinFracMatchStubsOnReco") ),
  minFracMatchStubsOnTP_  ( trackMatchDef_.getParameter<double>               ( "MinFracMatchStubsOnTP"  ) ),
  minNumMatchLayers_      ( trackMatchDef_.getParameter<unsigned int>         ( "MinNumMatchLayers"      ) ),
  minNumMatchPSLayers_    ( trackMatchDef_.getParameter<unsigned int>         ( "MinNumMatchPSLayers"    ) ),
  stubMatchStrict_        ( trackMatchDef_.getParameter<bool>                 ( "StubMatchStrict"        ) ),

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
 
  if (minNumMatchLayers_ > genMinStubLayers_)
    throw cms::Exception("Settings.cc: Invalid cfg parameters - You are setting the minimum number of layers incorrectly : type C.");
  
}

} // end namespace vertexFinder

#ifndef __TMTrackTrigger_VertexFinder_Settings_h__
#define __TMTrackTrigger_VertexFinder_Settings_h__
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <vector>
#include <iostream>
 
using namespace std;
 

namespace vertexFinder {

// Stores all configuration parameters + some hard-wired constants.
 
class Settings {
 
public:
  Settings(const edm::ParameterSet& iConfig);
  ~Settings(){}
 
  //=== Cuts on MC truth tracks for tracking efficiency measurements.
 
  double               genMinPt()                const   {return genMinPt_;}
  double               genMaxAbsEta()            const   {return genMaxAbsEta_;}
  double               genMaxVertR()             const   {return genMaxVertR_;}
  double               genMaxVertZ()             const   {return genMaxVertZ_;}
  vector<int>          genPdgIds()               const   {return genPdgIds_;}
  // Additional cut on MC truth tracks for algorithmic tracking efficiency measurements.
  unsigned int         genMinStubLayers()        const   {return genMinStubLayers_;} // Min. number of layers TP made stub in.

  //=== Rules for deciding when the track finding has found an L1 track candidate
 
  // Define layers using layer ID (true) or by bins in radius of 5 cm width (false)?
  bool                 useLayerID()              const   {return useLayerID_;}
  //Reduce this layer ID, so that it takes no more than 8 different values in any eta region (simplifies firmware)?
  bool                 reduceLayerID()           const   {return reduceLayerID_;}
 
  //=== Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).
 
  //--- Three different ways to define if a tracking particle matches a reco track candidate. (Usually, set two of them to ultra loose).
  // Min. fraction of matched stubs relative to number of stubs on reco track.
  double               minFracMatchStubsOnReco() const   {return minFracMatchStubsOnReco_;}
  // Min. fraction of matched stubs relative to number of stubs on tracking particle.
  double               minFracMatchStubsOnTP()   const   {return minFracMatchStubsOnTP_;}
  // Min. number of matched layers & min. number of matched PS layers..
  unsigned int         minNumMatchLayers()       const   {return minNumMatchLayers_;}
  unsigned int         minNumMatchPSLayers()     const   {return minNumMatchPSLayers_;}
  // Associate stub to TP only if the TP contributed to both its clusters? (If False, then associate even if only one cluster was made by TP).
  bool                 stubMatchStrict()         const   {return stubMatchStrict_;}

  //=== Vertex Reconstruction configuration
  // Vertex Reconstruction Id (0: GapClustering, 1: SimpleMergeClustering )
  unsigned int        vx_algoId()                const {return vx_algoId_;        }
  // Assumed Vertex Resolution
  float               vx_distance()              const {return vx_distance_;      }
  // Minimum number of tracks to accept vertex
  unsigned int        vx_minTracks()             const {return vx_minTracks_;     }
  /// Chi2 cut for the Adaptive Vertex Recostruction Algorithm
  float               vx_chi2cut()               const {return vx_chi2cut_;       }
  /// TDR assumed vertex width
  float               tdr_vx_width()             const {return tdr_vx_width_;     }
  /// Maximum distance between reconstructed and generated vertex, in order to consider the vertex as correctly reconstructed
  float               vx_recodistance()          const {return vx_recodistance_;  }
  /// Minimum number of high pT (pT > 10 GeV) tracks that the vertex has to contain to be a good hard interaction vertex candidate
  unsigned int        vx_minHighPtTracks()       const {return vx_minHighPtTracks_;}


   //=== Debug printout
  unsigned int         debug()                   const   {return debug_;}

 
  //=== Hard-wired constants
  // EJC Check this.  Found stub at r = 109.504 with flat geometry in 81X, so increased tracker radius for now.
  double               trackerOuterRadius()      const   {return 120.2;}  // max. occuring stub radius.
  // EJC Check this.  Found stub at r = 20.664 with flat geometry in 81X, so decreased tracker radius for now.
  double               trackerInnerRadius()      const   {return  20;}  // min. occuring stub radius.
  double               trackerHalfLength()       const   {return 270.;}  // half-length  of tracker. 
  double               layerIDfromRadiusBin()    const   {return 6.;}    // When counting stubs in layers, actually histogram stubs in distance from beam-line with this bin size.
 
  //=== Set and get B-field value in Tesla.
  // N.B. This must bet set for each event, and can't be initialized at the beginning of the job.
  void                 setBfield(float bField)           {bField_ = bField;}
  float                getBfield()               const   {if (bField_ == 0.) throw cms::Exception("Settings.h:You attempted to access the B field before it was initialized"); return bField_;}
 
private:
 
  // Parameter sets for differents types of configuration parameter.
  edm::ParameterSet    genCuts_;
  edm::ParameterSet    l1TrackDef_;
  edm::ParameterSet    trackMatchDef_;
  edm::ParameterSet    vertex_;

  // Cuts on truth tracking particles.
  double               genMinPt_;
  double               genMaxAbsEta_;
  double               genMaxVertR_;
  double               genMaxVertZ_;
  vector<int>          genPdgIds_;
  unsigned int         genMinStubLayers_;

  // Rules for deciding when the track-finding has found an L1 track candidate
  bool                 useLayerID_;
  bool                 reduceLayerID_;
 
  // Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).
  double               minFracMatchStubsOnReco_;
  double               minFracMatchStubsOnTP_;
  unsigned int         minNumMatchLayers_;
  unsigned int         minNumMatchPSLayers_;
  bool                 stubMatchStrict_;
 
  // Track Fitting Settings
  vector<string>       trackFitters_;
  double               chi2OverNdfCut_;
  bool                 detailedFitOutput_;
  unsigned int         numTrackFitIterations_;
  bool                 killTrackFitWorstHit_;
  double               generalResidualCut_;
  double               killingResidualCut_;


  // Vertex Reconstruction configuration
  unsigned int         vx_algoId_;
  float                vx_distance_;
  unsigned int         vx_minTracks_;
  float                vx_chi2cut_;
  float                tdr_vx_width_;
  float                vx_recodistance_;
  unsigned int         vx_minHighPtTracks_; 

  // Debug printout
  unsigned int         debug_;

  // B-field in Tesla
  float                bField_;
};

} // end namespace vertexFinder

#endif

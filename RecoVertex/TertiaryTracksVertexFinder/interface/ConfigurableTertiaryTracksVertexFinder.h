
#ifndef _ConfigurableTertiaryTracksVertexFinder_H_
#define _ConfigurableTertiaryTracksVertexFinder_H_

#include "RecoVertex/VertexPrimitives/interface/VertexReconstructor.h"
#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/VertexUpdator.h"
#include "RecoVertex/VertexPrimitives/interface/VertexTrackCompatibilityEstimator.h"
#include <vector>

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "RecoVertex/TrimmedKalmanVertexFinder/interface/ConfigurableTrimmedVertexFinder.h"

class V0SvFilter;
class Flight2DSvFilter;

class ConfigurableTertiaryTracksVertexFinder : public VertexReconstructor {

  public:

  ConfigurableTertiaryTracksVertexFinder(const VertexFitter * vf, 
    const VertexUpdator * vu, const VertexTrackCompatibilityEstimator * ve);

  virtual ~ConfigurableTertiaryTracksVertexFinder();

  virtual std::vector<TransientVertex> 
    vertices(const std::vector<reco::TransientTrack> &) const; 

  virtual std::vector<TransientVertex> 
    vertices(const std::vector<reco::TransientTrack> &, const TransientVertex& pv) const; 
 
  virtual ConfigurableTertiaryTracksVertexFinder * clone() const {
    return new ConfigurableTertiaryTracksVertexFinder(*this);
  }


  //inline void setPrimaryVertex(TransientVertex& ThePrimaryVertex) { 
  //  thePrimaryVertex = & ThePrimaryVertex; 
  //}

  private:

  std::vector<TransientVertex> reconstruct(
    const std::vector<reco::TransientTrack> & tracks, 
    const TransientVertex& pv) const; 

  static const bool debug = true;

  ConfigurableTrimmedVertexFinder* theTKVF;
  V0SvFilter* theV0SvFilter;
  Flight2DSvFilter* theFlight2DSvFilter;

  double theMinTrackPt,theMaxVtxMass;
  double theK0sMassWindow;
  double theMaxSigOnDistTrackToB;

  // parameters for Flight2DSvFilter
  double theMaxDist2D,theMinDist2D,theMinSign2D;
  int theMinTracks;

  //  TransientVertex* thePrimaryVertex;

};

#endif

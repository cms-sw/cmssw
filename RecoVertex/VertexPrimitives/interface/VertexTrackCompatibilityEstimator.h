#ifndef _VertexTrackCompatibilityEstimator_H
#define _VertexTrackCompatibilityEstimator_H

#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoVertex/VertexPrimitives/interface/DummyRecTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

/**
 * Pure abstract base class for compatibility estimators 
 * (distance, chi-squared, etc.)
 */

class VertexTrackCompatibilityEstimator {
 
public:

  VertexTrackCompatibilityEstimator(){}
  virtual ~VertexTrackCompatibilityEstimator(){}
  
  /**
   * Methods giving back the compatibility estimation
   */
  virtual float estimate(const CachingVertex & v, 
			 const RefCountedLinearizedTrackState track) const = 0;

  virtual float estimate(const reco::Vertex & v, 
			 const reco::TransientTrack & track) const = 0;

  // obsolete ?
  virtual float estimate(const CachingVertex & v, 
			 const RefCountedVertexTrack track) const = 0;
  /**
   * Clone method 
   */
  virtual VertexTrackCompatibilityEstimator * clone() const = 0; 
  
};

#endif

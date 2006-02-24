#ifndef KalmanVertexTrackCompatibilityEstimator_H
#define KalmanVertexTrackCompatibilityEstimator_H


#include "RecoVertex/VertexPrimitives/interface/VertexTrackCompatibilityEstimator.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "RecoVertex/VertexPrimitives/interface/VertexTrack.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackUpdator.h"
#include "RecoVertex/VertexTools/interface/LinearizedTrackStateFactory.h"
#include "RecoVertex/VertexTools/interface/VertexTrackFactory.h"

  /**
   * Calculates the compatiblity of a track with respect to a vertex 
   * using the Kalman filter algorithms. 
   * The compatibility is computed from the squared standardized residuals 
   * between the track and the vertex. 
   * If track and vertex errors are Gaussian and correct, 
   * this quantity is distributed as chi**2(ndf=2)). 
   * Can be used to identify outlying tracks.
   */

class KalmanVertexTrackCompatibilityEstimator:public VertexTrackCompatibilityEstimator
{

public:

  KalmanVertexTrackCompatibilityEstimator(){}

  virtual ~KalmanVertexTrackCompatibilityEstimator(){}

  /**
   * Track-to-vertex compatibility. 
   * The track weight is taken into account.
   * \param track The track for which the chi**2 has to be estimated.
   * \param v The vertex against which the chi**2 has to be estimated.
   * \return The chi**2.
   */

  virtual float estimate(const CachingVertex & vrt, const RefCountedVertexTrack track) const;

  virtual float estimate(const CachingVertex & v, 
			 const RefCountedLinearizedTrackState track) const;

//   virtual float estimate(const RecVertex & v, 
// 			 const RecTrack & track) const;

  virtual KalmanVertexTrackCompatibilityEstimator * clone() const
  {
    return new KalmanVertexTrackCompatibilityEstimator(* this);
  }


private:

  float estimateFittedTrack(const CachingVertex & v, const RefCountedVertexTrack track) const;
  float estimateNFittedTrack(const CachingVertex & v, const RefCountedVertexTrack track) const;  
  float estimateDifference(const CachingVertex & more, const CachingVertex & less, 
                                                       const RefCountedVertexTrack track) const;
  KalmanVertexUpdator updator;
  KalmanVertexTrackUpdator trackUpdator;
  LinearizedTrackStateFactory lTrackFactory;
  VertexTrackFactory vTrackFactory;

};

#endif

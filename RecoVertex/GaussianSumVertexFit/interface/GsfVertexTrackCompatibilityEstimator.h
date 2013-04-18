#ifndef GsfVertexTrackCompatibilityEstimator_H
#define GsfVertexTrackCompatibilityEstimator_H


#include "RecoVertex/VertexPrimitives/interface/VertexTrackCompatibilityEstimator.h"
#include "RecoVertex/VertexPrimitives/interface/VertexTrack.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "RecoVertex/VertexTools/interface/VertexTrackFactory.h"
#include "RecoVertex/GaussianSumVertexFit/interface/MultiPerigeeLTSFactory.h"
#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexUpdator.h"

#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KVFHelper.h"

  /**
   * Calculates the compatiblity of a track with respect to a vertex 
   * using the Kalman filter algorithms. 
   * The compatibility is computed from the squared standardized residuals 
   * between the track and the vertex. 
   * If track and vertex errors are Gaussian and correct, 
   * this quantity is distributed as chi**2(ndf=2)). 
   * Can be used to identify outlying tracks.
   */

class GsfVertexTrackCompatibilityEstimator:public VertexTrackCompatibilityEstimator<5>
{

public:

  typedef CachingVertex<5>::RefCountedVertexTrack RefCountedVertexTrack;

  GsfVertexTrackCompatibilityEstimator(){}

  virtual ~GsfVertexTrackCompatibilityEstimator(){}

  /**
   * Track-to-vertex compatibility. 
   * The track weight is taken into account.
   * \param track The track for which the chi**2 has to be estimated.
   * \param v The vertex against which the chi**2 has to be estimated.
   * \return The chi**2.
   */

  virtual BDpair estimate(const CachingVertex<5> & vrt, const RefCountedVertexTrack track,
			  unsigned int hint=UINT_MAX) const;

  virtual BDpair estimate(const CachingVertex<5> & v, 
			  const RefCountedLinearizedTrackState track,
			  unsigned int hint=UINT_MAX) const;

  virtual BDpair estimate(const reco::Vertex & vertex, 
			 const reco::TransientTrack & track) const;

  virtual GsfVertexTrackCompatibilityEstimator * clone() const
  {
    return new GsfVertexTrackCompatibilityEstimator(* this);
  }


private:

  BDpair estimateFittedTrack(const CachingVertex<5> & v, const RefCountedVertexTrack track) const;
  BDpair estimateNFittedTrack(const CachingVertex<5> & v, const RefCountedVertexTrack track) const;  

  GsfVertexUpdator updator;
//   KalmanVertexTrackUpdator trackUpdator;
  MultiPerigeeLTSFactory lTrackFactory;
  VertexTrackFactory<5> vTrackFactory;
//   KVFHelper helper;

};

#endif

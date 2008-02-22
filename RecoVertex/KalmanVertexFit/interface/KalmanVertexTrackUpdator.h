#ifndef KalmanVertexTrackUpdator_H
#define KalmanVertexTrackUpdator_H

#include "RecoVertex/VertexPrimitives/interface/VertexTrackUpdator.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "RecoVertex/VertexPrimitives/interface/VertexTrack.h"
#include "RecoVertex/VertexTools/interface/VertexTrackFactory.h"
#include "RecoVertex/KalmanVertexFit/interface/KVFHelper.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"

/** \class KalmanVertexTrackUpdator
 *  Performs the refit of the tracks with the vertex constraint, 
 *  using the Kalman filter algorithms.
 */

class KalmanVertexTrackUpdator : public VertexTrackUpdator {

public:


  /**
   *  Default constructor
   */

  KalmanVertexTrackUpdator(){}

  virtual ~KalmanVertexTrackUpdator(){}

  /**
   *   Refit of the track with the vertex constraint.
   *   \param vertex The vertex which has to be used as constraint.
   *   \param track  The track to refit.
   *   \return	The VertexTrack containing the refitted track and 
   *		the track-to-vertex covariance.
   */

  RefCountedVertexTrack update(const CachingVertex & vertex,
                               RefCountedVertexTrack track) const;

  /**
   *   Refit of the track with the vertex constraint.
   *   \param vertex The vertex which has to be used as constraint.
   *   \param track  The track to refit.
   *   \return	The refitted state with the track-to-vertex covariance.
   */

  pair<RefCountedRefittedTrackState, AlgebraicMatrix> 
	trackRefit(const VertexState & vertex,
		RefCountedLinearizedTrackState linTrackState) const;

  /**
   *  Clone method
   */

  KalmanVertexTrackUpdator * clone() const
  {
    return new KalmanVertexTrackUpdator(*this);
  }

private:
  VertexTrackFactory theVTFactory;
  KVFHelper helper;
  KalmanVertexUpdator updator;
};

#endif

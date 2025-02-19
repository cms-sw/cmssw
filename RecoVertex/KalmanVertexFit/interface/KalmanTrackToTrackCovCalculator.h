#ifndef KalmanTrackToTrackCovCalculator_H
#define KalmanTrackToTrackCovCalculator_H

#include "RecoVertex/VertexPrimitives/interface/TrackToTrackCovCalculator.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"

  /**
   * Calculates all the track-to-track covariance matrices using the Kalman 
   * filter algorithms after the vertex has been fit and the tracks refit.
   */

template <unsigned int N>
class KalmanTrackToTrackCovCalculator:public TrackToTrackCovCalculator<N>
{

public: 
 
  typedef typename CachingVertex<N>::RefCountedVertexTrack RefCountedVertexTrack;

  KalmanTrackToTrackCovCalculator() {}

  /**
   * Calculates all the track-to-track covariance matrices
   * \param vertex The vertex whose track-to-track covariance matrices have 
   * 	to be calculated.
   * \return The map containing the covariance matrices.
   */

 typename CachingVertex<N>::TrackToTrackMap operator() (const CachingVertex<N> & vertex) const;
 
 KalmanTrackToTrackCovCalculator * clone() const
 {
   return new KalmanTrackToTrackCovCalculator(* this);
 }

};


#endif

#ifndef KalmanTrackToTrackCovCalculator_H
#define KalmanTrackToTrackCovCalculator_H

#include "RecoVertex/VertexPrimitives/interface/TrackToTrackCovCalculator.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"

  /**
   * Calculates all the track-to-track covariance matrices using the Kalman 
   * filter algorithms after the vertex has been fit and the tracks refit.
   */

class KalmanTrackToTrackCovCalculator:public TrackToTrackCovCalculator
{

public: 
 
 KalmanTrackToTrackCovCalculator() {}

  /**
   * Calculates all the track-to-track covariance matrices
   * \param vertex The vertex whose track-to-track covariance matrices have 
   * 	to be calculated.
   * \return The map containing the covariance matrices.
   */

 TrackToTrackMap operator() (const CachingVertex & vertex) const;
 
 KalmanTrackToTrackCovCalculator * clone() const
 {
   return new KalmanTrackToTrackCovCalculator(* this);
 }

};


#endif

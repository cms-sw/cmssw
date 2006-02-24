#ifndef KalmanVertexUpdator_H
#define KalmanVertexUpdator_H

#include "RecoVertex/VertexPrimitives/interface/VertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackUpdator.h"

/**
 *  Vertex updator for the Kalman vertex filter.
 *  (c.f. R. Fruewirth et.al., Comp.Phys.Comm 96 (1996) 189
 */
class KalmanVertexTrackCompatibilityEstimator;

class KalmanVertexUpdator: public VertexUpdator {

public:

/**
 *  Method to add a track to an existing CachingVertex
 *
 */

   CachingVertex add(const CachingVertex & oldVertex,
        const RefCountedVertexTrack track) const;

/**
 *  Method removing already used VertexTrack from existing CachingVertex
 *
 */

   CachingVertex remove(const CachingVertex & oldVertex,
        const RefCountedVertexTrack track) const;

/**
 * Clone method
 */

   VertexUpdator * clone() const
   {
    return new KalmanVertexUpdator(* this);
   }


private:

    /**
     * Calculates the chi**2 increment
     */

    float vertexPositionChi2(const VertexState& oldVertex,
                             const GlobalPoint& newVertexPosition) const;

    KalmanVertexTrackUpdator trackUpdator;

public:

  friend class KalmanVertexTrackCompatibilityEstimator;

    /**
     * The methode which actually does the vertex update.
     */
  CachingVertex update(const CachingVertex & oldVertex,
                         const RefCountedVertexTrack track, float weight,
                         int sign ) const;

  VertexState positionUpdate (const VertexState & oldVertex,
	 const RefCountedLinearizedTrackState linearizedTrack, 
	 const float weight, int sign) const;

  double chi2Increment(const VertexState & oldVertex, 
	 const VertexState & newVertexState,
	 const RefCountedLinearizedTrackState linearizedTrack, 
	 float weight) const; 

};

#endif

#ifndef VertexTrack_H
#define VertexTrack_H

#include "RecoVertex/VertexPrimitives/interface/RefCountedLinearizedTrackState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "RecoVertex/VertexPrimitives/interface/RefCountedRefittedTrackState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

/** Track information relative to a track-to-vertex association. 
 *  The track weight corresponds to the distance 
 *  of the track to the seed position. 
 */

class VertexTrack : public ReferenceCounted {

public:

  /** Constructor with the linearized track data, vertex seed and weight
   */     
  VertexTrack(const RefCountedLinearizedTrackState lt, 
	      const VertexState v, 
	      float weight);

  /** Constructor with the linearized track data, vertex seed and weight
   *  and state at vertex, constrained by vertex
   */     
  VertexTrack(const RefCountedLinearizedTrackState lt, 
	      const VertexState v, 
	      float weight, const RefCountedRefittedTrackState & refittedState);

  /** Constructor with the linearized track data, vertex seed and weight
   *  and state and covariance at vertex, constrained by vertex
   */     
  VertexTrack(const RefCountedLinearizedTrackState lt, 
	      const VertexState v, 
	      float weight, const RefCountedRefittedTrackState & refittedState,
	      const AlgebraicMatrix & tVCov);

  /** Access methods
   */ 
  RefCountedLinearizedTrackState linearizedTrack() const { return theLinTrack; }
  VertexState vertexState() const { return theVertexState; }
  float weight() const { return theWeight; }
  bool refittedStateAvailable() const { return stAvailable; }
  bool tkToVertexCovarianceAvailable() const { return covAvailable; }

  /** Track state with vertex constraint
   */
  RefCountedRefittedTrackState refittedState() const { 
    if (!refittedStateAvailable()) { 
      throw VertexException("VertexTrack::refitted state not available"); 
    }
    return theRefittedState;
  }

  /** Track to vertex covariance 
   */   
  AlgebraicMatrix tkToVtxCovariance() const {
    if (!tkToVertexCovarianceAvailable()) {
      throw VertexException("VertexTrack::track to vertex covariance not available"); 
    }
    return tkTVCovariance;
  }

  /** Equality for finding a VertexTrack in a container
   *  Compares the RecTrack addresses
   */
  bool operator==(const VertexTrack & data) const
  {
    return ((*data.linearizedTrack().get()) == (*linearizedTrack().get()));
  }
  
  /** Method helping Kalman vertex fit
   */
  AlgebraicVector refittedParamFromEquation() const;
 

private:

  RefCountedLinearizedTrackState theLinTrack;
  VertexState theVertexState;
  float theWeight;
  bool stAvailable;
  bool covAvailable;
  RefCountedRefittedTrackState theRefittedState;
  AlgebraicMatrix  tkTVCovariance;
};

#endif

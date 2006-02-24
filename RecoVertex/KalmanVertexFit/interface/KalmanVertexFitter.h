#ifndef KalmanVertexFitter_H
#define KalmanVertexFitter_H

#include "RecoVertex/VertexTools/interface/SequentialVertexFitter.h"

/** An easy way to use the SequentialVertexFitter with the Kalman filter.
 *  As it is a VertexFitter, it is then to be used as such.
 *  The configurable parameters of the SequentialVertexFitter can still be
 *  set with either the set methods or the simpleConfigurables.
 *  By default, no smoothing is done, but it can still be chosen by the 
 *  boolean parameter in the constructor.
 */

class KalmanVertexFitter : public VertexFitter {
public:

  /**
   * The constructor, setting everything up to have a VertexFitter using the 
   * Kalman algorithm.
   * \param useSmoothing Specifies whether the tracks should be refit or not.
   */

  KalmanVertexFitter(bool useSmoothing = false);
  KalmanVertexFitter(const KalmanVertexFitter & other ) :
    theSequentialFitter ( other.theSequentialFitter->clone() ) {}

  virtual ~KalmanVertexFitter()
  {
    delete theSequentialFitter;
  }

  KalmanVertexFitter * clone() const
  {
    return new KalmanVertexFitter(* this);
  }

public:

  /** Fit vertex out of a set of RecTracks
   */
  virtual inline CachingVertex vertex(const vector<RecTrack> & tracks) const
  {
    return theSequentialFitter->vertex(tracks);
  }

  /** Fit vertex out of a set of VertexTracks
   */
  virtual inline CachingVertex 
  vertex(const vector<RefCountedVertexTrack> & tracks) const
  {
    return theSequentialFitter->vertex(tracks);
  }

  /** Fit vertex out of a set of RecTracks. 
   *  Uses the specified linearization point.
   */
  virtual inline CachingVertex 
  vertex(const vector<RecTrack> & tracks, const GlobalPoint& linPoint) const
  {
    return theSequentialFitter->vertex(tracks, linPoint);
  }

  /** Fit vertex out of a set of RecTracks. 
   *  Uses the specified point as both the linearization point AND as prior
   *  estimate of the vertex position. The error is used for the 
   *  weight of the prior estimate.
   */
  virtual inline CachingVertex 
  vertex(const vector<RecTrack> & tracks, const GlobalPoint& priorPos,
  	 const GlobalError& priorError) const
  {
    return theSequentialFitter->vertex(tracks, priorPos, priorError);
  }

  /** Fit vertex out of a set of VertexTracks.
   *  Uses the specified point and error as the prior estimate of the vertex.
   *  This position is not used to relinearize the tracks.
   */
  virtual inline CachingVertex 
  vertex(const vector<RefCountedVertexTrack> & tracks, 
	 const GlobalPoint& priorPos,
	 const GlobalError& priorError) const
  {
    return theSequentialFitter->vertex(tracks, priorPos, priorError);
  }

private:

  const SequentialVertexFitter * theSequentialFitter;
};

#endif

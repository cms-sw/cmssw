#ifndef KalmanVertexFitter_H
#define KalmanVertexFitter_H

#include "RecoVertex/VertexTools/interface/SequentialVertexFitter.h"

/** Least-squares vertex fitter implemented in the Kalman Filter formalism 
 *  Fits vertex position and, if smoothing is requested at construction, 
 *  constrains track parameters at the vertex. 
 *  A beam spot constraint can also be specified in form of a 3D point 
 *  and a 3D matrix describing a Gaussian beam spread. 
 *
 *  References: 
 *
 * <a href="http://cms.cern.ch/iCMS/jsp/openfile.jsp?type=NOTE&year=2006&files=NOTE2006_032.pdf"> T.Speer, K.Prokofiev, R.Fruhwirth, W.Waltenberger, P.Vanlaer, "Vertex Fitting in the CMS Tracker", CMS Note 2006/032</a>
 *
 * P.Billoir, S.Qian, "Fast vertex fitting...", NIM A311 (1992) 139. 
 *
 * R.Fruhwirth, R.Kubinec, W.Mitaroff, M.Regler, Comp. Physics Comm. 96, 189 (1996)
 * 
 * <a href="http://www.phys.ufl.edu/~avery/fitting.html"> P.Avery, lectures 
on track and vertex fitting" </a>
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
  virtual inline CachingVertex 
    vertex(const std::vector<reco::TransientTrack>  & tracks) const
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
    vertex(const std::vector<reco::TransientTrack>  & tracks, 
	   const GlobalPoint& linPoint) const
  {
    return theSequentialFitter->vertex(tracks, linPoint);
  }

  /** Fit vertex out of a set of RecTracks. 
   *  Uses the specified point as both the linearization point AND as prior
   *  estimate of the vertex position. The error is used for the 
   *  weight of the prior estimate.
   */
  virtual inline CachingVertex 
  vertex(const std::vector<reco::TransientTrack> & tracks, 
	 const GlobalPoint& priorPos,
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

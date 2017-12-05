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
 * <a href="http://cmsdoc.cern.ch/documents/03/in/in03_008.pdf"> `Vertex fitting with the Kalman filter formalism', T.speer, K.Prokofiev, R.Fruehwirth, CMS IN 2003/008</a><br>
 * <a href="http://cms.cern.ch/iCMS/jsp/openfile.jsp?type=NOTE&year=2006&files=NOTE2006_032.pdf"> T.Speer, K.Prokofiev, R.Fruhwirth, W.Waltenberger, P.Vanlaer, "Vertex Fitting in the CMS Tracker", CMS Note 2006/032</a><br>
 * P.Billoir, S.Qian, "Fast vertex fitting...", NIM A311 (1992) 139. <br>
 * R.Fruhwirth, R.Kubinec, W.Mitaroff, M.Regler, Comp. Physics Comm. 96, 189 (1996)<br>
 * <a href="http://www.phys.ufl.edu/~avery/fitting.html"> P.Avery, lectures 
on track and vertex fitting" </a>
 */

class KalmanVertexFitter : public VertexFitter<5> {
public:

  /**
   * The constructor, setting everything up to have a VertexFitter using the 
   * Kalman algorithm.
   * \param useSmoothing Specifies whether the tracks should be refit or not.
   */

  KalmanVertexFitter(bool useSmoothing = false);

  /**
   * Same as above, using a ParameterSet to set the convergence criteria
   */

  KalmanVertexFitter(const edm::ParameterSet& pSet, bool useSmoothing = false);

  KalmanVertexFitter(const KalmanVertexFitter & other ) :
    theSequentialFitter ( other.theSequentialFitter->clone() ) {}

  ~KalmanVertexFitter() override
  {
    delete theSequentialFitter;
  }

  KalmanVertexFitter * clone() const override
  {
    return new KalmanVertexFitter(* this);
  }

public:

  typedef CachingVertex<5>::RefCountedVertexTrack RefCountedVertexTrack;

  /** Fit vertex out of a set of RecTracks
   */
  inline CachingVertex<5> 
    vertex(const std::vector<reco::TransientTrack>  & tracks) const override
  {
    return theSequentialFitter->vertex(tracks);
  }

  /** Fit vertex out of a set of VertexTracks
   */
  inline CachingVertex<5> 
  vertex(const std::vector<RefCountedVertexTrack> & tracks) const override
  {
    return theSequentialFitter->vertex(tracks);
  }
  
  inline CachingVertex<5> 
  vertex(const std::vector<RefCountedVertexTrack> & tracks,
      const reco::BeamSpot & spot ) const override
  {
    return theSequentialFitter->vertex(tracks, spot );
  }


  /** Fit vertex out of a set of RecTracks. 
   *  Uses the specified linearization point.
   */
  inline CachingVertex<5> 
    vertex(const std::vector<reco::TransientTrack>  & tracks, 
	   const GlobalPoint& linPoint) const override
  {
    return theSequentialFitter->vertex(tracks, linPoint);
  }

  /** Fit vertex out of a set of RecTracks. 
   *  Uses the specified point as both the linearization point AND as prior
   *  estimate of the vertex position. The error is used for the 
   *  weight of the prior estimate.
   */
  inline CachingVertex<5> 
  vertex(const std::vector<reco::TransientTrack> & tracks, 
	 const GlobalPoint& priorPos,
  	 const GlobalError& priorError) const override
  {
    return theSequentialFitter->vertex(tracks, priorPos, priorError);
  }

  /** Fit vertex out of a set of TransientTracks. 
   *  The specified BeamSpot will be used as priot, but NOT for the linearization.
   * The specified LinearizationPointFinder will be used to find the linearization point.
   */
  inline CachingVertex<5> 
  vertex(const std::vector<reco::TransientTrack> & tracks, const reco::BeamSpot& beamSpot) const override
  {
    return theSequentialFitter->vertex(tracks, beamSpot);
  }



  /** Fit vertex out of a set of VertexTracks.
   *  Uses the specified point and error as the prior estimate of the vertex.
   *  This position is not used to relinearize the tracks.
   */
  inline CachingVertex<5> 
  vertex(const std::vector<RefCountedVertexTrack> & tracks, 
	 const GlobalPoint& priorPos,
	 const GlobalError& priorError) const override
  {
    return theSequentialFitter->vertex(tracks, priorPos, priorError);
  }
  
  /** Default convergence criteria
   */
  //  edm::ParameterSet defaultParameters() const;

private:

  void setup(const edm::ParameterSet& pSet,  bool useSmoothing );

  edm::ParameterSet defaultParameters() const ;

  const SequentialVertexFitter<5> * theSequentialFitter;
};

#endif

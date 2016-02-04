#ifndef AdaptiveGsfVertexFitter_H
#define AdaptiveGsfVertexFitter_H

#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"
#include "RecoVertex/LinearizationPointFinders/interface/DefaultLinearizationPointFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"

/** 
 * Sequential vertex fitter, to be used with the Gaussian Sum Vertex Filter
 * After the vertes fit, the tracks can be refit with the additional 
 * constraint of the vertex position.
 */


class AdaptiveGsfVertexFitter : public VertexFitter<5> {

public:

  typedef CachingVertex<5>::RefCountedVertexTrack RefCountedVertexTrack;

  /** Default constructor, using the given linearization point finder.
   *  \param linP	The LinearizationPointFinder to use
   *  \param useSmoothing  Specifies whether the tracks sould be refit.
   */ 
 
  AdaptiveGsfVertexFitter(const edm::ParameterSet& pSet,
	const LinearizationPointFinder & linP = DefaultLinearizationPointFinder());

  virtual ~AdaptiveGsfVertexFitter();

  /**
   * Copy constructor
   */

  AdaptiveGsfVertexFitter(const AdaptiveGsfVertexFitter & original);

  AdaptiveGsfVertexFitter * clone() const {
    return new AdaptiveGsfVertexFitter(* this);
  }

public:

  /** Fit vertex out of a set of RecTracks
   */
  virtual inline CachingVertex<5> 
    vertex(const std::vector<reco::TransientTrack>  & tracks) const
  {
    return theFitter->vertex(tracks);
  }

  /** Fit vertex out of a set of VertexTracks
   */
  virtual inline CachingVertex<5> 
  vertex(const std::vector<RefCountedVertexTrack> & tracks) const
  {
    return theFitter->vertex(tracks);
  }

  /** Fit vertex out of a set of RecTracks. 
   *  Uses the specified linearization point.
   */
  virtual inline CachingVertex<5> 
    vertex(const std::vector<reco::TransientTrack>  & tracks, 
	   const GlobalPoint& linPoint) const
  {
    return theFitter->vertex(tracks, linPoint);
  }

  /** Fit vertex out of a set of TransientTracks. 
   *  The specified BeamSpot will be used as priot, but NOT for the linearization.
   * The specified LinearizationPointFinder will be used to find the linearization point.
   */
  virtual inline CachingVertex<5> 
  vertex(const std::vector<reco::TransientTrack> & tracks, const reco::BeamSpot& beamSpot) const
  {
    return theFitter->vertex(tracks, beamSpot);
  }


  /** Fit vertex out of a set of RecTracks. 
   *  Uses the specified point as both the linearization point AND as prior
   *  estimate of the vertex position. The error is used for the 
   *  weight of the prior estimate.
   */
  virtual inline CachingVertex<5> 
  vertex(const std::vector<reco::TransientTrack> & tracks, 
	 const GlobalPoint& priorPos,
  	 const GlobalError& priorError) const
  {
    return theFitter->vertex(tracks, priorPos, priorError);
  }

  virtual inline CachingVertex<5> 
  vertex(const std::vector<RefCountedVertexTrack> & tracks,
      const reco::BeamSpot & spot ) const
  {
    return theFitter->vertex(tracks, spot );
  }

  /** Fit vertex out of a set of VertexTracks.
   *  Uses the specified point and error as the prior estimate of the vertex.
   *  This position is not used to relinearize the tracks.
   */
  virtual inline CachingVertex<5> 
  vertex(const std::vector<RefCountedVertexTrack> & tracks, 
	 const GlobalPoint& priorPos,
	 const GlobalError& priorError) const
  {
    return theFitter->vertex(tracks, priorPos, priorError);
  }

private:
  
  AdaptiveVertexFitter  * theFitter;
};

#endif

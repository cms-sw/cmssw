#ifndef GsfVertexFitter_H
#define GsfVertexFitter_H

#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"
// #include "RecoVertex/VertexPrimitives/interface/VertexSmoother.h"
// #include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexUpdator.h"
// #include "RecoVertex/VertexTools/interface/VertexTrackFactory.h"
#include "RecoVertex/LinearizationPointFinders/interface/DefaultLinearizationPointFinder.h"
// #include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexMerger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
// #include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"
#include "RecoVertex/VertexTools/interface/SequentialVertexFitter.h"

/** 
 * Sequential vertex fitter, to be used with the Gaussian Sum Vertex Filter
 * After the vertes fit, the tracks can be refit with the additional 
 * constraint of the vertex position.
 */


class GsfVertexFitter : public VertexFitter {

public:

  /** Default constructor, using the given linearization point finder.
   *  \param linP	The LinearizationPointFinder to use
   *  \param useSmoothing  Specifies whether the tracks sould be refit.
   */ 
 
  GsfVertexFitter(const edm::ParameterSet& pSet,
	const LinearizationPointFinder & linP = DefaultLinearizationPointFinder());

  virtual ~GsfVertexFitter();

  /**
   * Copy constructor
   */

  GsfVertexFitter(const GsfVertexFitter & original);

  GsfVertexFitter * clone() const {
    return new GsfVertexFitter(* this);
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

  /** Fit vertex out of a set of TransientTracks. 
   *  The specified BeamSpot will be used as priot, but NOT for the linearization.
   * The specified LinearizationPointFinder will be used to find the linearization point.
   */
  virtual inline CachingVertex 
  vertex(const vector<reco::TransientTrack> & tracks, const reco::BeamSpot& beamSpot) const
  {
    return theSequentialFitter->vertex(tracks, beamSpot);
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
  
  SequentialVertexFitter * theSequentialFitter;
};

#endif

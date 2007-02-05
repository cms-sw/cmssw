#ifndef GsfVertexFitter_H
#define GsfVertexFitter_H

#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"
#include "RecoVertex/VertexTools/interface/LinearizationPointFinder.h"
#include "RecoVertex/VertexPrimitives/interface/VertexSmoother.h"
#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexUpdator.h"
#include "RecoVertex/GaussianSumVertexFit/interface/MultiPerigeeLTSFactory.h"
#include "RecoVertex/VertexTools/interface/VertexTrackFactory.h"
#include "RecoVertex/LinearizationPointFinders/interface/DefaultLinearizationPointFinder.h"
#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexMerger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CommonDetAlgo/interface/DeepCopyPointerByClone.h"

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
	const LinearizationPointFinder & linP = DefaultLinearizationPointFinder(), 
	bool useSmoothing = false);

  virtual ~GsfVertexFitter();

  /**
   * Copy constructor
   */

  GsfVertexFitter(const GsfVertexFitter & original);


  /**
   *  Method to set the convergence criterion 
   *  (the maximum distance between the vertex computed in the previous
   *   and the current iterations to consider the fit to have converged)
   */

  void setMaximumDistance(float maxShift) {theMaxShift = maxShift;}


  /**
   *   Method to set the maximum number of iterations to perform
   */

  void setMaximumNumberOfIterations(int maxIterations)
  	{theMaxStep = maxIterations;}

 /**
  * Method returning the fitted vertex, from a container of reco::TransientTracks.
  * The linearization point will be searched with the given LP finder.
  * No prior vertex position will be used in the vertex fit.
  * \param tracks The container of reco::TransientTracks to fit.
  * \return The fitted vertex
  */
  virtual CachingVertex vertex(const std::vector<reco::TransientTrack> & tracks) const;

 /**
  * Method returning the fitted vertex, from a container of VertexTracks.
  * For the first loop, the LinearizedTrackState contained in the VertexTracks
  * will be used. If subsequent loops are needed, the new VertexTracks will
  * be created with the last estimate of the vertex as linearization point.
  * No prior vertex position will be used in the vertex fit.
  * \param tracks The container of VertexTracks to fit.
  * \return The fitted vertex
  */
  virtual CachingVertex vertex(const std::vector<RefCountedVertexTrack> & tracks) const;


  /** Fit vertex out of a set of reco::TransientTracks. Uses the specified linearization point.
   */
  virtual CachingVertex  vertex(const std::vector<reco::TransientTrack> & tracks, 
  		const GlobalPoint& linPoint) const;

  /** Fit vertex out of a set of reco::TransientTracks. 
   *   Uses the position as both the linearization point AND as prior
   *   estimate of the vertex position. The error is used for the 
   *   weight of the prior estimate.
   */
  virtual CachingVertex vertex(const std::vector<reco::TransientTrack> & tracks, 
  		const GlobalPoint& priorPos,
  		const GlobalError& priorError) const;

  /** Fit vertex out of a set of VertexTracks
   *   Uses the position and error for the prior estimate of the vertex.
   *   This position is not used to relinearize the tracks.
   */
  virtual CachingVertex vertex(const std::vector<RefCountedVertexTrack> & tracks, 
  		const GlobalPoint& priorPos,
  		const GlobalError& priorError) const;


  /**
   * Access methods
   */
  const LinearizationPointFinder * linearizationPointFinder() const
  {return theLinP;}

  const VertexUpdator * vertexUpdator() const
  {return &theUpdator;}

  const VertexSmoother * vertexSmoother() const
  {return theSmoother;}

  const float maxShift() const
  {return theMaxShift;}

  const int maxStep() const
  {return theMaxStep;}

  const int limitComponents() const {return limitComponents_;}

  GsfVertexFitter * clone() const {
    return new GsfVertexFitter(* this);
  }

protected:

  /**
   * Construct a container of VertexTrack from a set of reco::TransientTracks.
   * \param tracks The container of reco::TransientTracks.
   * \param seed The seed to use for the VertexTracks. This position will 
   *	also be used as the new linearization point.
   * \return The container of VertexTracks which are to be used in the next fit.
   */
  virtual std::vector<RefCountedVertexTrack> linearizeTracks(
  	const std::vector<reco::TransientTrack> & tracks, const VertexState state) const;

  /**
   * Construct new a container of VertexTrack with a new linearization point
   * and vertex seed, from an existing set of VertexTrack, from which only the 
   * reco::TransientTracks will be used.
   * \param tracks The original container of VertexTracks, from which the reco::TransientTracks 
   * 	will be extracted.
   * \param seed The seed to use for the VertexTracks. This position will 
   *	also be used as the new linearization point.
   * \return The container of VertexTracks which are to be used in the next fit.
   */
  virtual std::vector<RefCountedVertexTrack> reLinearizeTracks(
	const std::vector<RefCountedVertexTrack> & tracks, const VertexState state) const;

  /**
   * The methode where the vrte fit is actually done. The seed is used as the
   * prior estimate in the vertex fit (in case its error is large, it will have
   * little influence on the fit.
   * The tracks will be relinearized in case further loops are needed.
   *   \parameter tracks The tracks to use in the fit.
   *   \paraemter priorSeed The prior estimate of the vertex
   *   \return The fitted vertex
   */
  CachingVertex fit(const std::vector<RefCountedVertexTrack> & tracks,
  	const VertexState priorVertex, bool withPrior) const;


private:
  
  /**
   *   Reads the configurable parameters.
   */

//   void readParameters();


  float theMaxShift;
  int theMaxStep;
  bool limitComponents_;

  LinearizationPointFinder*  theLinP;
  GsfVertexUpdator  theUpdator;
  VertexSmoother * theSmoother;

  MultiPerigeeLTSFactory lTrackFactory;
  VertexTrackFactory vTrackFactory;
  DeepCopyPointerByClone<GsfVertexMerger> theMerger;
};

#endif

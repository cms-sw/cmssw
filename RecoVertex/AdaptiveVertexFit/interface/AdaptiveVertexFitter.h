#ifndef AdaptiveVertexFitter_H
#define AdaptiveVertexFitter_H

#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"
#include "RecoVertex/LinearizationPointFinders/interface/DefaultLinearizationPointFinder.h"
#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/VertexTools/interface/DummyVertexSmoother.h"
// #include "RecoVertex/AdaptiveVertexFit/interface/KalmanVertexSmoother.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexTrackCompatibilityEstimator.h"
#include "RecoVertex/VertexTools/interface/LinearizedTrackStateFactory.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
 * \class AdaptiveVertexFitter
 * An iterative reweighted fitter.
 * Very robust, very adaptive.
 *
 * See CMS Note 2007/008.
 *
 * Exceptions
 * VertexException( "Supplied fewer than two tracks" )
 * VertexException( "fewer than 2 significant tracks (w>threshold)" )
 *
 */

class AdaptiveVertexFitter : public VertexFitter<5> {

public:

  typedef ReferenceCountingPointer<VertexTrack<5> > RefCountedVertexTrack;
  typedef ReferenceCountingPointer<LinearizedTrackState<5> > RefCountedLinearizedTrackState;

  /**
   *   Reimplemented constructors to use any kind of
   *   linearisation point finder, vertex updator and smoother.
   *   If no smoother is to be used, do not specify an instance for it.
   */
  AdaptiveVertexFitter(
      const AnnealingSchedule & ann = GeometricAnnealing(),
      const LinearizationPointFinder & linP =
             DefaultLinearizationPointFinder(),
      const VertexUpdator<5> & updator = KalmanVertexUpdator<5>(),
      const VertexTrackCompatibilityEstimator<5> & estor =
             KalmanVertexTrackCompatibilityEstimator<5>(),
      const VertexSmoother<5> & smoother = DummyVertexSmoother<5>(),
      const AbstractLTSFactory<5> & ltsf = LinearizedTrackStateFactory() );

  AdaptiveVertexFitter( const AdaptiveVertexFitter & original );

  virtual ~AdaptiveVertexFitter();

 /**
  * Method returning the fitted vertex, from a container of reco::TransientTracks.
  * The linearization point will be searched with the given LP finder.
  * No prior vertex position will be used in the vertex fit.
  * \return The fitted vertex
  */
  virtual CachingVertex<5> vertex( const std::vector<reco::TransientTrack> & ) const;

 /**
  * Method returning the fitted vertex, from a container of VertexTracks.
  * For the first loop, the LinearizedTrack contained in the VertexTracks
  * will be used. If subsequent loops are needed, the new VertexTracks will
  * be created with the last estimate of the vertex as linearization point.
  * No prior vertex position will be used in the vertex fit.
  * \return The fitted vertex
  */
  virtual CachingVertex<5> vertex(const std::vector<RefCountedVertexTrack> & ) const;

  /**
   *  Same as above, only now also with BeamSpot constraint.
   */
  virtual CachingVertex<5> vertex(const std::vector<RefCountedVertexTrack> &,
     const reco::BeamSpot & spot ) const;

  /** Fit vertex out of a std::vector of reco::TransientTracks. Uses the specified
   * linearization point.
   */
  virtual CachingVertex<5> vertex( const std::vector<reco::TransientTrack> &,
                                const GlobalPoint& linPoint ) const;

  /** Fit vertex out of a set of reco::TransientTracks.
   *   Uses the position as both the linearization point AND as prior
   *   estimate of the vertex position. The error is used for the
   *   weight of the prior estimate.
   */
  virtual CachingVertex<5> vertex( const std::vector<reco::TransientTrack> &,
                                const GlobalPoint & priorPos,
                                const GlobalError & priorError ) const;

  /** Fit vertex out of a set of TransientTracks. 
   *  The specified BeamSpot will be used as priot, but NOT for the linearization.
   * The specified LinearizationPointFinder will be used to find the linearization point.
   */
  virtual CachingVertex<5> vertex(const std::vector<reco::TransientTrack> & tracks,
		const reco::BeamSpot& beamSpot) const;


  /**  Fit vertex out of a set of VertexTracks
   *   Uses the position and error for the prior estimate of the vertex.
   *   This position is not used to relinearize the tracks.
   */
  virtual CachingVertex<5> vertex( const std::vector<RefCountedVertexTrack> &,
                                const GlobalPoint & priorPos,
                                const GlobalError & priorError ) const;

  AdaptiveVertexFitter * clone() const;

  /**
   *  Set the weight threshold
   *  should be used only to find (once)
   *  a good value
   *  FIXME this should disappear in the final version
   */
  void setWeightThreshold ( float w );

  /**
   *   Reads the configurable parameters.
   *   \param maxshift if the vertex moves further than this (in cm),
   *   then we re-iterate.
   *   \param maxlpshift if the vertex moves further than this,
   *   then we re-linearize the tracks.
   *   \param maxstep that's the maximum of iterations that we allow for.
   *   \param weightthreshold that's the minimum track weight
   *   for a track to be considered "significant".
   *   If fewer than two tracks are significant, an exception is thrown.
   */
  void setParameters( double maxshift=0.0001, double maxlpshift=0.1, 
                      unsigned maxstep=30, double weightthreshold=.001 );

  /**
   *  Sets parameters.
   *  The following parameters are expected:
   *  maxshift,  maxlpshift,  maxstep,  weightthreshold
   */
  void setParameters ( const edm::ParameterSet & );

  void gsfIntermediarySmoothing(bool sm) { gsfIntermediarySmoothing_ = sm; }

  bool gsfIntermediarySmoothing() const { return gsfIntermediarySmoothing_;}

private:
  /**
   * Construct new a container of VertexTrack with a new linearization point
   * and vertex seed, from an existing set of VertexTrack, from which only the
   * recTracks will be used.
   * \param tracks The original container of VertexTracks, from which the reco::TransientTracks
   *     will be extracted.
   * \param vertex The seed to use for the VertexTracks. This position will
   *    also be used as the new linearization point.
   * \return The container of VertexTracks which are to be used in the next fit.
   */
  std::vector<RefCountedVertexTrack> reLinearizeTracks(
                const std::vector<RefCountedVertexTrack> & tracks,
                const CachingVertex<5> & vertex ) const;


  /**
   * Construct a new container of VertexTracks with new weights 
   * accounting for vertex error, from an existing set of LinearizedTracks. 
   */
  std::vector<RefCountedVertexTrack> reWeightTracks(
                        const std::vector<RefCountedLinearizedTrackState> &,
                        const CachingVertex<5> & seed ) const;

  /**
   * Construct new a container of VertexTracks with new weights 
   * accounting for vertex error, from an existing set of VertexTracks. 
   * From these the LinearizedTracks will be reused.
   */
  std::vector<RefCountedVertexTrack> reWeightTracks(
                        const std::vector<RefCountedVertexTrack> &,
                        const CachingVertex<5> & seed) const;


  /**
   *  Weight the tracks, for the first time, using
   *  KalmanChiSquare.
   */
  
  std::vector<RefCountedVertexTrack> weightTracks(
                        const std::vector<RefCountedLinearizedTrackState> &,
                        const VertexState & seed ) const;

  /**
   *  Linearize tracks, for the first time in the iteration.
   */
  std::vector<RefCountedVertexTrack> linearizeTracks(
                        const std::vector<reco::TransientTrack> &,
                        const VertexState & ) const;
  /**
   *  perform the fit
   */
  CachingVertex<5> fit( const std::vector<RefCountedVertexTrack> & tracks,
                     const VertexState & priorSeed,
                     bool withPrior) const;

  double getWeight ( float chi2 ) const;
private:
  double theMaxShift;
  double theMaxLPShift;
  int theMaxStep;
  double theWeightThreshold;
  mutable int theNr;

  LinearizationPointFinder * theLinP;
  VertexUpdator<5> * theUpdator;
  VertexSmoother<5> * theSmoother;
  AnnealingSchedule * theAssProbComputer;
  VertexTrackCompatibilityEstimator<5> * theComp;
  const AbstractLTSFactory<5> * theLinTrkFactory;
  bool gsfIntermediarySmoothing_;
  mutable int mctr_;
};

#endif

#ifndef _SequentialVertexSmoother_H_
#define _SequentialVertexSmoother_H_


#include "RecoVertex/VertexPrimitives/interface/VertexSmoother.h"
#include "RecoVertex/VertexPrimitives/interface/VertexTrackUpdator.h"
#include "RecoVertex/VertexPrimitives/interface/VertexSmoothedChiSquaredEstimator.h"
#include "RecoVertex/VertexPrimitives/interface/TrackToTrackCovCalculator.h"

/**
 *  The class which handles the track-refit, smoothed chi**2 and track-to-track 
 *  covariance matrix calculations.
 */

template <unsigned int N>
class SequentialVertexSmoother : public VertexSmoother<N> {

public:

  typedef ReferenceCountingPointer<RefittedTrackState<N> > RefCountedRefittedTrackState;
  typedef ReferenceCountingPointer<VertexTrack<N> > RefCountedVertexTrack;
  typedef ReferenceCountingPointer<LinearizedTrackState<N> > RefCountedLinearizedTrackState;

  /**
   *  The constructor, where the different components to be used are specified.
   *  \param vtu The algorithm to refit the tracks  with the vertex constraint (or smoothing)
   *  \param vse The algorithm to calculate the smoothed chi**2
   *  \return covCalc The algorithm the track-to-track covariance matrix. 
   *  If this option is not required, this pointer should be 0.
   */
  SequentialVertexSmoother(const VertexTrackUpdator<N> & vtu, 
			   const VertexSmoothedChiSquaredEstimator<N> & vse, 
			   const TrackToTrackCovCalculator<N> & covCalc);

  virtual ~SequentialVertexSmoother();

  /**
   *  Special copy constructor cloning the private data
   */
  SequentialVertexSmoother(const SequentialVertexSmoother<N> & smoother);

  /**
   *  Methode which will refit the tracks with the vertex constraint, 
   *  calculate the smoothed vertex chi**2, and, if required, the 
   *  track-to-track covariance matrix.
   *  \param initVertex is the initial guess of the vertex (a-priori 
   *	information), used at the start of the fit.
   *  \param newVertex is the final estimate of the vertex, as given by the 
   *	last update.
   *  \return the final vertex estimate, with all the supplementary information
   */
  virtual CachingVertex<N> smooth(const CachingVertex<N> & vertex) const;

  /**
   *  Access methods
   */
  const VertexTrackUpdator<N> * vertexTrackUpdator() const
    { return theVertexTrackUpdator; }
  const VertexSmoothedChiSquaredEstimator<N> * vertexSmoothedChiSquaredEstimator() const
    { return theVertexSmoothedChiSquaredEstimator; }
  const TrackToTrackCovCalculator<N> * trackToTrackCovCalculator() const
    { return theTrackToTrackCovCalculator; }

  /**
   * Clone method 
   */
  virtual SequentialVertexSmoother<N> * clone() const 
  {
    return new SequentialVertexSmoother(* this);
  }
  
private:
   VertexTrackUpdator<N> * theVertexTrackUpdator;       
   VertexSmoothedChiSquaredEstimator<N> * theVertexSmoothedChiSquaredEstimator;
   TrackToTrackCovCalculator<N> * theTrackToTrackCovCalculator;
};

#endif

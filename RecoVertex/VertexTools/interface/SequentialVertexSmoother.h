#ifndef _SequentialVertexSmoother_H_
#define _SequentialVertexSmoother_H_


#include "RecoVertex/VertexPrimitives/interface/VertexSmoother.h"
#include "RecoVertex/VertexPrimitives/interface/VertexTrackUpdator.h"
#include "RecoVertex/VertexPrimitives/interface/VertexSmoothedChiSquaredEstimator.h"
#include "RecoVertex/VertexPrimitives/interface/TrackToTrackCovCalculator.h"

#include "RecoVertex/VertexPrimitives/interface/TrackMap.h"
#include "RecoVertex/VertexPrimitives/interface/TrackToTrackMap.h"


class CachingVertex;

/**
 *  The class which handles the track-refit, smoothed chi**2 and track-to-track 
 *  covariance matrix calculations.
 */

class SequentialVertexSmoother : public VertexSmoother {

public:

  /**
   *  The constructor, where the different components to be used are specified.
   *  \param vtu The algorithm to refit the tracks  with the vertex constraint (or smoothing)
   *  \param vse The algorithm to calculate the smoothed chi**2
   *  \return covCalc The algorithm the track-to-track covariance matrix. 
   *  If this option is not required, this pointer should be 0.
   */
  SequentialVertexSmoother(const VertexTrackUpdator & vtu, 
			   const VertexSmoothedChiSquaredEstimator & vse, 
			   const TrackToTrackCovCalculator & covCalc);

  virtual ~SequentialVertexSmoother();

  /**
   *  Special copy constructor cloning the private data
   */
  SequentialVertexSmoother(const SequentialVertexSmoother & smoother);

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
  virtual CachingVertex smooth(const CachingVertex & vertex) const;

  /**
   *  Access methods
   */
  const VertexTrackUpdator * vertexTrackUpdator() const
    { return theVertexTrackUpdator; }
  const VertexSmoothedChiSquaredEstimator * vertexSmoothedChiSquaredEstimator() const
    { return theVertexSmoothedChiSquaredEstimator; }
  const TrackToTrackCovCalculator * trackToTrackCovCalculator() const
    { return theTrackToTrackCovCalculator; }

  /**
   * Clone method 
   */
  virtual SequentialVertexSmoother * clone() const 
  {
    return new SequentialVertexSmoother(* this);
  }
  
private:
   VertexTrackUpdator * theVertexTrackUpdator;       
   VertexSmoothedChiSquaredEstimator * theVertexSmoothedChiSquaredEstimator;
   TrackToTrackCovCalculator * theTrackToTrackCovCalculator;
};

#endif

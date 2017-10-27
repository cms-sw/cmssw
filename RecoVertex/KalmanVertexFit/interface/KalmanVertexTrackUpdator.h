#ifndef KalmanVertexTrackUpdator_H
#define KalmanVertexTrackUpdator_H

#include "RecoVertex/VertexPrimitives/interface/VertexTrackUpdator.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "RecoVertex/VertexPrimitives/interface/VertexTrack.h"
#include "RecoVertex/VertexTools/interface/VertexTrackFactory.h"
#include "RecoVertex/KalmanVertexFit/interface/KVFHelper.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"

/** \class KalmanVertexTrackUpdator
 *  Performs the refit of the tracks with the vertex constraint, 
 *  using the Kalman filter algorithms.
 */

template <unsigned int N>
class KalmanVertexTrackUpdator : public VertexTrackUpdator<N> {

public:

  typedef typename CachingVertex<N>::RefCountedVertexTrack RefCountedVertexTrack;
  typedef typename VertexTrack<N>::RefCountedLinearizedTrackState RefCountedLinearizedTrackState;
  typedef typename VertexTrack<N>::RefCountedRefittedTrackState RefCountedRefittedTrackState;


  /**
   *  Default constructor
   */

  KalmanVertexTrackUpdator(){}

  ~KalmanVertexTrackUpdator() override{}

  /**
   *   Refit of the track with the vertex constraint.
   *   \param vertex The vertex which has to be used as constraint.
   *   \param track  The track to refit.
   *   \return	The VertexTrack containing the refitted track and 
   *		the track-to-vertex covariance.
   */

  RefCountedVertexTrack update(const CachingVertex<N> & vertex,
                               RefCountedVertexTrack track) const override;


  /**
   *  Clone method
   */

  KalmanVertexTrackUpdator<N> * clone() const override
  {
    return new KalmanVertexTrackUpdator(*this);
  }

  typedef ROOT::Math::SMatrix<double,3,N-2,ROOT::Math::MatRepStd<double,3,N-2> > AlgebraicMatrix3M;
  typedef ROOT::Math::SMatrix<double,N+1,N+1,ROOT::Math::MatRepSym<double,N+1> > AlgebraicSymMatrixOO;
  typedef std::pair< RefCountedRefittedTrackState, AlgebraicSymMatrixOO > trackMatrixPair; 

  /**
   *   Refit of the track with the vertex constraint.
   *   \param vertex The vertex which has to be used as constraint.
   *   \param track  The track to refit.
   *   \return	The refitted state with the track-to-vertex covariance.
   */

  trackMatrixPair trackRefit(const VertexState & vertex,
		RefCountedLinearizedTrackState linTrackState,
		float weight) const;

private:

  VertexTrackFactory<N> theVTFactory;
  KVFHelper<N> helper;
  KalmanVertexUpdator<N> updator;
};

#endif

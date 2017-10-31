#ifndef KalmanVertexUpdator_H
#define KalmanVertexUpdator_H

#include "RecoVertex/VertexPrimitives/interface/VertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KVFHelper.h"

/**
 *  Vertex updator for the Kalman vertex filter.
 *  (c.f. R. Fruewirth et.al., Comp.Phys.Comm 96 (1996) 189
 */

template <unsigned int N>
class KalmanVertexUpdator: public VertexUpdator<N> {

public:

  typedef typename CachingVertex<N>::RefCountedVertexTrack RefCountedVertexTrack;
  typedef typename VertexTrack<N>::RefCountedLinearizedTrackState RefCountedLinearizedTrackState;

/**
 *  Method to add a track to an existing CachingVertex
 * An invalid vertex is returned in case of problems during the update.
 */

   CachingVertex<N> add(const CachingVertex<N> & oldVertex,
        const RefCountedVertexTrack track) const override;

/**
 *  Method removing already used VertexTrack from existing CachingVertex
 * An invalid vertex is returned in case of problems during the update.
 */

   CachingVertex<N> remove(const CachingVertex<N> & oldVertex,
        const RefCountedVertexTrack track) const override;

/**
 * Clone method
 */

   VertexUpdator<N> * clone() const override
   {
    return new KalmanVertexUpdator(* this);
   }

    /**
     * The methode which actually does the vertex update.
     * An invalid vertex is returned in case of problems during the update.
     */
  CachingVertex<N> update(const CachingVertex<N> & oldVertex,
                         const RefCountedVertexTrack track, float weight,
                         int sign ) const;

  VertexState positionUpdate (const VertexState & oldVertex,
	 const RefCountedLinearizedTrackState linearizedTrack, 
	 const float weight, int sign) const;

  std::pair <bool, double> chi2Increment(const VertexState & oldVertex, 
	 const VertexState & newVertexState,
	 const RefCountedLinearizedTrackState linearizedTrack, 
	 float weight) const; 

private:

  typedef ROOT::Math::SVector<double,N> AlgebraicVectorN;
  typedef ROOT::Math::SVector<double,N-2> AlgebraicVectorM;
  typedef ROOT::Math::SMatrix<double,N,3,ROOT::Math::MatRepStd<double,N,3> > AlgebraicMatrixN3;
  typedef ROOT::Math::SMatrix<double,N,N-2,ROOT::Math::MatRepStd<double,N,N-2> > AlgebraicMatrixNM;
  typedef ROOT::Math::SMatrix<double,N-2,3,ROOT::Math::MatRepStd<double,N-2,3> > AlgebraicMatrixM3;
  typedef ROOT::Math::SMatrix<double,N,N,ROOT::Math::MatRepSym<double,N> > AlgebraicSymMatrixNN;
  typedef ROOT::Math::SMatrix<double,N+1,N+1,ROOT::Math::MatRepSym<double,N+1> > AlgebraicSymMatrixOO;
  typedef ROOT::Math::SMatrix<double,N+1,N+1,ROOT::Math::MatRepStd<double,N+1,N+1> > AlgebraicMatrixOO;
  typedef ROOT::Math::SMatrix<double,N-2,N-2,ROOT::Math::MatRepSym<double,N-2> > AlgebraicSymMatrixMM;

  KVFHelper<N> helper;

};

#endif

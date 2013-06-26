#include "RecoVertex/VertexPrimitives/interface/VertexTrack.h"


template <unsigned int N>
VertexTrack<N>::VertexTrack(const RefCountedLinearizedTrackState lt, 
			 const VertexState v, 
			 float weight) 
  : theLinTrack(lt), theVertexState(v), theWeight(weight),
    stAvailable(false), covAvailable(false), smoothedChi2_(-1.) {}


template <unsigned int N>
VertexTrack<N>::VertexTrack(const RefCountedLinearizedTrackState lt, 
			 const VertexState v, float weight,
			 const RefCountedRefittedTrackState & refittedState,
			 float smoothedChi2)
  : theLinTrack(lt), theVertexState(v), theWeight(weight),
    stAvailable(true), covAvailable(false), theRefittedState(refittedState),
    smoothedChi2_(smoothedChi2) {}


template <unsigned int N>
VertexTrack<N>::VertexTrack(const RefCountedLinearizedTrackState lt, 
			 const VertexState v, float weight, 
			 const RefCountedRefittedTrackState & refittedState,
			 float smoothedChi2, const AlgebraicSymMatrixOO & fullCov) 
  : theLinTrack(lt), theVertexState(v), theWeight(weight),
    stAvailable(true), covAvailable(true), 
    theRefittedState(refittedState), fullCovariance_(fullCov),
    smoothedChi2_(smoothedChi2) {}


template <unsigned int N>
typename VertexTrack<N>::AlgebraicVectorN VertexTrack<N>::refittedParamFromEquation() const 
{
  return linearizedTrack()->refittedParamFromEquation(theRefittedState);
}

template class VertexTrack<5>;
template class VertexTrack<6>;

//   /** Track to vertex covariance 
//    */   
// template <unsigned int N>
// typename VertexTrack<N>::AlgebraicMatrix3M VertexTrack<N>::tkToVtxCovariance() const {
//     if (!tkToVertexCovarianceAvailable()) {
//       throw VertexException("VertexTrack::track to vertex covariance not available"); 
//     }
// //     if (N==5) {
// //       ROOT::Math::SMatrix<double,6,6,ROOT::Math::MatRepSym<double,6> > b6 = fullCovariance_;
// //       return b.Sub<AlgebraicMatrix3M>(3,0);
// //       //a = b.Sub< ROOT::Math::SMatrix<double,3,N-2,ROOT::Math::MatRepStd<double,3,N-2> > >(3,0);
// //     }
//     ROOT::Math::SMatrix<double,3,N-2,ROOT::Math::MatRepStd<double,3,N-2> > a;
//     return a;
//   }

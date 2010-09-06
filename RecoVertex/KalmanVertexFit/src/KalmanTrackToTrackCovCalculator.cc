#include "RecoVertex/KalmanVertexFit/interface/KalmanTrackToTrackCovCalculator.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

template <unsigned int N>
typename CachingVertex<N>::TrackToTrackMap
KalmanTrackToTrackCovCalculator<N>::operator() 
	(const CachingVertex<N> & vertex) const
{
  typedef ROOT::Math::SMatrix<double,N,3,ROOT::Math::MatRepStd<double,N,3> > AlgebraicMatrixN3;
  typedef ROOT::Math::SMatrix<double,N,N-2,ROOT::Math::MatRepStd<double,N,N-2> > AlgebraicMatrixNM;
  typedef ROOT::Math::SMatrix<double,N-2,3,ROOT::Math::MatRepStd<double,N-2,3> > AlgebraicMatrixM3;
  typedef ROOT::Math::SMatrix<double,3,N-2,ROOT::Math::MatRepStd<double,3,N-2> > AlgebraicMatrix3M;
  typedef ROOT::Math::SMatrix<double,N,N,ROOT::Math::MatRepSym<double,N> > AlgebraicSymMatrixNN;
  typedef ROOT::Math::SMatrix<double,N-2,N-2,ROOT::Math::MatRepSym<double,N-2> > AlgebraicSymMatrixMM;
  typedef ROOT::Math::SMatrix<double,N-2,N-2,ROOT::Math::MatRepStd<double,N-2,N-2> > AlgebraicMatrixMM;

  typename CachingVertex<N>::TrackToTrackMap returnMap;
  int ifail = 0;
  std::vector<RefCountedVertexTrack> tracks = vertex.tracks();

//vertex initial data needed
  AlgebraicSymMatrix33 vertexC = vertex.error().matrix_new();

  for(typename std::vector<RefCountedVertexTrack>::iterator i = tracks.begin(); 
  	i != tracks.end(); i++)
  {        
    const AlgebraicMatrixN3 & leftA = (*i)->linearizedTrack()->positionJacobian();
    const AlgebraicMatrixNM & leftB = (*i)->linearizedTrack()->momentumJacobian();
    AlgebraicSymMatrixNN leftG = (*i)->linearizedTrack()->predictedStateWeight(ifail);
    AlgebraicSymMatrixMM leftW = ROOT::Math::SimilarityT(leftB,leftG);

    ifail = ! leftW.Invert();
    if(ifail != 0) throw VertexException
    	("KalmanTrackToTrackCovarianceCalculator::leftW matrix inversion failed");
    AlgebraicMatrixM3 leftPart = leftW * (ROOT::Math::Transpose(leftB)) * leftG * leftA;
    typename CachingVertex<N>::TrackMap internalMap;
    for(typename std::vector<RefCountedVertexTrack>::iterator j = tracks.begin(); j != tracks.end(); j++)
    {

      if(*i < *j)
      {

	const AlgebraicMatrixN3 & rightA = (*j)->linearizedTrack()->positionJacobian();
        const AlgebraicMatrixNM & rightB = (*j)->linearizedTrack()->momentumJacobian();
        AlgebraicSymMatrixNN rightG = (*j)->linearizedTrack()->predictedStateWeight(ifail);
        AlgebraicSymMatrixMM rightW = ROOT::Math::SimilarityT(rightB,rightG);

        ifail = ! rightW.Invert();

        if(ifail != 0) throw VertexException
	  ("KalmanTrackToTrackCovarianceCalculator::rightW matrix inversion failed");
        AlgebraicMatrix3M rightPart = (ROOT::Math::Transpose(rightA)) * rightG * rightB * rightW;
	internalMap[(*j)] = leftPart * vertexC * rightPart;
      }       
    }        
    returnMap[*i] = internalMap;
  }
  return returnMap;
}

template class KalmanTrackToTrackCovCalculator<5>;
template class KalmanTrackToTrackCovCalculator<6>;

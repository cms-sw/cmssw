#include "RecoVertex/KalmanVertexFit/interface/KalmanTrackToTrackCovCalculator.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

TrackToTrackMap KalmanTrackToTrackCovCalculator::operator() 
	(const CachingVertex & vertex) const
{
  TrackToTrackMap returnMap;
  int ifail = 0;
  vector<RefCountedVertexTrack> tracks = vertex.tracks();

//vertex initial data needed
  AlgebraicSymMatrix33 vertexC = vertex.error().matrix_new();

  for(vector<RefCountedVertexTrack>::iterator i = tracks.begin(); 
  	i != tracks.end(); i++)
  {        
    AlgebraicMatrix53 leftA = (*i)->linearizedTrack()->positionJacobian();
    AlgebraicMatrix53 leftB = (*i)->linearizedTrack()->momentumJacobian();
    AlgebraicSymMatrix55 leftG = (*i)->linearizedTrack()->predictedStateWeight();
    AlgebraicSymMatrix33 leftW = ROOT::Math::SimilarityT(leftB,leftG);

    ifail = ! leftW.Invert();
    if(ifail != 0) throw VertexException
    	("KalmanTrackToTrackCovarianceCalculator::leftW matrix inversion failed");
    AlgebraicMatrix33 leftPart = leftW * (ROOT::Math::Transpose(leftB)) * leftG * leftA;
    TrackMap internalMap;
    for(vector<RefCountedVertexTrack>::iterator j = tracks.begin(); j != tracks.end(); j++)
    {

      if(*i < *j)
      {

	AlgebraicMatrix53 rightA = (*j)->linearizedTrack()->positionJacobian();
        AlgebraicMatrix53 rightB = (*j)->linearizedTrack()->momentumJacobian();
        AlgebraicSymMatrix55 rightG = (*j)->linearizedTrack()->predictedStateWeight();
        AlgebraicSymMatrix33 rightW = ROOT::Math::SimilarityT(rightB,rightG);

        ifail = ! rightW.Invert();

        if(ifail != 0) throw VertexException
	  ("KalmanTrackToTrackCovarianceCalculator::rightW matrix inversion failed");
        AlgebraicMatrix33 rightPart = (ROOT::Math::Transpose(rightA)) * rightG * rightB * rightW;
	AlgebraicMatrix33 covariance  = leftPart * vertexC * rightPart;
        internalMap[(*j)] = covariance;
      }       
    }        
    returnMap[*i] = internalMap;
  }
  return returnMap;
}

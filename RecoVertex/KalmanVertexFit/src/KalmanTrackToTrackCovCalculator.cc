#include "RecoVertex/KalmanVertexFit/interface/KalmanTrackToTrackCovCalculator.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

TrackToTrackMap KalmanTrackToTrackCovCalculator::operator() 
	(const CachingVertex & vertex) const
{
  TrackToTrackMap returnMap;
  int ifail = 0;
  vector<RefCountedVertexTrack> tracks = vertex.tracks();

//vertex initial data needed
  AlgebraicSymMatrix vertexC = vertex.error().matrix();

  for(vector<RefCountedVertexTrack>::iterator i = tracks.begin(); 
  	i != tracks.end(); i++)
  {        
    AlgebraicMatrix leftA = (*i)->linearizedTrack()->positionJacobian();
    AlgebraicMatrix leftB = (*i)->linearizedTrack()->momentumJacobian();
    AlgebraicSymMatrix leftG = (*i)->linearizedTrack()->predictedStateWeight();
    AlgebraicSymMatrix leftW = leftG.similarityT(leftB);
    leftW.invert(ifail);
    if(ifail != 0) throw VertexException
    	("KalmanTrackToTrackCovarianceCalculator::leftW matrix inversion failed");
    AlgebraicMatrix leftPart = leftW * leftB.T() * leftG * leftA;
    TrackMap internalMap;
    for(vector<RefCountedVertexTrack>::iterator j = tracks.begin(); j != tracks.end(); j++)
    {

      if(*i < *j)
      {

	AlgebraicMatrix rightA = (*j)->linearizedTrack()->positionJacobian();
        AlgebraicMatrix rightB = (*j)->linearizedTrack()->momentumJacobian();
        AlgebraicSymMatrix rightG = (*j)->linearizedTrack()->predictedStateWeight();
	AlgebraicMatrix rightW = rightG.similarityT(rightB);
        rightW.invert(ifail);
        if(ifail != 0) throw VertexException
	  ("KalmanTrackToTrackCovarianceCalculator::rightW matrix inversion failed");
        AlgebraicMatrix rightPart = rightA.T() * rightG * rightB * rightW;
	AlgebraicMatrix covariance  = leftPart * vertexC * rightPart;
        internalMap[(*j)] = covariance;
      }       
    }        
    returnMap[*i] = internalMap;
  }
  return returnMap;
}

#include "RecoVertex/GaussianSumVertexFit/interface/MultiGaussianStateFromVertex.h"
#include "RecoVertex/GaussianSumVertexFit/interface/RCGaussianStateFactory.h"
#include "RecoVertex/GaussianSumVertexFit/interface/BasicMultiVertexState.h"
using namespace std;

MultiGaussianStateFromVertex::MultiGaussianStateFromVertex(
			const VertexState & aState)
  : theState(aState)
{
  RCGaussianStateFactory theFactory;
  vector<VertexState> vsComp = aState.components();
  for (vector<VertexState>::iterator iVS = vsComp.begin(); iVS != vsComp.end();
  	++iVS)
  {
    theComponents.push_back(theFactory.singleGaussianState(*iVS));
  }
}

RCSingleGaussianState
MultiGaussianStateFromVertex::createSingleState (const AlgebraicVector & aMean,
	const AlgebraicSymMatrix & aCovariance, double aWeight) const
{
  return RCSingleGaussianState(
   new SingleGaussianStateFromVertex(
  	VertexState(GlobalPoint(aMean[0], aMean[1], aMean[2]),
	GlobalError(aCovariance), aWeight) ) );
}

ReferenceCountingPointer<MultiGaussianState>
MultiGaussianStateFromVertex::createNewState(
	const vector<RCSingleGaussianState> & stateV) const
{
  vector<VertexState> vertexComponents;
  vertexComponents.reserve(stateV.size());
  for (vector<RCSingleGaussianState>::const_iterator iSGS = stateV.begin();
	iSGS != stateV.end(); ++iSGS)
  {
    vertexComponents.push_back( VertexState(
    	GlobalPoint((**iSGS).mean()[0], (**iSGS).mean()[1], (**iSGS).mean()[2]),
        GlobalError((**iSGS).covariance()), (**iSGS).weight()) );
  }

  VertexState vertexState( new BasicMultiVertexState( vertexComponents));
  return ReferenceCountingPointer<MultiGaussianState>(
      new MultiGaussianStateFromVertex(vertexState) );
}


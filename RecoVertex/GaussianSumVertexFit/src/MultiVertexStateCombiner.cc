#include "RecoVertex/GaussianSumVertexFit/interface/MultiVertexStateCombiner.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"


VertexState
MultiVertexStateCombiner::combine(const VSC & theMixture) const
{

  if (theMixture.empty()) {
    throw VertexException
    ("MultiVertexStateCombiner:: VertexState container to collapse is empty");
  }

  if (theMixture.size()==1) {
//     #ifndef CMS_NO_COMPLEX_RETURNS
      return VertexState(theMixture.front().position(), theMixture.front().error(), 1.0);
//     #else
//       VertexState theFinalVM(theMixture.front().position(), theMixture.front().error(), 1.0);
//       return theFinalVM;
//     #endif
  }


  AlgebraicVector meanPosition(3,0);
  double weightSum = 0.;
  double vtxWeight;
  AlgebraicSymMatrix measCovar1(3,0), measCovar2(3,0);
  for (VSC::const_iterator mixtureIter1 = theMixture.begin();
  	mixtureIter1 != theMixture.end(); mixtureIter1++ ) {
    vtxWeight = mixtureIter1->weightInMixture();

    GlobalPoint vertexPosition = mixtureIter1->position();
    AlgebraicVector vertexCoord1(3);
    vertexCoord1[0] = vertexPosition.x();
    vertexCoord1[1] = vertexPosition.y();
    vertexCoord1[2] = vertexPosition.z();

//    AlgebraicVector position = mixtureIter1->position().vector(); //???
    weightSum += vtxWeight;
    meanPosition += vtxWeight * vertexCoord1;

    measCovar1 += vtxWeight * mixtureIter1->error().matrix();
    for (VSC::const_iterator mixtureIter2 = mixtureIter1+1;
  	mixtureIter2 != theMixture.end(); mixtureIter2++ ) {
      GlobalPoint vertexPosition2 = mixtureIter2->position();
      AlgebraicVector vertexCoord2(3);
      vertexCoord2[0] = vertexPosition2.x();
      vertexCoord2[1] = vertexPosition2.y();
      vertexCoord2[2] = vertexPosition2.z();
      AlgebraicVector posDiff = vertexCoord1 - vertexCoord2;
      AlgebraicSymMatrix s(1,1); //stupid trick to make CLHEP work decently
      measCovar2 +=vtxWeight * mixtureIter2->weightInMixture() * 
      				s.similarity(posDiff.T().T());
    }
  }
  meanPosition /= weightSum;
  AlgebraicSymMatrix measCovar = measCovar1/weightSum + measCovar2/weightSum/weightSum;

  GlobalPoint newPos(meanPosition[0], meanPosition[1], meanPosition[2]);

// #ifndef CMS_NO_COMPLEX_RETURNS
  return VertexState(newPos, GlobalError(measCovar), weightSum);
// #else
//   VertexState theFinalVS(newPos, GlobalError(measCovar), weightSum);
//   return theFinalVS;
// #endif
}

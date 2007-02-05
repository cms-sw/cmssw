#include "RecoVertex/GaussianSumVertexFit/interface/BasicMultiVertexState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

using namespace std;

BasicMultiVertexState::
BasicMultiVertexState(const vector<VertexState>& vsComp) :
  theComponents(vsComp), theCombinedStateUp2Date( false) {}


GlobalPoint BasicMultiVertexState::position() const
{
  checkCombinedState();
  return theCombinedState.position();
}


GlobalError BasicMultiVertexState::error() const
{
  checkCombinedState();
  return theCombinedState.error();
}


GlobalWeight BasicMultiVertexState::weight() const
{
  checkCombinedState();
  return theCombinedState.weight();
}


AlgebraicVector BasicMultiVertexState::weightTimesPosition() const
{
  checkCombinedState();
  return theCombinedState.weightTimesPosition();
}


// RefCountedVertexSeed BasicMultiVertexState::seedWithoutTracks() const
// {
//   checkCombinedState();
//   return theCombinedState.seedWithoutTracks();
// }

double BasicMultiVertexState::weightInMixture() const {
  if (theComponents.empty()) {
    cout << "Asking for weight of empty MultiVertexState, returning zero!" << endl;
    throw VertexException("Asking for weight of empty MultiVertexState, returning zero!");
    return 0.;
  }

  double weight = 0.;
  for (vector<VertexState>::const_iterator it = theComponents.begin(); 
  	it != theComponents.end(); it++) {
    weight += it->weightInMixture();
  }
  return weight;
}

void BasicMultiVertexState::checkCombinedState() const
{
  if (theCombinedStateUp2Date) return;

  theCombinedState = theCombiner.combine(theComponents);
  theCombinedStateUp2Date = true;
}


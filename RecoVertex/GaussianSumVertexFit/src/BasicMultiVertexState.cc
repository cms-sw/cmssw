#include "RecoVertex/GaussianSumVertexFit/interface/BasicMultiVertexState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

using namespace std;

BasicMultiVertexState::BasicMultiVertexState(const vector<VertexState>& vsComp)
    : valid(true), theComponents(vsComp), theCombinedStateUp2Date(false) {}

GlobalPoint BasicMultiVertexState::position() const {
  checkCombinedState();
  return theCombinedState.position();
}

double BasicMultiVertexState::time() const {
  checkCombinedState();
  return theCombinedState.time();
}

GlobalError BasicMultiVertexState::error() const {
  checkCombinedState();
  return theCombinedState.error();
}

double BasicMultiVertexState::timeError() const {
  checkCombinedState();
  return theCombinedState.timeError();
}

GlobalError BasicMultiVertexState::error4D() const {
  checkCombinedState();
  return theCombinedState.error4D();
}

GlobalWeight BasicMultiVertexState::weight() const {
  checkCombinedState();
  return theCombinedState.weight();
}

GlobalWeight BasicMultiVertexState::weight4D() const {
  checkCombinedState();
  return theCombinedState.weight4D();
}

AlgebraicVector3 BasicMultiVertexState::weightTimesPosition() const {
  checkCombinedState();
  return theCombinedState.weightTimesPosition();
}

AlgebraicVector4 BasicMultiVertexState::weightTimesPosition4D() const {
  checkCombinedState();
  return theCombinedState.weightTimesPosition4D();
}

// RefCountedVertexSeed BasicMultiVertexState::seedWithoutTracks() const
// {
//   checkCombinedState();
//   return theCombinedState.seedWithoutTracks();
// }

double BasicMultiVertexState::weightInMixture() const {
  if (!valid)
    throw VertexException("BasicSingleVertexState::invalid");
  if (theComponents.empty()) {
    cout << "Asking for weight of empty MultiVertexState, returning zero!" << endl;
    throw VertexException("Asking for weight of empty MultiVertexState, returning zero!");
    return 0.;
  }

  double weight = 0.;
  for (vector<VertexState>::const_iterator it = theComponents.begin(); it != theComponents.end(); it++) {
    weight += it->weightInMixture();
  }
  return weight;
}

void BasicMultiVertexState::checkCombinedState() const {
  if (!valid)
    throw VertexException("BasicSingleVertexState::invalid");
  if (theCombinedStateUp2Date)
    return;

  theCombinedState = theCombiner.combine(theComponents);
  theCombinedStateUp2Date = true;
}

#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateAssembler.h"
#include "TrackingTools/GsfTools/interface/GetComponents.h"
#include "TrackingTools/GsfTools/interface/BasicMultiTrajectoryState.h"
#include "TrackingTools/GsfTools/src/TrajectoryStateLessWeight.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

MultiTrajectoryStateAssembler::MultiTrajectoryStateAssembler()
    : combinationDone(false), thePzError(false), theValidWeightSum(0.), theInvalidWeightSum(0.) {
  //
  // parameters (could be configurable)
  //
  sortStates = false;
  minValidFraction = 0.01;
  minFractionalWeight = 1.e-6;  // 4;
}

void MultiTrajectoryStateAssembler::addState(const TrajectoryStateOnSurface tsos) {
  //
  // refuse to add states after combination has been done
  //
  if (combinationDone)
    throw cms::Exception("LogicError") << "MultiTrajectoryStateAssembler: trying to add states after combination";
  //
  // Verify validity of state to be added
  //
  if (!tsos.isValid())
    throw cms::Exception("LogicError") << "MultiTrajectoryStateAssembler: trying to add invalid state";
  //
  // Add components (i.e. state to be added can be single or multi state)
  //
  GetComponents comps(tsos);
  const MultiTSOS &components(comps());
  addStateVector(components);
}

void MultiTrajectoryStateAssembler::addStateVector(const MultiTSOS &states) {
  //
  // refuse to add states after combination has been done
  //
  if (combinationDone)
    throw cms::Exception("LogicError") << "MultiTrajectoryStateAssembler: trying to add states after combination";
  //
  // sum up weights (all components are supposed to be valid!!!) and
  // check for consistent pz
  //
  double sum(0.);
  double pzFirst = theStates.empty() ? 0. : theStates.front().localParameters().pzSign();
  for (MultiTSOS::const_iterator i = states.begin(); i != states.end(); i++) {
    if (!(i->isValid()))
      throw cms::Exception("LogicError") << "MultiTrajectoryStateAssembler: trying to add invalid state";
    // weights
    sum += i->weight();
    // check on p_z
    if (!theStates.empty() && pzFirst * i->localParameters().pzSign() < 0.)
      thePzError = true;
  }
  theValidWeightSum += sum;
  //
  // add to vector of states
  //
  theStates.insert(theStates.end(), states.begin(), states.end());
}

void MultiTrajectoryStateAssembler::addInvalidState(const double weight) {
  //
  // change status of combination (contains at least one invalid state)
  //
  theInvalidWeightSum += weight;
}

TrajectoryStateOnSurface MultiTrajectoryStateAssembler::combinedState() {
  //
  // Prepare resulting state vector
  //
  if (!prepareCombinedState())
    return TSOS();
  //
  // If invalid states in input: use reweighting
  //
  if (theInvalidWeightSum > 0.)
    return reweightedCombinedState(theValidWeightSum + theInvalidWeightSum);
  //
  // Return new multi state without reweighting
  //
  return TSOS((BasicTrajectoryState *)(new BasicMultiTrajectoryState(theStates)));
}

TrajectoryStateOnSurface MultiTrajectoryStateAssembler::combinedState(const float newWeight) {
  //
  // Prepare resulting state vector
  //
  if (!prepareCombinedState())
    return TSOS();
  //
  // return reweighted state
  //
  return reweightedCombinedState(newWeight);
}

bool MultiTrajectoryStateAssembler::prepareCombinedState() {
  //
  // Protect against empty combination (no valid input state)
  //
  if (invalidCombinedState())
    return false;
  //
  // Check for states with wrong pz
  //
  if (thePzError)
    removeWrongPz();
  //
  // Check for minimum fraction of valid states
  //
  double allWeights(theValidWeightSum + theInvalidWeightSum);
  if (theInvalidWeightSum > 0. && theValidWeightSum < minValidFraction * allWeights)
    return false;
  //
  // remaining part to be done only once
  //
  if (combinationDone)
    return true;
  combinationDone = true;
  //
  // Remove states with negligible weights
  //
  removeSmallWeights();
  if (invalidCombinedState())
    return false;
  //
  // Sort output by weights?
  //
  if (sortStates)
    sort(theStates.begin(), theStates.end(), TrajectoryStateLessWeight());

  return true;
}

TrajectoryStateOnSurface MultiTrajectoryStateAssembler::reweightedCombinedState(const double newWeight) const {
  //
  // check status
  //
  if (invalidCombinedState())
    return TSOS();
  //
  // scaling factor
  //
  double factor = theValidWeightSum > 0. ? newWeight / theValidWeightSum : 1;
  //
  // create new vector of states & combined state
  //
  MultiTSOS reweightedStates;
  reweightedStates.reserve(theStates.size());
  for (auto const &is : theStates) {
    auto oldWeight = is.weight();
    reweightedStates.emplace_back(factor * oldWeight,
                                  is.localParameters(),
                                  is.localError(),
                                  is.surface(),
                                  &(is.globalParameters().magneticField()),
                                  is.surfaceSide());
  }
  return TSOS((BasicTrajectoryState *)(new BasicMultiTrajectoryState(reweightedStates)));
}

void MultiTrajectoryStateAssembler::removeSmallWeights() {
  //
  // check total weight
  //
  auto totalWeight(theInvalidWeightSum + theValidWeightSum);
  if (totalWeight == 0.) {
    theStates.clear();
    return;
  }
  theStates.erase(
      std::remove_if(theStates.begin(),
                     theStates.end(),
                     [&](MultiTSOS::value_type const &s) { return s.weight() < minFractionalWeight * totalWeight; }),
      theStates.end());
}

void MultiTrajectoryStateAssembler::removeWrongPz() {
  LogDebug("GsfTrackFitters") << "MultiTrajectoryStateAssembler: found at least one state with inconsistent pz\n"
                              << "  #state / weights before cleaning = " << theStates.size() << " / "
                              << theValidWeightSum << " / " << theInvalidWeightSum;
  //
  // Calculate average pz
  //
  double meanPz(0.);
  for (auto const &is : theStates)
    meanPz += is.weight() * is.localParameters().pzSign();
  meanPz /= theValidWeightSum;
  //
  // Now keep only states compatible with the average pz
  //
  theValidWeightSum = 0.;
  MultiTSOS oldStates(theStates);
  theStates.clear();
  for (auto const &is : oldStates) {
    if (meanPz * is.localParameters().pzSign() >= 0.) {
      theValidWeightSum += is.weight();
      theStates.push_back(is);
    } else {
      theInvalidWeightSum += is.weight();
      LogDebug("GsfTrackFitters") << "removing  weight / pz / global position = " << is.weight() << " "
                                  << is.localParameters().pzSign() << " " << is.globalPosition();
    }
  }
  LogDebug("GsfTrackFitters") << "  #state / weights after cleaning = " << theStates.size() << " / "
                              << theValidWeightSum << " / " << theInvalidWeightSum;
}

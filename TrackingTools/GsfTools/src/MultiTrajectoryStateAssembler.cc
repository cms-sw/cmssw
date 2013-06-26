#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateAssembler.h"

#include "TrackingTools/GsfTools/interface/BasicMultiTrajectoryState.h"
#include "TrackingTools/GsfTools/src/TrajectoryStateLessWeight.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

MultiTrajectoryStateAssembler::MultiTrajectoryStateAssembler () :
  combinationDone(false),
  thePzError(false),
  theValidWeightSum(0.),
  theInvalidWeightSum(0.)
{
  //
  // parameters (could be configurable)
  //
  sortStates = false;
  minValidFraction = 0.01;
  minFractionalWeight = 1.e-6;
  //   //
  //   // Timers
  //   //
  //   if ( theTimerAdd==0 ) {
  //     theTimerAdd = 
  //       &(*TimingReport::current())[string("MultiTrajectoryStateAssembler::addState")]; 
  //     theTimerAdd->switchCPU(false);
  //     theTimerComb = 
  //       &(*TimingReport::current())[string("MultiTrajectoryStateAssembler::combinedState")]; 
  //     theTimerComb->switchCPU(false);
  //   }
}  

void MultiTrajectoryStateAssembler::addState (const TrajectoryStateOnSurface tsos) {
  //   // Timer
  //   TimeMe t(*theTimerAdd,false);
  //
  // refuse to add states after combination has been done
  //
  if ( combinationDone )
    throw cms::Exception("LogicError") 
      << "MultiTrajectoryStateAssembler: trying to add states after combination";
  //
  // Verify validity of state to be added
  //
  if ( !tsos.isValid() )
    throw cms::Exception("LogicError") << "MultiTrajectoryStateAssembler: trying to add invalid state";
  //
  // Add components (i.e. state to be added can be single or multi state)
  //
  MultiTSOS components(tsos.components());
  addStateVector(components);
}

void MultiTrajectoryStateAssembler::addStateVector (const MultiTSOS& states)
{
  //
  // refuse to add states after combination has been done
  //
  if ( combinationDone )
    throw cms::Exception("LogicError") 
      << "MultiTrajectoryStateAssembler: trying to add states after combination";
  //
  // sum up weights (all components are supposed to be valid!!!) and
  // check for consistent pz
  //
  double sum(0.);
  double pzFirst = theStates.empty() ? 0. : theStates.front().localParameters().pzSign();
  for ( MultiTSOS::const_iterator i=states.begin();
	i!=states.end(); i++ ) {
    if ( !(i->isValid()) )
      throw cms::Exception("LogicError") 
	<< "MultiTrajectoryStateAssembler: trying to add invalid state";
    // weights
    sum += i->weight();
    // check on p_z
    if ( !theStates.empty() && 
	 pzFirst*i->localParameters().pzSign()<0. )  thePzError = true;
  }
  theValidWeightSum += sum;
  //
  // add to vector of states
  //
  theStates.insert(theStates.end(),states.begin(),states.end());
}


void MultiTrajectoryStateAssembler::addInvalidState (const double weight) {
  //
  // change status of combination (contains at least one invalid state)
  //
  theInvalidWeightSum += weight;
}

TrajectoryStateOnSurface MultiTrajectoryStateAssembler::combinedState () {
  //   // Timer
  //   TimeMe t(*theTimerComb,false);
  //
  // Prepare resulting state vector
  //
  if ( !prepareCombinedState() )  return TSOS();
  //
  // If invalid states in input: use reweighting
  //
  if ( theInvalidWeightSum>0. )  
    return reweightedCombinedState(theValidWeightSum+theInvalidWeightSum);
  //
  // Return new multi state without reweighting
  //
  return TSOS(new BasicMultiTrajectoryState(theStates));
}

TrajectoryStateOnSurface MultiTrajectoryStateAssembler::combinedState (const float newWeight) {
  //   // Timer
  //   TimeMe t(*theTimerComb,false);
  //
  // Prepare resulting state vector
  //
  if ( !prepareCombinedState() )  return TSOS();
  //
  // return reweighted state
  //
  return reweightedCombinedState(newWeight);
}

bool
MultiTrajectoryStateAssembler::prepareCombinedState () {
  //
  // Protect against empty combination (no valid input state)
  //
  if ( invalidCombinedState() )  return false;
  //
  // Check for states with wrong pz
  //
  if ( thePzError )  removeWrongPz();
  //
  // Check for minimum fraction of valid states
  //
  double allWeights(theValidWeightSum+theInvalidWeightSum);
  if ( theInvalidWeightSum>0. && (theValidWeightSum/allWeights)<minValidFraction )  return false;
  //
  // remaining part to be done only once
  //
  if ( combinationDone )  return true;
  else  combinationDone = true;
  //
  // Remove states with negligible weights
  //
  removeSmallWeights();
  if ( invalidCombinedState() )  return false;
  //
  // Sort output by weights?
  //
  if ( sortStates ) 
    sort(theStates.begin(),theStates.end(),TrajectoryStateLessWeight());

  return true;
}

TrajectoryStateOnSurface
MultiTrajectoryStateAssembler::reweightedCombinedState (const double newWeight) const {
  //
  // check status
  //
  if ( invalidCombinedState() )  return TSOS();
  //
  // scaling factor
  //
  double factor = theValidWeightSum>0. ? newWeight/theValidWeightSum : 1;
  //
  // create new vector of states & combined state
  //
  MultiTSOS reweightedStates;
  reweightedStates.reserve(theStates.size());
  for ( MultiTSOS::const_iterator i=theStates.begin();
	i!=theStates.end(); i++ ) {
    double oldWeight = i->weight();
    reweightedStates.push_back(TrajectoryStateOnSurface(i->localParameters(),
							i->localError(),
							i->surface(),
							&(i->globalParameters().magneticField()),
							i->surfaceSide(),
							factor*oldWeight));
  }
  return TSOS(new BasicMultiTrajectoryState(reweightedStates));
}

void
MultiTrajectoryStateAssembler::removeSmallWeights()
{
  //
  // check total weight
  //
  double totalWeight(theInvalidWeightSum+theValidWeightSum);
  if ( totalWeight == 0. ) {
    theStates.clear();
    return;
  }
  //
  // Loop until no more states are removed
  //
  bool redo;
  do {
    redo = false;
    for ( MultiTSOS::iterator i=theStates.begin();
	  i!=theStates.end(); i++ ) {
      if ( (*i).weight()/totalWeight < minFractionalWeight ) {
	theStates.erase(i);
	redo = true;
	break;
      }
    }
  } while (redo);
}

void
MultiTrajectoryStateAssembler::removeWrongPz () {
  //   edm::LogDebug("MultiTrajectoryStateAssembler") 
  //     << "MultiTrajectoryStateAssembler: found at least one state with inconsistent pz\n"
  //     << "  #state / weights before cleaning = " << theStates.size()
  //     << " / " << theValidWeightSum
  //     << " / " << theInvalidWeightSum;
  //
  // Calculate average pz
  //
  double meanPz(0.);
  for ( MultiTSOS::const_iterator is=theStates.begin();
	is!=theStates.end(); is++ ) {
    meanPz += is->weight()*is->localParameters().pzSign();
    //     edm::LogDebug("MultiTrajectoryStateAssembler") 
    //       << "  weight / pz / global position = " << is->weight() 
    //       << " " << is->localParameters().pzSign() 
    //       << " " << is->globalPosition();
  }
  meanPz /= theValidWeightSum;
  //
  // Now keep only states compatible with the average pz
  //
  //   double oldValidWeight(theValidWeightSum);
  theValidWeightSum = 0.;
  MultiTSOS oldStates(theStates);
  theStates.clear();
  for ( MultiTSOS::const_iterator is=oldStates.begin();
	is!=oldStates.end(); is++ ) {
    if ( meanPz*is->localParameters().pzSign()>=0. ) {
      theValidWeightSum += is->weight();
      theStates.push_back(*is);
    }
    else {
      theInvalidWeightSum += is->weight();
    }
  }
  //   edm::LogDebug("MultiTrajectoryStateAssembler") 
  //     << "  #state / weights after cleaning = " << theStates.size()
  //     << " / " << theValidWeightSum
  //     << " / " << theInvalidWeightSum;
}

// TimingReport::Item * MultiTrajectoryStateAssembler::theTimerAdd(0);
// TimingReport::Item * MultiTrajectoryStateAssembler::theTimerComb(0);

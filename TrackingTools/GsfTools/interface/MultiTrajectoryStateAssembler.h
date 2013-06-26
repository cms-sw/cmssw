#ifndef _TrackerReco_MultiTrajectoryStateAssembler_h_
#define _TrackerReco_MultiTrajectoryStateAssembler_h_

// #include "Utilities/Notification/interface/TimingReport.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include <vector>
// #include <map>

/** \class MultiTrajectoryStateAssembler
 * Collects trajectory states and returns a MultiTrajectoryState.
 */

class MultiTrajectoryStateAssembler {

private:
  typedef TrajectoryStateOnSurface TSOS;
  typedef std::vector<TrajectoryStateOnSurface> MultiTSOS;

public:
  //
  // constructors
  //
  MultiTrajectoryStateAssembler ();
  
  /** Adds a new TrajectoryStateOnSurface to the list 
   *  of components
   */
  void addState (const TrajectoryStateOnSurface);

  /// Adds (the weight of an) invalid state to the list
  void addInvalidState (const double);

  /** Returns the resulting MultiTrajectoryState 
   *  with weight = sum of all valid components.
   */
  TrajectoryStateOnSurface combinedState ();
  /** Returns the resulting MultiTrajectoryState 
   *  renormalised to specified weight.
   */
  TrajectoryStateOnSurface combinedState (const float weight);


private:
  /** Adds a vector of trajectory states
   *  to the list of components
   */
  void addStateVector (const MultiTSOS&);
  /// Checks status of combined state
  inline bool invalidCombinedState () const
  {
    //
    // Protect against empty combination (no valid input state)
    //
    return theStates.empty();
  }
  /// Preparation of combined state (cleaning & sorting)
  bool prepareCombinedState();
  /** Returns the resulting MultiTrajectoryState
   *  with user-supplied total weight.
   */
  TrajectoryStateOnSurface reweightedCombinedState (const double) const;
  /** Removes states with negligible weight (no renormalisation
   * of total weight!).
   */
  void removeSmallWeights ();
  /// Removes states with local p_z != average p_z
  void removeWrongPz ();

private:
  bool sortStates;
  float minValidFraction;
  float minFractionalWeight;

  bool combinationDone;
  bool thePzError;

  double theValidWeightSum;
  double theInvalidWeightSum;
  MultiTSOS theStates;

//   static TimingReport::Item * theTimerAdd;
//   static TimingReport::Item * theTimerComb;

};

#endif

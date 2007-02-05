#ifndef MultiGaussianStateAssembler_h_
#define MultiGaussianStateAssembler_h_

// #include "Utilities/Notification/interface/TimingReport.h"
#include "TrackingTools/GsfTools/interface/RCSingleGaussianState.h"
#include "TrackingTools/GsfTools/interface/RCMultiGaussianState.h"

#include <vector>

/** \class MultiGaussianStateAssembler
 * Collects gaussian states and returns a MultiGaussianState.
 */

class MultiGaussianStateAssembler {

private:
  typedef std::vector<RCSingleGaussianState> SGSVector;

public:
  //
  // constructors
  //
  MultiGaussianStateAssembler (const RCMultiGaussianState & state);
  
  /** Adds a new MultiGaussianState to the list 
   *  of components
   */
  void addState (const RCMultiGaussianState& state);
  void addState (const RCSingleGaussianState& state);

  /** Returns the resulting MultiGaussianState 
   *  with weight = sum of all valid components.
   */
  RCMultiGaussianState combinedState ();
  /** Returns the resulting MultiGaussianState 
   *  renormalised to specified weight.
   */
  RCMultiGaussianState combinedState (const float weight);


private:
  /** Adds a vector of gaussian states
   *  to the list of components
   */
  void addStateVector (const SGSVector&);

  /**
   * Preparation of combined state (cleaning & sorting)
   */
  bool prepareCombinedState();

  /** Returns the resulting MultiGaussianState
   *  with user-supplied total weight.
   */

  RCMultiGaussianState reweightedCombinedState (const double) const;

  /** Removes states with negligible weight (no renormalisation
   * of total weight!).
   */
  void removeSmallWeights ();

private:
  const RCMultiGaussianState theInitialState;
  bool sortStates;
  double minFractionalWeight;

  bool combinationDone;

  double theValidWeightSum;
  SGSVector theStates;
//   static TimingReport::Item * theTimerAdd;
//   static TimingReport::Item * theTimerComb;
  
};

#endif

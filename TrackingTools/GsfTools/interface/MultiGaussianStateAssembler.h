#ifndef MultiGaussianStateAssembler_h_
#define MultiGaussianStateAssembler_h_

#include "TrackingTools/GsfTools/interface/SingleGaussianState.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState.h"

#include <vector>

/** \class MultiGaussianStateAssembler
 * Collects gaussian states and returns a MultiGaussianState.
 */

template <unsigned int N>
class MultiGaussianStateAssembler {

private:
  typedef SingleGaussianState<N> SingleState;
  typedef MultiGaussianState<N> MultiState;
  typedef typename MultiGaussianState<N>::SingleStatePtr SingleStatePtr;
  typedef typename MultiGaussianState<N>::SingleStateContainer SingleStateContainer;

public:
  //
  // constructors
  //
  MultiGaussianStateAssembler (const MultiState & state);
  
  /** Adds a new MultiGaussianState to the list 
   *  of components
   */
  void addState (const MultiState& state);
  void addState (const SingleStatePtr& state);

  /** Returns the resulting MultiGaussianState 
   *  with weight = sum of all valid components.
   */
  MultiState combinedState ();
  /** Returns the resulting MultiGaussianState 
   *  renormalised to specified weight.
   */
  MultiState combinedState (const float weight);


private:
  /** Adds a vector of gaussian states
   *  to the list of components
   */
  void addStateVector (const SingleStateContainer&);

  /**
   * Preparation of combined state (cleaning & sorting)
   */
  bool prepareCombinedState();

  /** Returns the resulting MultiGaussianState
   *  with user-supplied total weight.
   */

  MultiState reweightedCombinedState (const double) const;

  /** Removes states with negligible weight (no renormalisation
   * of total weight!).
   */
  void removeSmallWeights ();

private:
  const MultiState theInitialState;
//   bool sortStates;
  double minFractionalWeight;

  bool combinationDone;

  double theValidWeightSum;
  SingleStateContainer theStates;
//   static TimingReport::Item * theTimerAdd;
//   static TimingReport::Item * theTimerComb;
  
};

#include "TrackingTools/GsfTools/interface/MultiGaussianStateAssembler.icc"

#endif

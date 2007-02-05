#ifndef LargestWeightsStateMerger_H
#define LargestWeightsStateMerger_H

#include "TrackingTools/GsfTools/interface/MultiGaussianStateMerger.h"

/** Merging of a Gaussian mixture by keeping
 *  the number Nmax components with the largest weights.
 */

class LargestWeightsStateMerger : public MultiGaussianStateMerger {

 public:

  LargestWeightsStateMerger(int n) : Nmax(n), theSmallestWeightsMerging(true) {
//     initConfigurables();
  }

  virtual LargestWeightsStateMerger* clone() const
  {  
    return new LargestWeightsStateMerger(*this);
  }
  
  /** Method which does the actual merging. Returns a trimmed MultiGaussianState.
   */

  virtual RCMultiGaussianState merge(const RCMultiGaussianState& mgs) const;
  
 private:

//   void initConfigurables();
  
  int Nmax;
  bool theSmallestWeightsMerging;

};  

#endif

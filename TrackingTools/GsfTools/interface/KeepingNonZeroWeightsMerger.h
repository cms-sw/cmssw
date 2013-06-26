#ifndef KeepingNonZeroWeightsMerger_H
#define KeepingNonZeroWeightsMerger_H

#include "TrackingTools/GsfTools/interface/MultiGaussianStateMerger.h"

/** Merging of a Gaussian mixture by keeping
 *  the components with weights larger than a cut value.
 */

class  KeepingNonZeroWeightsMerger : public MultiGaussianStateMerger {

 public:

  KeepingNonZeroWeightsMerger() : cut(1.e-7) {}
  
  KeepingNonZeroWeightsMerger(const double& value) : cut(value) {}

  virtual KeepingNonZeroWeightsMerger* clone() const
  {  
    return new KeepingNonZeroWeightsMerger(*this);
  }
  
  /** Method which does the actual merging. Returns a trimmed MultiGaussianState
   */

  virtual RCMultiGaussianState merge(const RCMultiGaussianState& mgs) const;

 private:
  
  double cut;

};  

#endif // KeepingNonZeroWeightsMerger_H

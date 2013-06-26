#ifndef MultiGaussianStateCombiner1D_H
#define MultiGaussianStateCombiner1D_H

#include "TrackingTools/GsfTools/interface/SingleGaussianState1D.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"

  /**
   * Class to collapse (combine) a Gaussian mixture of states
   * into one.
   * (c.f. R. Fruewirth et.al., Comp.Phys.Comm 100 (1997) 1
   */

class MultiGaussianStateCombiner1D {

private:
  typedef std::vector<SingleGaussianState1D> VSC;
  
public:

  SingleGaussianState1D combine(const MultiGaussianState1D& theState) const;
  SingleGaussianState1D combine(const VSC& theComponents) const;

};

#endif

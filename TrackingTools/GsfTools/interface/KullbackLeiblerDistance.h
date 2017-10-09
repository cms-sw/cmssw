#ifndef KullbackLeiblerDistance_H
#define KullbackLeiblerDistance_H

#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"

/** Calculation of Kullback-Leibler distance between two Gaussian components.
 */

template <unsigned int N>
class KullbackLeiblerDistance final : public DistanceBetweenComponents<N> {
public:
  
  /** Method which calculates the actual Kullback-Leibler distance.
   */
 double operator() (const SingleGaussianState<N>&, 
			     const SingleGaussianState<N>&) const override;

  virtual KullbackLeiblerDistance<N>* clone() const override
  {  
    return new KullbackLeiblerDistance<N>(*this);
  }
};  

#include "TrackingTools/GsfTools/interface/KullbackLeiblerDistance.icc"

#endif // KullbackLeiblerDistance_H

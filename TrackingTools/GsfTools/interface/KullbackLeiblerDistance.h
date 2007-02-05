#ifndef KullbackLeiblerDistance_H
#define KullbackLeiblerDistance_H

#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"

/** Calculation of Kullback-Leibler distance between two Gaussian components.
 */

class KullbackLeiblerDistance : public DistanceBetweenComponents {

 public:

  /** Method which calculates the actual Kullback-Leibler distance.
   */

  virtual double operator() (const RCSingleGaussianState&, 
			     const RCSingleGaussianState&) const;

  virtual KullbackLeiblerDistance* clone() const
  {  
    return new KullbackLeiblerDistance(*this);
  }

  
};  

#endif // KullbackLeiblerDistance_H

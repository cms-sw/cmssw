#ifndef KullbackLeiblerDistance_H
#define KullbackLeiblerDistance_H

#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"

/** Calculation of Kullback-Leibler distance between two Gaussian components.
 */

template <unsigned int N>
class KullbackLeiblerDistance : public DistanceBetweenComponents<N> {

private:
  typedef typename SingleGaussianState<N>::Vector Vector;
  typedef typename SingleGaussianState<N>::Matrix Matrix;
  
public:
  
  /** Method which calculates the actual Kullback-Leibler distance.
   */

  virtual double operator() (const SingleGaussianState<N>&, 
			     const SingleGaussianState<N>&) const;

  virtual KullbackLeiblerDistance<N>* clone() const
  {  
    return new KullbackLeiblerDistance<N>(*this);
  }

// private:
//   double trace (const Matrix& matrix) const;
};  

#include "TrackingTools/GsfTools/interface/KullbackLeiblerDistance.icc"

#endif // KullbackLeiblerDistance_H

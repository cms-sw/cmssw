#ifndef ExtendedPerigeeTrajectoryError_H
#define ExtendedPerigeeTrajectoryError_H

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class ExtendedPerigeeTrajectoryError
{ 
public:
 ExtendedPerigeeTrajectoryError(): weightAvailable(false),vl(false)
 {}

 ExtendedPerigeeTrajectoryError(const AlgebraicSymMatrix66& covariance):
                               cov(covariance),weightAvailable(false),
			       vl(true)
 {}


/**
 * Access methods
 */

 bool isValid() const
 {return vl;}

 bool weightIsAvailable() const
 {return weightAvailable;}

 const AlgebraicSymMatrix66 & covarianceMatrix()const
 {return cov;}
 
 const AlgebraicSymMatrix66 & weightMatrix(int & error)const
 {
  error = 0;
  if(! weightIsAvailable()) {
    weight = cov.Inverse(error);
   if(error != 0) LogDebug("RecoVertex/ExtendedPerigeeTrajectoryError") 
       << "unable to invert covariance matrix\n";
   weightAvailable = true;
  }
  return weight;
 }
 
private:
 AlgebraicSymMatrix66 cov;
 mutable AlgebraicSymMatrix66 weight;
 mutable bool weightAvailable;
 mutable bool vl;
};
#endif

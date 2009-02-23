#ifndef ExtendedPerigeeTrajectoryError_H
#define ExtendedPerigeeTrajectoryError_H

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

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
  if(! weightIsAvailable())
  {
   int ifail;
//   cout<<"weight is requested for covariance:"<<cov<<endl;
   weight = cov.Inverse(ifail);
   if(ifail != 0) throw VertexException("ExtendedPerigeeTrajectoryError::unable to invert covariance matrix"); 
   weightAvailable = true;
  }
  
//  cout<<"and the weight is: "<< weight<<endl;
  return weight;
 }
 
private:
 AlgebraicSymMatrix66 cov;
 mutable AlgebraicSymMatrix66 weight;
 mutable bool weightAvailable;
 mutable bool vl;
};
#endif

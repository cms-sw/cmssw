#ifndef ExtendedPerigeeTrajectoryError_H
#define ExtendedPerigeeTrajectoryError_H

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"

class ExtendedPerigeeTrajectoryError
{ 
public:
 ExtendedPerigeeTrajectoryError(): weightAvailable(false),vl(false)
 {}

 ExtendedPerigeeTrajectoryError(const AlgebraicSymMatrix& covariance):
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

 const AlgebraicSymMatrix & covarianceMatrix()const
 {return cov;}
 
 const AlgebraicSymMatrix & weightMatrix()const
 {
  if(! weightIsAvailable())
  {
   int ifail;
//   cout<<"weight is requested for covariance:"<<cov<<endl;
   weight = cov.inverse(ifail);
   if(ifail != 0) throw VertexException("ExtendedPerigeeTrajectoryError::unable to invert covariance matrix"); 
   weightAvailable = true;
  }
  
//  cout<<"and the weight is: "<< weight<<endl;
  return weight;
 }
 
private:
 AlgebraicSymMatrix cov;
 mutable AlgebraicSymMatrix weight;
 mutable bool weightAvailable;
 mutable bool vl;
};
#endif

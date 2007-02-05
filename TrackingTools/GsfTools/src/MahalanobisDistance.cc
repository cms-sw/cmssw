#include "TrackingTools/GsfTools/interface/MahalanobisDistance.h"

double MahalanobisDistance::operator() (const RCSingleGaussianState& sgs1, 
					const RCSingleGaussianState& sgs2) const {

//ThS: That check will have to happen somewhere else for TSOS...

//   if (&tsos1.surface() != &tsos2.surface()) {
//     cout << "Trying to calculate distance between components defined "
// 	 << "at different surfaces - returning zero!" << endl;
//     return 0.;
//   }

  AlgebraicVector mu1 = sgs1->mean();
  AlgebraicSymMatrix V1 = sgs1->covariance();
  AlgebraicVector mu2 = sgs2->mean();
  AlgebraicSymMatrix V2 = sgs2->covariance();

  int ierr;
  AlgebraicSymMatrix VsumInverse = (V1 + V2).inverse(ierr);
  AlgebraicVector mudiff = mu1 - mu2;

  double dist = VsumInverse.similarity(mudiff);

  return dist;
}

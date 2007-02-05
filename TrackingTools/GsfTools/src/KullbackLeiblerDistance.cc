#include "TrackingTools/GsfTools/interface/KullbackLeiblerDistance.h"

double KullbackLeiblerDistance::operator() (const RCSingleGaussianState& sgs1, 
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
  AlgebraicSymMatrix G1 = V1.inverse(ierr);
  AlgebraicSymMatrix G2 = V2.inverse(ierr);
  AlgebraicVector mudiff = mu1 - mu2;
  AlgebraicSymMatrix Vdiff = V1 - V2;
  AlgebraicSymMatrix Gdiff = G2 - G1;
  AlgebraicSymMatrix Gsum = G1 + G2;

  double dist = (Vdiff * Gdiff).trace() + Gsum.similarity(mudiff);

  return dist;
}

#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateCombiner.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

TrajectoryStateOnSurface 
MultiTrajectoryStateCombiner::combine(const std::vector<TrajectoryStateOnSurface>& tsos) const {
  
  if (tsos.empty()) {
    edm::LogError("MultiTrajectoryStateCombiner") 
      << "Trying to collapse empty set of trajectory states!";
    return TrajectoryStateOnSurface();
  }

  double pzSign = tsos.front().localParameters().pzSign();
  for (std::vector<TrajectoryStateOnSurface>::const_iterator it = tsos.begin(); 
       it != tsos.end(); it++) {
    if (it->localParameters().pzSign() != pzSign) {
      edm::LogError("MultiTrajectoryStateCombiner") 
	<< "Trying to collapse trajectory states with different signs on p_z!";
      return TrajectoryStateOnSurface();
    }
  }
  
  if (tsos.size() == 1) {
    return TrajectoryStateOnSurface(tsos.front());
  }
  
  double sumw = 0.;
  int dim = tsos.front().localParameters().vector().num_row();
  AlgebraicVector mean(dim,0);
  AlgebraicSymMatrix covarPart1(dim,0), covarPart2(dim,0);
  for (std::vector<TrajectoryStateOnSurface>::const_iterator it1 = tsos.begin(); 
       it1 != tsos.end(); it1++) {
    double weight = it1->weight();
    AlgebraicVector param = it1->localParameters().vector();
    sumw += weight;
    mean += weight * param;
    covarPart1 += weight * it1->localError().matrix();
    for (std::vector<TrajectoryStateOnSurface>::const_iterator it2 = it1 + 1; 
	 it2 != tsos.end(); it2++) {
      AlgebraicVector diff = param - it2->localParameters().vector();
      AlgebraicSymMatrix s(1,1); //stupid trick to make CLHEP work decently
      covarPart2 += weight * it2->weight() * s.similarity(diff.T().T());
    }   
  }
  mean /= sumw;
  AlgebraicSymMatrix covar = covarPart1/sumw + covarPart2/sumw/sumw;

  return TrajectoryStateOnSurface(LocalTrajectoryParameters(mean, pzSign), 
				  LocalTrajectoryError(covar), 
				  tsos.front().surface(), 
				  &(tsos.front().globalParameters().magneticField()),
				  tsos.front().surfaceSide(), 
				  sumw);
}


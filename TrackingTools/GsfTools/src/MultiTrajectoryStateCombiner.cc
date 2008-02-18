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
  //int dim = tsos.front().localParameters().vector().num_row();
  AlgebraicVector5 mean;
  AlgebraicSymMatrix55 covarPart1, covarPart2;
  for (std::vector<TrajectoryStateOnSurface>::const_iterator it1 = tsos.begin(); 
       it1 != tsos.end(); it1++) {
    double weight = it1->weight();
    AlgebraicVector5 param = it1->localParameters().vector();
    sumw += weight;
    mean += weight * param;
    covarPart1 += weight * it1->localError().matrix();
    for (std::vector<TrajectoryStateOnSurface>::const_iterator it2 = it1 + 1; 
	 it2 != tsos.end(); it2++) {
      AlgebraicVector5 diff = param - it2->localParameters().vector();
      AlgebraicSymMatrix11 s = AlgebraicMatrixID(); //stupid trick to make CLHEP work decently
      covarPart2 += weight * it2->weight() * 
      				ROOT::Math::Similarity(AlgebraicMatrix51(diff.Array(), 5), s);
                        //FIXME: we can surely write this thing in a better way
    }   
  }
  double sumwI = 1.0/sumw;
  mean *= sumwI;
  covarPart1 *= sumwI; covarPart2 *= (sumwI*sumwI);
  AlgebraicSymMatrix55 covar = covarPart1 + covarPart2;

  return TrajectoryStateOnSurface(LocalTrajectoryParameters(mean, pzSign), 
				  LocalTrajectoryError(covar), 
				  tsos.front().surface(), 
				  &(tsos.front().globalParameters().magneticField()),
				  tsos.front().surfaceSide(), 
				  sumw);
}


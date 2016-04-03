#include "TrackingTools/TrajectoryCleaning/interface/FastTrajectoryCleaner.h"
void FastTrajectoryCleaner::clean( TrajectoryPointerContainer & tc) const
{
  edm::LogError("FastTrajectoryCleaner") << "not implemented for Trajectory";
  assert(false);
}

void FastTrajectoryCleaner::clean( TempTrajectoryContainer & tc) const
{

  if (tc.size() <= 1) return; // nothing to clean
  float maxScore= -std::numeric_limits<float>::max();
  TempTrajectory * bestTr = nullptr;
  for (auto & it : tc) {
    if (!it.isValid()) continue;
    auto const & pd = it.measurements();
    // count active degree of freedom
    int dof=0;
    for (auto const & im : pd) {
      if(dismissSeed_ & (im.estimate()==0)) continue;
      auto const & h = im.recHitR();
      if (!h.isValid()) continue;
      dof+=h.dimension();
    }
    float score = validHitBonus_*dof - missingHitPenalty_*it.lostHits() - it.chiSquared();
    if (score>=maxScore) {
     bestTr = &it;
     maxScore = score;
    }
  }
  assert(bestTr);
  for (auto & it : tc) {
   if ((&it)!=bestTr) it.invalidate();
  }


}

/*
     auto score = [&](Trajectory const&t)->float {
            // possible variant under study
            // auto ns = t.foundHits()-t.trailingFoundHits();
            //auto penalty =  0.8f*missingHitPenalty_;
            // return validHitBonus_*(t.foundHits()-0.2f*t.cccBadHits())  - penalty*t.lostHits() - t.chiSquared();
         // classical score
         return validHitBonus_*t.foundHits()  - missingHitPenalty_*t.lostHits() - t.chiSquared();
     };
*/

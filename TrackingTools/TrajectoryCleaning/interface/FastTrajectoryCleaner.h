#ifndef TrajectoryCleaning_FastTrajectoryCleaner_h
#define TrajectoryCleaning_FastTrajectoryCleaner_h

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"

/** A concrete TrajectoryCleaner that assumes all trajectories
 * coming from the same seed and therefore incompatible
 *  The "best" trajectory of is kept, the others are invalidated.
 *  The goodness of a track is defined in terms of Chi^2, number of
 *  reconstructed hits, and number of lost hits.
 *  As it can be used during PatternReco there is the option to not consider hits from the common seed
 */


class FastTrajectoryCleaner final : public TrajectoryCleaner {
 public:

  using TrajectoryPointerContainer = TrajectoryCleaner::TrajectoryPointerContainer;
  using	TempTrajectoryContainer = TrajectoryCleaner::TempTrajectoryContainer;

  FastTrajectoryCleaner() :
    validHitBonus_(0.5f*5.0f),
    missingHitPenalty_(20.0f),
    dismissSeed_(true){}


  FastTrajectoryCleaner(float bonus, float penalty, bool noSeed=true) :
    validHitBonus_(0.5f*bonus),
    missingHitPenalty_(penalty),
    dismissSeed_(noSeed){}



  FastTrajectoryCleaner(const edm::ParameterSet & iConfig) :
    validHitBonus_(0.5*iConfig.getParameter<double>("ValidHitBonus")),
    missingHitPenalty_(iConfig.getParameter<double>("MissingHitPenalty")),
    dismissSeed_(iConfig.getParameter<bool>("dismissSeed")){}

  ~FastTrajectoryCleaner() override{}

  void clean(TempTrajectoryContainer&) const override;
  void clean(TrajectoryPointerContainer&) const override;

 private:
  float validHitBonus_; // here per dof
  float missingHitPenalty_;
  bool dismissSeed_;
};

#endif

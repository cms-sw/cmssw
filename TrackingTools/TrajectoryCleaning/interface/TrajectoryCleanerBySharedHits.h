#ifndef TrajectoryCleaning_TrajectoryCleanerBySharedHits_h
#define TrajectoryCleaning_TrajectoryCleanerBySharedHits_h

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"

/** A concrete TrajectoryCleaner that considers two trajectories
 *  to be mutually exclusive if they share more than some fraction
 *  of their hits.
 *  The "best" trajectory of each set of mutually exclusive ones
 *  is kept, the others are eliminated.
 *  The goodness of a track is defined in terms of Chi^2, number of
 *  reconstructed hits, and number of lost hits.
 */


class TrajectoryCleanerBySharedHits : public TrajectoryCleaner {

 public:

  typedef std::vector<Trajectory*> 	TrajectoryPointerContainer;

  TrajectoryCleanerBySharedHits() :
    theFraction(0.19),
    validHitBonus_(5.0),
    missingHitPenalty_(20.0),
    allowSharedFirstHit(true){}
  TrajectoryCleanerBySharedHits(const edm::ParameterSet & iConfig) :
    theFraction(iConfig.getParameter<double>("fractionShared")),
    validHitBonus_(iConfig.getParameter<double>("ValidHitBonus")),
    missingHitPenalty_(iConfig.getParameter<double>("MissingHitPenalty")),
    allowSharedFirstHit(iConfig.getParameter<bool>("allowSharedFirstHit")){}

  virtual ~TrajectoryCleanerBySharedHits(){};

  using TrajectoryCleaner::clean;
  virtual void clean( TrajectoryPointerContainer&) const;

 private:
  double theFraction;
  double validHitBonus_;
  double missingHitPenalty_;
  bool allowSharedFirstHit;

};

#endif

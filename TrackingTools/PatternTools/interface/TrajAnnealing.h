#ifndef TrackingTools_PatternTools_TrajAnnealing_h
#define TrackingTools_PatternTools_TrajAnnealing_h

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include <vector>

/**  This class allow to save all the traj info
  *  for each annealing cycle 
  *  suitable for algorithm like DAF
  *  (not used in default options)
 */

class TrajAnnealing {
public:
  TrajAnnealing() {}
  TrajAnnealing(const Trajectory&, float);

  float getAnnealing() const { return annealing_; }
  Trajectory const& getTraj() const { return traj_; }

  //vector of weights
  std::vector<float> const& weights() const { return theWeights; }
  std::vector<float>& weights() { return theWeights; }

private:
  Trajectory traj_;
  float annealing_ = 0;
  std::vector<float> theWeights;
  TrackingRecHit::RecHitContainer theHits_;

  std::pair<float, std::vector<float> > getAnnealingWeight(const TrackingRecHit& aRecHit) const;
};

// this is our new product, it is simply a
// collection of TrajAnnealing held in an std::vector
using TrajAnnealingCollection = std::vector<TrajAnnealing>;

#endif

#ifndef CD_KFFittingSmoother_H_
#define CD_KFFittingSmoother_H_

#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"

/** A TrajectorySmoother that rpeats the forward fit before smoothing.
 *  This is necessary e.g. when the seed introduced a bias (by using
 *  a beam contraint etc.). 
 */

class KFFittingSmoother : public TrajectoryFitter {

public:
  /// constructor with predefined fitter and smoother and propagator
  KFFittingSmoother(const TrajectoryFitter& aFitter,
                    const TrajectorySmoother& aSmoother) :
    theFitter(aFitter.clone()),
    theSmoother(aSmoother.clone()) {}
  
  virtual ~KFFittingSmoother();
  
  virtual vector<Trajectory> fit(const Trajectory& t) const;
  virtual vector<Trajectory> fit(const TrajectorySeed& aSeed,
				 const edm::OwnVector<TransientTrackingRecHit>& hits, 
				 const TrajectoryStateOnSurface& firstPredTsos) const;
  virtual vector<Trajectory> fit(const TrajectorySeed& aSeed,
				 const edm::OwnVector<TransientTrackingRecHit>& hits) const;

  const TrajectoryFitter* fitter() const {return theFitter;}
  const TrajectorySmoother* smoother() const {return theSmoother;}

  KFFittingSmoother* clone() const {
    return new KFFittingSmoother(*theFitter,*theSmoother);
  }
  
private:

  const TrajectoryFitter* theFitter
;  const TrajectorySmoother* theSmoother;

  vector<Trajectory> smoothingStep(vector<Trajectory>& fitted) const;
  TrajectoryStateWithArbitraryError   tsosWithError;
  
};

#endif //CD_KFFittingSmoother_H_

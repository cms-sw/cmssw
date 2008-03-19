#ifndef CD_KFFittingSmoother_H_
#define CD_KFFittingSmoother_H_

/** \class KFFittingSmoother
 *  A TrajectorySmoother that rpeats the forward fit before smoothing.
 *  This is necessary e.g. when the seed introduced a bias (by using
 *  a beam contraint etc.). Ported from ORCA
 *
 *  $Date: 2008/02/11 11:58:46 $
 *  $Revision: 1.10 $
 *  \author todorov, cerati
 */

#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"

class KFFittingSmoother : public TrajectoryFitter {

public:
  /// constructor with predefined fitter and smoother and propagator
  KFFittingSmoother(const TrajectoryFitter& aFitter,
                    const TrajectorySmoother& aSmoother,
		    double estimateCut = -1,
		    int minNumberOfHits = 5,
		    bool rejectTracks = false) :
    theFitter(aFitter.clone()),
    theSmoother(aSmoother.clone()),
    theEstimateCut(estimateCut),
    theMinNumberOfHits(minNumberOfHits),
    rejectTracksFlag(rejectTracks) {}
  
  virtual ~KFFittingSmoother();
  
  virtual std::vector<Trajectory> fit(const Trajectory& t) const;
  virtual std::vector<Trajectory> fit(const TrajectorySeed& aSeed,
				 const RecHitContainer& hits, 
				 const TrajectoryStateOnSurface& firstPredTsos) const;
  virtual std::vector<Trajectory> fit(const TrajectorySeed& aSeed,
				 const RecHitContainer& hits) const;

  const TrajectoryFitter* fitter() const {return theFitter;}
  const TrajectorySmoother* smoother() const {return theSmoother;}

  KFFittingSmoother* clone() const {
    return new KFFittingSmoother(*theFitter,*theSmoother);
  }
  
private:

  const TrajectoryFitter* theFitter;
  const TrajectorySmoother* theSmoother;
  double theEstimateCut;
  int theMinNumberOfHits;
  bool rejectTracksFlag;
  
  void smoothingStep(std::vector<Trajectory>& fitted, std::vector<Trajectory> & smoothed) const;
  TrajectoryStateWithArbitraryError   tsosWithError;
  
};

#endif //CD_KFFittingSmoother_H_

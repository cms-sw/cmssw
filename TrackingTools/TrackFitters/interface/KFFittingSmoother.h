#ifndef CD_KFFittingSmoother_H_
#define CD_KFFittingSmoother_H_

/** \class KFFittingSmoother
 *  A TrajectorySmoother that rpeats the forward fit before smoothing.
 *  This is necessary e.g. when the seed introduced a bias (by using
 *  a beam contraint etc.). Ported from ORCA
 *
 *  $Date: 2012/09/14 12:39:32 $
 *  $Revision: 1.18 $
 *  \author todorov, cerati
 */

#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"

class KFFittingSmoother GCC11_FINAL : public TrajectoryFitter {

public:
  /// constructor with predefined fitter and smoother and propagator
  KFFittingSmoother(const TrajectoryFitter& aFitter,
                    const TrajectorySmoother& aSmoother,
		    double estimateCut = -1,
		    double logPixelProbabilityCut = -16.0, 
		    int minNumberOfHits = 5,
		    bool rejectTracks = false,
		    bool BreakTrajWith2ConsecutiveMissing = false,
		    bool NoInvalidHitsBeginEnd = false) :
    theFitter(aFitter.clone()),
    theSmoother(aSmoother.clone()),
    theEstimateCut(estimateCut),

    // ggiurgiu@fnal.gov
    theLogPixelProbabilityCut( logPixelProbabilityCut ),
    
    theMinNumberOfHits(minNumberOfHits),
    rejectTracksFlag(rejectTracks),
    breakTrajWith2ConsecutiveMissing(BreakTrajWith2ConsecutiveMissing),
    noInvalidHitsBeginEnd(NoInvalidHitsBeginEnd) {}
  
  virtual ~KFFittingSmoother();
  
  Trajectory fitOne(const Trajectory& t, fitType type) const;
  Trajectory fitOne(const TrajectorySeed& aSeed,
				 const RecHitContainer& hits, 
				 const TrajectoryStateOnSurface& firstPredTsos, fitType type) const;
  Trajectory fitOne(const TrajectorySeed& aSeed,
		    const RecHitContainer& hits, fitType type) const;

  const TrajectoryFitter* fitter() const {return theFitter;}
  const TrajectorySmoother* smoother() const {return theSmoother;}

  KFFittingSmoother* clone() const {
    return new KFFittingSmoother(*theFitter,*theSmoother,
				 theEstimateCut,theLogPixelProbabilityCut,
				 theMinNumberOfHits,rejectTracksFlag,
				 breakTrajWith2ConsecutiveMissing,noInvalidHitsBeginEnd);
  }
  
private:

  Trajectory smoothingStep(Trajectory const & fitted) const {return smoother()->trajectory(fitted);}

private:
  
  const TrajectoryFitter* theFitter;
  const TrajectorySmoother* theSmoother;
  double theEstimateCut;

  double theLogPixelProbabilityCut; // ggiurgiu@fnal.gov

  int theMinNumberOfHits;
  bool rejectTracksFlag;
  bool breakTrajWith2ConsecutiveMissing;
  bool noInvalidHitsBeginEnd;
  
  TrajectoryStateWithArbitraryError   tsosWithError;
  
  /// Method to check that the trajectory has no NaN in the states and chi2
  bool checkForNans(const Trajectory &theTraj) const;
  
};

#endif //CD_KFFittingSmoother_H_

#ifndef CD_FlexibleKFFittingSmoother_H_
#define CD_FlexibleKFFittingSmoother_H_

/** \class FlexibleKFFittingSmoother
 *  Combine different FittingSmoother in a single module
 *
 *  $Date: 2012/09/01 11:08:33 $
 *  $Revision: 1.2 $
 *  \author mangano
 */

#include "TrackingTools/TrackFitters/interface/KFFittingSmoother.h"

class FlexibleKFFittingSmoother GCC11_FINAL : public TrajectoryFitter {

public:
  /// constructor with predefined fitter and smoother and propagator
  FlexibleKFFittingSmoother(const TrajectoryFitter& standardFitter,
			    const TrajectoryFitter& looperFitter) :
    theStandardFitter(standardFitter.clone()),
    theLooperFitter(looperFitter.clone()) {}
  
  virtual ~FlexibleKFFittingSmoother();
  
  Trajectory fitOne(const Trajectory& t,fitType type) const{ return fitter(type)->fitOne(t,type);}
  
  
  Trajectory fitOne(const TrajectorySeed& aSeed,
		    const RecHitContainer& hits, 
		    const TrajectoryStateOnSurface& firstPredTsos,
		    fitType type) const {return fitter(type)->fitOne(aSeed,hits,firstPredTsos,type); }
  
  Trajectory fitOne(const TrajectorySeed& aSeed,
		    const RecHitContainer& hits,
		    fitType type) const { return fitter(type)->fitOne(aSeed,hits,type); }
  
  FlexibleKFFittingSmoother* clone() const {
    return new FlexibleKFFittingSmoother(*theStandardFitter,*theLooperFitter);
  }
  
 private:
  
  const TrajectoryFitter* fitter(fitType type) const {
    return (type==standard) ? theStandardFitter : theLooperFitter;
  }
  
  const TrajectoryFitter* theStandardFitter;
  const TrajectoryFitter* theLooperFitter;
  
};

#endif //CD_FlexibleKFFittingSmoother_H_

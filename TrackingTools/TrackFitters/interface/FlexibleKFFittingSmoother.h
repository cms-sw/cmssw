#ifndef CD_FlexibleKFFittingSmoother_H_
#define CD_FlexibleKFFittingSmoother_H_

/** \class FlexibleKFFittingSmoother
 *  Combine different FittingSmoother in a single module
 *
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

  virtual ~FlexibleKFFittingSmoother(){};

  Trajectory fitOne(const Trajectory& t,fitType type) const{ return fitter(type)->fitOne(t,type);}


  Trajectory fitOne(const TrajectorySeed& aSeed,
		    const RecHitContainer& hits,
		    const TrajectoryStateOnSurface& firstPredTsos,
		    fitType type) const {return fitter(type)->fitOne(aSeed,hits,firstPredTsos,type); }

  Trajectory fitOne(const TrajectorySeed& aSeed,
		    const RecHitContainer& hits,
		    fitType type) const { return fitter(type)->fitOne(aSeed,hits,type); }

  std::unique_ptr<TrajectoryFitter> clone() const override{
    return std::unique_ptr<TrajectoryFitter>(
        new FlexibleKFFittingSmoother(*theStandardFitter,*theLooperFitter));
        }

 private:

        const TrajectoryFitter* fitter(fitType type) const {
      return (type==standard) ? theStandardFitter.get() : theLooperFitter.get();
    }

    const std::unique_ptr<TrajectoryFitter> theStandardFitter;
    const std::unique_ptr<TrajectoryFitter> theLooperFitter;

  };

#endif //CD_FlexibleKFFittingSmoother_H_

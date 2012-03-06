#ifndef CD_FlexibleKFFittingSmoother_H_
#define CD_FlexibleKFFittingSmoother_H_

/** \class FlexibleKFFittingSmoother
 *  Combine different FittingSmoother in a single module
 *
 *  $Date: 2012/03/06 $
 *  $Revision: 1.1 $
 *  \author mangano
 */

#include "TrackingTools/TrackFitters/interface/KFFittingSmoother.h"

class FlexibleKFFittingSmoother : public TrajectoryFitter {

public:
  /// constructor with predefined fitter and smoother and propagator
  FlexibleKFFittingSmoother(const TrajectoryFitter& standardFitter,
			    const TrajectoryFitter& looperFitter) :
    theStandardFitter(standardFitter.clone()),
    theLooperFitter(looperFitter.clone()) {}
  
  virtual ~FlexibleKFFittingSmoother();
  
  virtual std::vector<Trajectory> fit(const Trajectory& t) const {
    return theStandardFitter->fit(t);
  }

  virtual std::vector<Trajectory> fit(const Trajectory& t,fitType type) const;


  virtual std::vector<Trajectory> fit(const TrajectorySeed& aSeed,
				      const RecHitContainer& hits, 
				      const TrajectoryStateOnSurface& firstPredTsos) const{
    return theStandardFitter->fit(aSeed,hits,firstPredTsos);
  }

  virtual std::vector<Trajectory> fit(const TrajectorySeed& aSeed,
				      const RecHitContainer& hits, 
				      const TrajectoryStateOnSurface& firstPredTsos,
				      fitType type) const;
  
  virtual std::vector<Trajectory> fit(const TrajectorySeed& aSeed,
				      const RecHitContainer& hits) const{
    return theStandardFitter->fit(aSeed,hits);
  }

  virtual std::vector<Trajectory> fit(const TrajectorySeed& aSeed,
				      const RecHitContainer& hits,
				      fitType type) const;

  //const TrajectoryFitter* fitter() const {return theFitter;}
  //const TrajectorySmoother* smoother() const {return theSmoother;}

  FlexibleKFFittingSmoother* clone() const {
    return new FlexibleKFFittingSmoother(*theStandardFitter,*theLooperFitter);
  }
  
private:
  const TrajectoryFitter* theStandardFitter;
  const TrajectoryFitter* theLooperFitter;
  
};

#endif //CD_FlexibleKFFittingSmoother_H_

#ifndef BasicMultiTrajectoryState_H
#define BasicMultiTrajectoryState_H

#include "TrackingTools/TrajectoryState/interface/BasicTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateCombiner.h"

class BasicMultiTrajectoryState : public BasicTrajectoryState {

  typedef TrajectoryStateOnSurface        TSOS;  
  
public:

  BasicMultiTrajectoryState( const std::vector<TSOS>& tsvec); 

  BasicMultiTrajectoryState() {}


  /** Rescaling the error of the mixture with a given factor. Please note that
   *  this rescaling is imposed on each of the components of the mixture and does
   *  therefore not exactly correspond to rescaling theCombinedState with the same
   *  factor.
   */

  void rescaleError(double factor);

  virtual BasicMultiTrajectoryState* clone() const {
    return new BasicMultiTrajectoryState(*this);
  }

  virtual std::vector<TrajectoryStateOnSurface> components() const {
    return theStates;
  }


  virtual bool canUpdateLocalParameters() const { return false; }
  virtual void update( const LocalTrajectoryParameters& p,
                       const Surface& aSurface,
                       const MagneticField* field,
                       const SurfaceSide side ) ;
  virtual void update( const LocalTrajectoryParameters& p,
                       const LocalTrajectoryError& err,
                       const Surface& aSurface,
                       const MagneticField* field,
                       const SurfaceSide side,
                       double weight ) ;
private:

  std::vector<TSOS> theStates;

  void combine();

};

#endif

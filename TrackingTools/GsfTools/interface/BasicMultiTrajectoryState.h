#ifndef BasicMultiTrajectoryState_H
#define BasicMultiTrajectoryState_H

#include "TrackingTools/TrajectoryState/interface/BasicTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "FWCore/Utilities/interface/Exception.h"

/** Class which combines a set of components of a Gaussian mixture
 *  into a single component. Given all the components of a mixture, it
 *  calculates the mean and covariance matrix of the entire mixture.
 *  This combiner class can also be used in the process of transforming a
 *  Gaussian mixture into another Gaussian mixture with a smaller number
 *  of components. The relevant formulas can be found in
 *  R. Fruhwirth, Computer Physics Communications 100 (1997), 1.
 */
class BasicMultiTrajectoryState final : public BasicTrajectoryState {

  typedef TrajectoryStateOnSurface        TSOS;  
  
public:

  explicit BasicMultiTrajectoryState( const std::vector<TSOS>& tsvec); 

  BasicMultiTrajectoryState() {}


  /** Rescaling the error of the mixture with a given factor. Please note that
   *  this rescaling is imposed on each of the components of the mixture and does
   *  therefore not exactly correspond to rescaling theCombinedState with the same
   *  factor.
   */

  void rescaleError(double factor);

  pointer clone() const override {
    return build<BasicMultiTrajectoryState>(*this);
  }

  using	Components = BasicTrajectoryState::Components;
  Components const & components() const override {
    return theStates;
  }
  bool singleState() const override { return false;}


  virtual bool canUpdateLocalParameters() const override { return false; }
  virtual void update( const LocalTrajectoryParameters& p,
                       const Surface& aSurface,
                       const MagneticField* field,
                       const SurfaceSide side ) override;

  virtual void update(double weight,
                       const LocalTrajectoryParameters& p,
                       const LocalTrajectoryError& err,
                       const Surface& aSurface,
                       const MagneticField* field,
                       const SurfaceSide side) override;
private:

  Components theStates;

  void combine() dso_internal;

};

#endif

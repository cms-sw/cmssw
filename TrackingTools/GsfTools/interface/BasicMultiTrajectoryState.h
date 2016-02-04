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

  bool isValid() const { return !theStates.empty() && theStates.front().isValid();}

  bool hasError() const {
    if (isValid()) return theStates.front().hasError();
    return false;
  }

  const GlobalTrajectoryParameters& globalParameters() const {
    checkCombinedState();
    return theCombinedState.globalParameters();
  }

  GlobalPoint globalPosition() const {
    checkCombinedState();
    return theCombinedState.globalPosition();
  }
    
  GlobalVector globalMomentum() const {
    checkCombinedState();
    return theCombinedState.globalMomentum();
  }

  GlobalVector globalDirection() const {
    checkCombinedState();
    return theCombinedState.globalDirection();
  }

  TrackCharge charge() const {
    checkCombinedState();
    return theCombinedState.charge();
  }

  double signedInverseMomentum() const {
    checkCombinedState();
    return theCombinedState.signedInverseMomentum();
  }

  double transverseCurvature() const {
    checkCombinedState();
    return theCombinedState.transverseCurvature();
  }

  const CartesianTrajectoryError& cartesianError() const {
    checkCombinedState();
    return theCombinedState.cartesianError();
  }

  const CurvilinearTrajectoryError& curvilinearError() const {
    checkCombinedState();
    return theCombinedState.curvilinearError();
  }

  FreeTrajectoryState* freeTrajectoryState(bool withErrors = true) const {
    checkCombinedState();
    return theCombinedState.freeTrajectoryState(withErrors);
  }

  const MagneticField* magneticField() const;
  
  const LocalTrajectoryParameters& localParameters() const {
    checkCombinedState();
    return theCombinedState.localParameters();
  }

  LocalPoint localPosition() const {
    checkCombinedState();
    return theCombinedState.localPosition();
  }

  LocalVector localMomentum() const {
    checkCombinedState();
    return theCombinedState.localMomentum();
  }

  LocalVector localDirection() const {
    checkCombinedState();
    return theCombinedState.localDirection();
  }

  const LocalTrajectoryError& localError() const {
    checkCombinedState();
    return theCombinedState.localError();
  }

  const Surface& surface() const {
    if (!isValid()) 
      throw cms::Exception("LogicError") 
	<< "surface() called for invalid MultiTrajectoryState";
    return theStates.front().surface();
  }

  double weight() const;

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

  /// Position relative to material, defined relative to momentum vector.
  virtual SurfaceSide surfaceSide() const;

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

  mutable TSOS theCombinedState;
  mutable bool theCombinedStateUp2Date;
  MultiTrajectoryStateCombiner theCombiner;

  void checkCombinedState() const;

};

#endif

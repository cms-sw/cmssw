#ifndef _TRACKER_FREETRAJECTORYSTATE_H_
#define _TRACKER_FREETRAJECTORYSTATE_H_

// base trajectory state class
// track parameters and error covariance matrix

#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CartesianTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "DataFormats/TrajectoryState/interface/TrackCharge.h"
#include "TrackingTools/TrajectoryParametrization/interface/TrajectoryStateExceptions.h"

#include <iosfwd>

#include "FWCore/Utilities/interface/Visibility.h"
#include "FWCore/Utilities/interface/Likely.h"

/** A 6-dimensional state vector of a helix given at some point in 
 *  space along the helix, and the associated error matrix.
 *  The error can be obtained in two different parametrizations:
 *  CurvilinearTrajectoryError and CartesianTrajectoryError
 *  (see descriptions of corresponding classes).
 *  The FreeTrajectoryState can be instantiated with either of these
 *  error parametrisations; it converts from one to the other internally.
 */

class FreeTrajectoryState {
public:
  // construst
  //default constructor - needed for Persistency

  FreeTrajectoryState() : theCurvilinearError(InvalidError()) {}

  FreeTrajectoryState(const GlobalTrajectoryParameters& aGlobalParameters)
      : theGlobalParameters(aGlobalParameters), theCurvilinearError(InvalidError()) {}

  FreeTrajectoryState(const GlobalPoint& aX,
                      const GlobalVector& aP,
                      TrackCharge aCharge,
                      const MagneticField* fieldProvider)
      : theGlobalParameters(aX, aP, aCharge, fieldProvider), theCurvilinearError(InvalidError()) {}

  FreeTrajectoryState(const GlobalPoint& aX,
                      const GlobalVector& aP,
                      TrackCharge aCharge,
                      const MagneticField* fieldProvider,
                      GlobalVector fieldValue)
      : theGlobalParameters(aX, aP, aCharge, fieldProvider, fieldValue), theCurvilinearError(InvalidError()) {}

  FreeTrajectoryState(const GlobalTrajectoryParameters& aGlobalParameters,
                      const CurvilinearTrajectoryError& aCurvilinearError)
      : theGlobalParameters(aGlobalParameters), theCurvilinearError(aCurvilinearError) {}

  FreeTrajectoryState(const GlobalTrajectoryParameters& aGlobalParameters,
                      const CartesianTrajectoryError& aCartesianError)
      : theGlobalParameters(aGlobalParameters) {
    createCurvilinearError(aCartesianError);
  }

  FreeTrajectoryState(const GlobalTrajectoryParameters& aGlobalParameters,
                      const CartesianTrajectoryError&,
                      const CurvilinearTrajectoryError& aCurvilinearError)
      : theGlobalParameters(aGlobalParameters), theCurvilinearError(aCurvilinearError) {}

  // access
  // propagate access to parameters
  GlobalPoint position() const { return theGlobalParameters.position(); }
  GlobalVector momentum() const { return theGlobalParameters.momentum(); }
  TrackCharge charge() const { return theGlobalParameters.charge(); }
  double signedInverseMomentum() const { return theGlobalParameters.signedInverseMomentum(); }
  double transverseCurvature() const { return theGlobalParameters.transverseCurvature(); }

  // direct access

  bool hasCurvilinearError() const { return theCurvilinearError.valid(); }

  bool hasError() const { return hasCurvilinearError(); }

  const GlobalTrajectoryParameters& parameters() const { return theGlobalParameters; }

  CartesianTrajectoryError cartesianError() const {
    if UNLIKELY (!hasError())
      missingError();
    CartesianTrajectoryError aCartesianError;
    createCartesianError(aCartesianError);
    return aCartesianError;
  }

  const CurvilinearTrajectoryError& curvilinearError() const {
    if UNLIKELY (!hasError())
      missingError();
    return theCurvilinearError;
  }

  void rescaleError(double factor);

  void setCartesianError(const CartesianTrajectoryError& err) { createCurvilinearError(err); }
  void setCartesianError(const AlgebraicSymMatrix66& err) { createCurvilinearError(CartesianTrajectoryError(err)); }

  CurvilinearTrajectoryError& setCurvilinearError() { return theCurvilinearError; }

  void setCurvilinearError(const CurvilinearTrajectoryError& err) { theCurvilinearError = err; }
  //  void setCurvilinearError(const AlgebraicSymMatrix55 &err) {
  //        theCurvilinearError = CurvilinearTrajectoryError(err);
  //  }

private:
  void missingError() const;  // dso_internal;

  // convert curvilinear errors to cartesian
  void createCartesianError(CartesianTrajectoryError& aCartesianError) const;  // dso_internal;

  // convert cartesian errors to curvilinear
  void createCurvilinearError(CartesianTrajectoryError const& aCartesianError) const;  // dso_internal;

private:
  GlobalTrajectoryParameters theGlobalParameters;
  mutable CurvilinearTrajectoryError theCurvilinearError;
};

std::ostream& operator<<(std::ostream& os, const FreeTrajectoryState& fts);

#endif

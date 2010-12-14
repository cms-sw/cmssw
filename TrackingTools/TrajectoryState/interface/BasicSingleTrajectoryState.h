#ifndef BasicSingleTrajectoryState_H
#define BasicSingleTrajectoryState_H

#include "TrackingTools/TrajectoryState/interface/BasicTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h"

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointer.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/TrajectoryStateExceptions.h"

/// vvv DEBUG
#include <iostream>

//class MagneticField;
#include "MagneticField/Engine/interface/MagneticField.h"

/** Concrete implementation for the state of one trajectory on a surface.
 */

class BasicSingleTrajectoryState : public BasicTrajectoryState {
public:

  /** Constructor from FTS and surface. For surfaces with material
   *  the side of the surface should be specified explicitely.
   */
  BasicSingleTrajectoryState( const FreeTrajectoryState& fts,
			      const Surface& aSurface,
			      const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);
  /** Constructor from global parameters and surface. For surfaces with material
   *  the side of the surface should be specified explicitely.
   */
  BasicSingleTrajectoryState( const GlobalTrajectoryParameters& par,
			      const Surface& aSurface,
			      const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);
  /** Constructor from global parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely.
   */
  BasicSingleTrajectoryState( const GlobalTrajectoryParameters& par,
			      const CartesianTrajectoryError& err,
			      const Surface& aSurface,
			      const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);
  /** Constructor from global parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely. 
   *  For multi-states the weight should be specified explicitely.
   */
  BasicSingleTrajectoryState( const GlobalTrajectoryParameters& par,
			      const CurvilinearTrajectoryError& err,
			      const Surface& aSurface,
			      const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface,
			      double weight = 1.);
  /** Constructor from global parameters, errors and surface. For multi-states the
   *  weight should be specified explicitely. For backward compatibility without
   *  specification of the side of the surface.
   */
  BasicSingleTrajectoryState( const GlobalTrajectoryParameters& par,
			      const CurvilinearTrajectoryError& err,
			      const Surface& aSurface,
			      double weight);
  /** Constructor from local parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely.
   */
  BasicSingleTrajectoryState( const LocalTrajectoryParameters& par,
			      const Surface& aSurface,
			      const MagneticField* field,
			      const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);
  /** Constructor from local parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely. 
   *  For multi-states the weight should be specified explicitely.
   */
  BasicSingleTrajectoryState( const LocalTrajectoryParameters& par,
			      const LocalTrajectoryError& err,
			      const Surface& aSurface,
			      const MagneticField* field,
			      const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface,
			      double weight = 1.);
  /** Constructor from local parameters, errors and surface. For multi-states the
   *  weight should be specified explicitely. For backward compatibility without
   *  specification of the side of the surface.
   */
  BasicSingleTrajectoryState( const LocalTrajectoryParameters& par,
			      const LocalTrajectoryError& err,
			      const Surface& aSurface,
			      const MagneticField* field,
			      double weight);

/// construct invalid trajectory state (without parameters)
  BasicSingleTrajectoryState(const Surface& aSurface);

  virtual ~BasicSingleTrajectoryState();

  bool isValid() const {
    return theFreeState || theLocalParametersValid;
  }
  static void notValid();

  bool hasError() const;
  
  void missingError(char const * where) const;

// access global parameters/errors
  const GlobalTrajectoryParameters& globalParameters() const {
    return freeTrajectoryState(false)->parameters();
  }
  GlobalPoint globalPosition() const {
    return freeTrajectoryState(false)->position();
  }
  GlobalVector globalMomentum() const {
    return freeTrajectoryState(false)->momentum();
  }
  GlobalVector globalDirection() const {
    return freeTrajectoryState(false)->momentum().unit();
  }  
  TrackCharge charge() const {
    return freeTrajectoryState(false)->charge();
  }
  double signedInverseMomentum() const {
    return freeTrajectoryState(false)->signedInverseMomentum();
  }
  double transverseCurvature() const {
    return freeTrajectoryState(false)->transverseCurvature();
  }
  const CartesianTrajectoryError& cartesianError() const {
    if(!hasError()) missingError(" accesing cartesian error.");
    return freeTrajectoryState()->cartesianError();
  }
  const CurvilinearTrajectoryError& curvilinearError() const {
    if(!hasError()) missingError(" accesing curvilinearerror.");
    return freeTrajectoryState()->curvilinearError();
  }


  FreeTrajectoryState* freeTrajectoryState(bool withErrors = true) const;
  
  const MagneticField *magneticField() const { return theField; }

// access local parameters/errors
  const LocalTrajectoryParameters& localParameters() const {
    if (!isValid()) notValid();
    if (!theLocalParametersValid)
      createLocalParameters();
    return theLocalParameters;
  }
  LocalPoint localPosition() const {
    return localParameters().position();
  }
  LocalVector localMomentum() const {
    return localParameters().momentum();
  }
  LocalVector localDirection() const {
    return localMomentum().unit();
  }

  const LocalTrajectoryError& localError() const {
    if (!hasError()) missingError(" accessing local error.");
    if (!theLocalErrorValid)
      createLocalError();
    return theLocalError;
  }

  const Surface& surface() const {
    return *theSurfaceP;
  }

  virtual double weight() const {return theWeight;} 

  void rescaleError(double factor);


  BasicSingleTrajectoryState* clone() const {
    return new BasicSingleTrajectoryState(*this);
  }

  /// Position relative to material, defined relative to momentum vector.
  virtual SurfaceSide surfaceSide() const {
    return theSurfaceSide;
  }

  virtual bool canUpdateLocalParameters() const { return true; }
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

// create global parameters and errors from local
  void checkGlobalParameters() const;
  void checkCurvilinError() const;
  void checkCartesianError() const;

// create local parameters and errors from global
  void createLocalParameters() const;
  // create local errors from global
  void createLocalError() const;
  void createLocalErrorFromCartesianError() const;
  void createLocalErrorFromCurvilinearError() const;

private:

  mutable DeepCopyPointer<FreeTrajectoryState> theFreeState;

  mutable LocalTrajectoryError      theLocalError;
  mutable LocalTrajectoryParameters theLocalParameters;
  mutable bool                      theLocalParametersValid;
  mutable bool                      theLocalErrorValid;

  mutable bool theGlobalParamsUp2Date;
  mutable bool theCartesianErrorUp2Date;
  mutable bool theCurvilinErrorUp2Date;

 
  SurfaceSide theSurfaceSide;
  ConstReferenceCountingPointer<Surface> theSurfaceP;

  double theWeight;
  const MagneticField* theField;

};

#endif

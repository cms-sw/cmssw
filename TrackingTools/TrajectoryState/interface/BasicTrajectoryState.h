#ifndef BasicTrajectoryState_H
#define BasicTrajectoryState_H

#include "TrackingTools/TrajectoryState/interface/ProxyBase.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "TrackingTools/TrajectoryState/interface/CopyUsingClone.h"

#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CartesianTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"

#include <vector>


#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h"

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointer.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/TrajectoryStateExceptions.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

/// vvv DEBUG
// #include <iostream>

//class MagneticField;
#include "MagneticField/Engine/interface/MagneticField.h"



class TrajectoryStateOnSurface;


/** No so Abstract (anyore) base class for TrajectoryState.
 *  It is ReferenceCounted.
 *
 * VI 8/12/2011   content of BasicSingleTrajectoryState moved here....
 * fully devirtualized
 */
class BasicTrajectoryState : public ReferenceCountedInEvent  {
public:

  typedef BasicTrajectoryState                              BTSOS;
  typedef ProxyBase< BTSOS, CopyUsingClone<BTSOS> >         Proxy;
  typedef ReferenceCountingPointer<BasicTrajectoryState>    RCPtr;
  typedef SurfaceSideDefinition::SurfaceSide SurfaceSide;

private:
  friend class ProxyBase< BTSOS, CopyUsingClone<BTSOS> >;
  friend class ReferenceCountingPointer<BasicTrajectoryState>;
public:

  // default constructor : to make root happy
  BasicTrajectoryState(){}

 /// construct invalid trajectory state (without parameters)
  explicit BasicTrajectoryState(const Surface& aSurface);

  virtual ~BasicTrajectoryState();

  /** Constructor from FTS and surface. For surfaces with material
   *  the side of the surface should be specified explicitely.
   */
  BasicTrajectoryState( const FreeTrajectoryState& fts,
			      const Surface& aSurface,
			      const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);
  /** Constructor from global parameters and surface. For surfaces with material
   *  the side of the surface should be specified explicitely.
   */
  BasicTrajectoryState( const GlobalTrajectoryParameters& par,
			      const Surface& aSurface,
			      const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);
  /** Constructor from global parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely.
   */
  BasicTrajectoryState( const GlobalTrajectoryParameters& par,
			      const CartesianTrajectoryError& err,
			      const Surface& aSurface,
			      const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);

  /** Constructor from global parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely. 
   *  For multi-states the weight should be specified explicitely.
   */
  BasicTrajectoryState( const GlobalTrajectoryParameters& par,
			      const CurvilinearTrajectoryError& err,
			      const Surface& aSurface,
			      const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface,
			      double weight = 1.);
  /** Constructor from global parameters, errors and surface. For multi-states the
   *  weight should be specified explicitely. For backward compatibility without
   *  specification of the side of the surface.
   */
  BasicTrajectoryState( const GlobalTrajectoryParameters& par,
			      const CurvilinearTrajectoryError& err,
			      const Surface& aSurface,
			      double weight);
  /** Constructor from local parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely.
   */
  BasicTrajectoryState( const LocalTrajectoryParameters& par,
			      const Surface& aSurface,
			      const MagneticField* field,
			      const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);
  /** Constructor from local parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely. 
   *  For multi-states the weight should be specified explicitely.
   */
  BasicTrajectoryState( const LocalTrajectoryParameters& par,
			      const LocalTrajectoryError& err,
			      const Surface& aSurface,
			      const MagneticField* field,
			      const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface,
			      double weight = 1.);
  /** Constructor from local parameters, errors and surface. For multi-states the
   *  weight should be specified explicitely. For backward compatibility without
   *  specification of the side of the surface.
   */
  BasicTrajectoryState( const LocalTrajectoryParameters& par,
			      const LocalTrajectoryError& err,
			      const Surface& aSurface,
			      const MagneticField* field,
			      double weight);

 
  bool isValid() const {
    return theFreeState || theLocalParametersValid;
  }


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

  const CartesianTrajectoryError cartesianError() const {
    if unlikely(!hasError()) {
	missingError(" accesing cartesian error.");
	return CartesianTrajectoryError();
      }
    return freeTrajectoryState()->cartesianError();
  }
  const CurvilinearTrajectoryError& curvilinearError() const {
    if unlikely(!hasError()) {
	missingError(" accesing curvilinearerror.");
	static CurvilinearTrajectoryError crap;
	return crap;
      }
    return freeTrajectoryState()->curvilinearError();
  }


  FreeTrajectoryState* freeTrajectoryState(bool withErrors = true) const;
  
  const MagneticField *magneticField() const { return theField; }

// access local parameters/errors
  const LocalTrajectoryParameters& localParameters() const {
    if unlikely(!isValid()) notValid();
    if unlikely(!theLocalParametersValid)
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
    if unlikely(!hasError()) {
	missingError(" accessing local error.");
	return theLocalError;
      }
    if unlikely(theLocalError.invalid()) createLocalError();
    return theLocalError;
  }

  const Surface& surface() const {
    return *theSurfaceP;
  }

  double weight() const {return theWeight;} 

  void rescaleError(double factor);



  /// Position relative to material, defined relative to momentum vector.
  SurfaceSide surfaceSide() const {
    return theSurfaceSide;
  }

  bool hasError() const {
    return (theFreeState && theFreeState->hasError()) || theLocalError.valid();
  }


  virtual BasicTrajectoryState* clone() const=0;

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

public:
  virtual std::vector<TrajectoryStateOnSurface> components() const;

private:

  static void notValid();

  
  void missingError(char const * where) const; // dso_internal;

// create global parameters and errors from local
  void checkGlobalParameters() const dso_internal;
  void checkCurvilinError() const  dso_internal;

// create local parameters and errors from global
  void createLocalParameters() const;
  // create local errors from global
  void createLocalError() const;
  void createLocalErrorFromCurvilinearError() const  dso_internal;

private:

  mutable DeepCopyPointer<FreeTrajectoryState> theFreeState;

  mutable LocalTrajectoryError      theLocalError;
  mutable LocalTrajectoryParameters theLocalParameters;

  mutable bool theLocalParametersValid;
  mutable bool theGlobalParamsUp2Date;

 
  SurfaceSide theSurfaceSide;
  ConstReferenceCountingPointer<Surface> theSurfaceP;

  double theWeight;
  const MagneticField* theField;

};

#endif

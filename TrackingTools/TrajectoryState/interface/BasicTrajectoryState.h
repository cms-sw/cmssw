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


#ifdef DO_BTSCount
class BTSCount {
public:
  BTSCount(){}
  virtual ~BTSCount();
  BTSCount(BTSCount const &){} 

  static unsigned int  maxReferences;
  static unsigned long long  aveReferences;
  static unsigned long long  toteReferences;

  void addReference() const { ++referenceCount_ ; referenceMax_ = std::max(referenceMax_, referenceCount_); }
  void removeReference() const { 
    if( 0 == --referenceCount_ ) {
      delete const_cast<BTSCount*>(this);
    }
  }
  
  unsigned int references() const {return referenceCount_;}
private :
#ifdef CMS_NOCXX11
  mutable unsigned int referenceCount_;
  mutable unsigned int referenceMax_;
#else
  mutable unsigned int referenceCount_=0;
  mutable unsigned int referenceMax_ =0;
#endif
};
#else
typedef ReferenceCountedInEvent  BTSCount;
#endif

/** No so Abstract (anyore) base class for TrajectoryState.
 *  It is ReferenceCounted.
 *
 * VI 8/12/2011   content of BasicSingleTrajectoryState moved here....
 * fully devirtualized
 */
class BasicTrajectoryState : public BTSCount {
  public:

  typedef BasicTrajectoryState                              BTSOS;
  typedef ProxyBase< BTSOS, CopyUsingClone<BTSOS> >         Proxy;
  typedef ReferenceCountingPointer<BasicTrajectoryState>    RCPtr;
  typedef SurfaceSideDefinition::SurfaceSide SurfaceSide;
  typedef Surface SurfaceType;

private:
  friend class ProxyBase< BTSOS, CopyUsingClone<BTSOS> >;
  friend class ReferenceCountingPointer<BasicTrajectoryState>;
public:

  // default constructor : to make root happy
  BasicTrajectoryState() : theValid(false), theWeight(0){}

 /// construct invalid trajectory state (without parameters)
  explicit BasicTrajectoryState(const SurfaceType& aSurface);

  virtual ~BasicTrajectoryState();

  /** Constructor from FTS and surface. For surfaces with material
   *  the side of the surface should be specified explicitely.
   */
  BasicTrajectoryState( const FreeTrajectoryState& fts,
			      const SurfaceType& aSurface,
			      const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);
  /** Constructor from global parameters and surface. For surfaces with material
   *  the side of the surface should be specified explicitely.
   */
  BasicTrajectoryState( const GlobalTrajectoryParameters& par,
			      const SurfaceType& aSurface,
			      const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);
  /** Constructor from global parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely.
   */
  BasicTrajectoryState( const GlobalTrajectoryParameters& par,
			      const CartesianTrajectoryError& err,
			      const SurfaceType& aSurface,
			      const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);

  /** Constructor from global parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely. 
   *  For multi-states the weight should be specified explicitely.
   */
  BasicTrajectoryState( const GlobalTrajectoryParameters& par,
			      const CurvilinearTrajectoryError& err,
			      const SurfaceType& aSurface,
			      const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface,
			      double weight = 1.);
  /** Constructor from global parameters, errors and surface. For multi-states the
   *  weight should be specified explicitely. For backward compatibility without
   *  specification of the side of the surface.
   */
  BasicTrajectoryState( const GlobalTrajectoryParameters& par,
			      const CurvilinearTrajectoryError& err,
			      const SurfaceType& aSurface,
			      double weight);
  /** Constructor from local parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely.
   */
  BasicTrajectoryState( const LocalTrajectoryParameters& par,
			      const SurfaceType& aSurface,
			      const MagneticField* field,
			      const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);
  /** Constructor from local parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely. 
   *  For multi-states the weight should be specified explicitely.
   */
  BasicTrajectoryState( const LocalTrajectoryParameters& par,
			      const LocalTrajectoryError& err,
			      const SurfaceType& aSurface,
			      const MagneticField* field,
			      const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface,
			      double weight = 1.);
  /** Constructor from local parameters, errors and surface. For multi-states the
   *  weight should be specified explicitely. For backward compatibility without
   *  specification of the side of the surface.
   */
  BasicTrajectoryState( const LocalTrajectoryParameters& par,
			      const LocalTrajectoryError& err,
			      const SurfaceType& aSurface,
			      const MagneticField* field,
			      double weight);

  bool isValid() const { return theValid; }


// access global parameters/errors
  const GlobalTrajectoryParameters& globalParameters() const {
    return theFreeState.parameters();
  }
  GlobalPoint globalPosition() const {
    return theFreeState.position();
  }
  GlobalVector globalMomentum() const {
    return theFreeState.momentum();
  }
  GlobalVector globalDirection() const {
    return theFreeState.momentum().unit();
  }  
  TrackCharge charge() const {
    return theFreeState.charge();
  }
  double signedInverseMomentum() const {
    return theFreeState.signedInverseMomentum();
  }
  double transverseCurvature() const {
    return theFreeState.transverseCurvature();
  }

  const CartesianTrajectoryError cartesianError() const {
    if unlikely(!hasError()) {
	missingError(" accesing cartesian error.");
	return CartesianTrajectoryError();
      }
    return freeTrajectoryState(true)->cartesianError();
  }
  const CurvilinearTrajectoryError& curvilinearError() const {
    if unlikely(!hasError()) {
	missingError(" accesing curvilinearerror.");
	static CurvilinearTrajectoryError crap;
	return crap;
      }
    return freeTrajectoryState(true)->curvilinearError();
  }



  FreeTrajectoryState* freeTrajectoryState(bool withErrors=true) const {
    if unlikely(!isValid()) notValid();
    if(withErrors && hasError()) { // this is the right thing
      checkCurvilinError();
    }
    return &theFreeState;
  }

  
  const MagneticField *magneticField() const { return &theFreeState.parameters().magneticField(); }

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

  const SurfaceType& surface() const {
    return *theSurfaceP;
  }

  double weight() const {return theWeight;} 

  void rescaleError(double factor);



  /// Position relative to material, defined relative to momentum vector.
  SurfaceSide surfaceSide() const {
    return theSurfaceSide;
  }

  bool hasError() const {
    return theFreeState.hasError() || theLocalError.valid();
  }


  virtual BasicTrajectoryState* clone() const=0;

  virtual bool canUpdateLocalParameters() const { return true; }

  virtual void update( const LocalTrajectoryParameters& p,
                       const SurfaceType& aSurface,
                       const MagneticField* field,
                       const SurfaceSide side ) ;

  virtual void update( const LocalTrajectoryParameters& p,
                       const LocalTrajectoryError& err,
                       const SurfaceType& aSurface,
                       const MagneticField* field,
                       const SurfaceSide side,
                       double weight ) ;

public:
  virtual std::vector<TrajectoryStateOnSurface> components() const;

private:

  static void notValid();

  
  void missingError(char const * where) const; // dso_internal;

  // create global errors from local
  void checkCurvilinError() const; //  dso_internal;
  
  // create local parameters and errors from global
  void createLocalParameters() const;
  // create local errors from global
  void createLocalError() const;
  void createLocalErrorFromCurvilinearError() const  dso_internal;
  
private:

  mutable FreeTrajectoryState theFreeState;

  mutable LocalTrajectoryError      theLocalError;
  mutable LocalTrajectoryParameters theLocalParameters;

  mutable bool theLocalParametersValid;
  mutable bool theValid;

 
  SurfaceSide theSurfaceSide;
  ConstReferenceCountingPointer<SurfaceType> theSurfaceP;

  double theWeight;

};

#endif

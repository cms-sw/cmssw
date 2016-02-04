#ifndef BasicTrajectoryState_H
#define BasicTrajectoryState_H

#include "TrackingTools/TrajectoryState/interface/ProxyBase.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "TrackingTools/TrajectoryState/interface/CopyUsingClone.h"
//#include "CommonDet/DetUtilities/interface/ReferenceCountingPointer.h"

#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CartesianTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"

#include <vector>

class FreeTrajectoryState;
class Surface;
class TrajectoryStateOnSurface;

/** Abstract base class for TrajectoryState.
 *  It is ReferenceCounted.
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

  virtual ~BasicTrajectoryState();

  virtual BasicTrajectoryState* clone() const = 0;

  virtual bool isValid() const = 0;

  virtual bool hasError() const = 0;

  virtual const GlobalTrajectoryParameters& globalParameters() const = 0;

  virtual GlobalPoint globalPosition() const = 0;

  virtual GlobalVector globalMomentum() const = 0;

  virtual GlobalVector globalDirection() const = 0;

  virtual TrackCharge charge() const = 0;

  virtual double signedInverseMomentum() const = 0;

  virtual double transverseCurvature() const = 0;

  virtual const CartesianTrajectoryError& cartesianError() const = 0;

  virtual const CurvilinearTrajectoryError& curvilinearError() const = 0;

  virtual FreeTrajectoryState* freeTrajectoryState(bool withErrors = true) const = 0;

  virtual const MagneticField *magneticField() const = 0;

  virtual const LocalTrajectoryParameters& localParameters() const = 0;

  virtual LocalPoint localPosition() const = 0;

  virtual LocalVector localMomentum() const = 0;

  virtual LocalVector localDirection() const = 0;

  virtual const LocalTrajectoryError& localError() const = 0;

  virtual const Surface& surface() const = 0;

  virtual double weight() const {return 1.;} 

  virtual void rescaleError(double factor) = 0;

  virtual std::vector<TrajectoryStateOnSurface> components() const;
  /*
  virtual std::vector<RCPtr> components() const {
    std::vector<RCPtr> result(1);
    result[0] = const_cast<BasicTrajectoryState*>(this);
    return result;
  }
  */

  /// Position relative to material, defined relative to momentum vector.
  virtual SurfaceSide surfaceSide() const = 0;

  virtual bool canUpdateLocalParameters() const = 0;
  virtual void update( const LocalTrajectoryParameters& p,
                       const Surface& aSurface,
                       const MagneticField* field,
                       const SurfaceSide side ) = 0;
  virtual void update( const LocalTrajectoryParameters& p,
                       const LocalTrajectoryError& err,
                       const Surface& aSurface,
                       const MagneticField* field,
                       const SurfaceSide side,
                       double weight ) = 0;
};

#endif

#ifndef PathToPlane2Order_H
#define PathToPlane2Order_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "TrackingTools/GeomPropagators/interface/HelixPlaneCrossing.h"
#include "CartesianState.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

/** Computes the path length to reach a plane in general magnetic field.
 *  The problem (starting state and plane) is transformed to a frame where the
 *  starting field is along Z, and the AnalyticalHelixPlaneCrossing is used
 *  to compute the path length.
 */

class RKLocalFieldProvider;

class dso_internal PathToPlane2Order {
public:

    typedef Plane::Scalar                                Scalar;
    typedef Basic3DVector<Scalar>                        Vector3D;
    typedef GloballyPositioned<Scalar>                   Frame;

    PathToPlane2Order( const RKLocalFieldProvider& fld, const Frame* fieldFrame) : 
      theField(fld), theFieldFrame(fieldFrame) {}

    /// the position and momentum are local in the FieldFrame;
    /// the plane is in the global frame
    std::pair<bool,double> operator()( const Plane& plane, 
				       const Vector3D& position,
				       const Vector3D& momentum,
				       double charge,
				       const PropagationDirection propDir = alongMomentum);

    std::pair<bool,double> operator()( const Plane& plane, 
				       const GlobalPoint& position,
				       const GlobalVector& momentum,
				       double charge,
				       const PropagationDirection propDir = alongMomentum) {
	return operator()( plane, theFieldFrame->toLocal(position).basicVector(), 
			   theFieldFrame->toLocal(momentum).basicVector(), charge, propDir);
    }

private:

    const RKLocalFieldProvider& theField;
    const Frame*                theFieldFrame;
};

#endif

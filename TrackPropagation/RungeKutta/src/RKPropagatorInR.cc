#include "TrackPropagation/RungeKutta/interface/RKPropagatorInR.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
// #include "CommonReco/RKPropagators/interface/RK4PreciseSolver.h"
#include "RKCylindricalDistance.h"
#include "CylindricalLorentzForce.h"
#include "RKLocalFieldProvider.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "RKAdaptiveSolver.h"
#include "RKOne4OrderStep.h"
#include "RKOneCashKarpStep.h"
#include "CylindricalState.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"

TrajectoryStateOnSurface 
RKPropagatorInR::propagate (const FreeTrajectoryState& ts, const Cylinder& cyl) const
{
  //typedef RK4PreciseSolver<double,5>           Solver;
    typedef RKAdaptiveSolver<double,RKOne4OrderStep, 5>     Solver;
    //typedef RKAdaptiveSolver<Scalar,RKOneCashKarpStep, 5>   Solver;
    typedef double                                          Scalar;
    typedef Solver::Vector                                  RKVector;

    GlobalPoint pos( ts.position());
    GlobalVector mom( ts.momentum());

    LocalPoint startpos = cyl.toLocal(pos);
    LocalVector startmom = cyl.toLocal(mom);

    CylindricalState startState( startpos, startmom, ts.charge());
    RKVector start = startState.parameters();

    RKLocalFieldProvider localField( *theVolume, cyl);
    CylindricalLorentzForce<double,5> deriv(localField);
    RKCylindricalDistance<double,5> dist;
    double eps = 1.e-5;
    Solver solver;
    try {
	Scalar step = cyl.radius() - startState.rho();
	RKVector rkresult = solver( startState.rho(), start, step, deriv, dist, eps);
	CylindricalState endState( cyl.radius(), rkresult, startState.prSign());
	return TrajectoryStateOnSurface( GlobalTrajectoryParameters( cyl.toGlobal( endState.position()), 
								     cyl.toGlobal( endState.momentum()),
								     TrackCharge(endState.charge()), 
								     theVolume),
					 cyl);
    }
    catch (CylindricalLorentzForceException& e) {
        // the propagation failed due to momentum almost parallel to the plane.
        // This does not mean the propagation is impossible, but it should be done
	// in a different parametrization (e.g. s)  
	return TrajectoryStateOnSurface();
    }
}

TrajectoryStateOnSurface 
RKPropagatorInR::propagate (const FreeTrajectoryState&, const Plane&) const
{
    return TrajectoryStateOnSurface();
}

std::pair< TrajectoryStateOnSurface, double> 
RKPropagatorInR::propagateWithPath (const FreeTrajectoryState&, const Plane&) const
{
    return std::pair< TrajectoryStateOnSurface, double>();
}

std::pair< TrajectoryStateOnSurface, double> 
RKPropagatorInR::propagateWithPath (const FreeTrajectoryState&, const Cylinder&) const
{
    return std::pair< TrajectoryStateOnSurface, double>();
}

Propagator * RKPropagatorInR::clone() const
{
    return new RKPropagatorInR(*this);
}

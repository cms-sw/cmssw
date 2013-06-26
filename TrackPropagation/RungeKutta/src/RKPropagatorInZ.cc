#include "TrackPropagation/RungeKutta/interface/RKPropagatorInZ.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
// #include "CommonReco/RKPropagators/interface/RK4PreciseSolver.h"
#include "RKCurvilinearDistance.h"
#include "CurvilinearLorentzForce.h"
#include "RKLocalFieldProvider.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "RKAdaptiveSolver.h"
#include "RKOne4OrderStep.h"
#include "RKOneCashKarpStep.h"

TrajectoryStateOnSurface 
RKPropagatorInZ::propagate (const FreeTrajectoryState& ts, const Plane& plane) const
{
  //typedef RK4PreciseSolver<double,5>           Solver;
  //typedef RKAdaptiveSolver<double,RKOne4OrderStep, 5>   Solver;
    typedef RKAdaptiveSolver<double,RKOneCashKarpStep, 5>   Solver;
    typedef Solver::Vector                       RKVector;

    GlobalPoint pos( ts.position());
    GlobalVector mom( ts.momentum());

    // cout << "RKPropagatorInZ: starting from FTS " << ts << endl;

    LocalPoint startpos = plane.toLocal(pos);
    LocalVector startmom = plane.toLocal(mom);
    double pzSign = startmom.z() > 0 ? 1.0 : -1.0;

    // cout << "In local plane coordinates: " << startpos << ", momentum " << startmom << endl;

    RKVector start;
    start(0) = startpos.x();
    start(1) = startpos.y();
    start(2) = startmom.x()/startmom.z();
    start(3) = startmom.y()/startmom.z();
    start(4) = pzSign * ts.charge() / startmom.mag();

    // cout << "RKPropagatorInZ: Solving with par " <<  startpos.z() << " and state " << start << endl;

    RKLocalFieldProvider localField( *theVolume, plane);

    CurvilinearLorentzForce<double,5> deriv(localField);
    RKCurvilinearDistance<double,5> dist;
    double eps = 1.e-5;
    Solver solver;
    try {
	RKVector rkresult = solver( startpos.z(), start, -startpos.z(), deriv, dist, eps);

	return TrajectoryStateOnSurface( LocalTrajectoryParameters( rkresult(4), rkresult(2), rkresult(3),
								    rkresult(0), rkresult(1), pzSign),
					 plane, theVolume);
    }
    catch (CurvilinearLorentzForceException& e) {
        // the propagation failed due to momentum almost parallel to the plane.
        // This does not mean the propagation is impossible, but it should be done
	// in a different parametrization (e.g. s)  
	return TrajectoryStateOnSurface();
    }
}

TrajectoryStateOnSurface 
RKPropagatorInZ::propagate (const FreeTrajectoryState&, const Cylinder&) const
{
    return TrajectoryStateOnSurface();
}

std::pair< TrajectoryStateOnSurface, double> 
RKPropagatorInZ::propagateWithPath (const FreeTrajectoryState&, const Plane&) const
{
    return std::pair< TrajectoryStateOnSurface, double>();
}

std::pair< TrajectoryStateOnSurface, double> 
RKPropagatorInZ::propagateWithPath (const FreeTrajectoryState&, const Cylinder&) const
{
    return std::pair< TrajectoryStateOnSurface, double>();
}

Propagator * RKPropagatorInZ::clone() const
{
    return new RKPropagatorInZ(*this);
}

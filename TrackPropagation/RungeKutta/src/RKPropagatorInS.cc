#include "TrackPropagation/RungeKutta/interface/RKPropagatorInS.h"
#include "TrackPropagation/RungeKutta/interface/RKCartesianDistance.h"
#include "TrackPropagation/RungeKutta/interface/CartesianLorentzForce.h"
#include "TrackPropagation/RungeKutta/interface/RKLocalFieldProvider.h"
#include "Geometry/Surface/interface/Plane.h"
#include "Geometry/Surface/interface/Cylinder.h"
#include "TrackPropagation/RungeKutta/interface/RKAdaptiveSolver.h"
#include "TrackPropagation/RungeKutta/interface/RKOne4OrderStep.h"
#include "TrackPropagation/RungeKutta/interface/RKOneCashKarpStep.h"
#include "TrackPropagation/RungeKutta/interface/PathToPlane2Order.h"
#include "TrackPropagation/RungeKutta/interface/CartesianStateAdaptor.h"
#include "TrackingTools/GeomPropagators/interface/StraightLineCylinderCrossing.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"

#include "TrackPropagation/RungeKutta/interface/FrameChanger.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"
#include "TrackPropagation/RungeKutta/interface/AnalyticalErrorPropagation.h"
#include "TrackPropagation/RungeKutta/interface/GlobalParametersWithPath.h"

std::pair<TrajectoryStateOnSurface,double>
RKPropagatorInS::propagateWithPath(const FreeTrajectoryState& fts, 
				   const Plane& plane) const
{
  GlobalParametersWithPath gp =  propagateParametersOnPlane( fts, plane);
  if (!gp) return TsosWP(TrajectoryStateOnSurface(),0.);
  else {
    AnalyticalErrorPropagation errorprop;
    return errorprop( fts, plane, SurfaceSideDefinition::beforeSurface,
		      gp.parameters(),gp.s());
  }
}

std::pair< TrajectoryStateOnSurface, double> 
RKPropagatorInS::propagateWithPath (const FreeTrajectoryState& fts, const Cylinder& cyl) const
{
  GlobalParametersWithPath gp =  propagateParametersOnCylinder( fts, cyl);
  if (!gp) return TsosWP(TrajectoryStateOnSurface(),0.);
  else {
    AnalyticalErrorPropagation errorprop;
    return errorprop( fts, cyl, SurfaceSideDefinition::beforeSurface,
		      gp.parameters(),gp.s());
  }
  
}

GlobalParametersWithPath
RKPropagatorInS::propagateParametersOnPlane( const FreeTrajectoryState& ts, 
					     const Plane& plane) const
{
  if (theVolume != 0) {
    std::cout << "RKPropagatorInS: starting prop to plane in volume with pos " << theVolume->position()
	      << " Z axis " << theVolume->toGlobal( LocalVector(0,0,1)) << std::endl;

    std::cout << "The starting position is " << ts.position() << " (global) "
	      << theVolume->toLocal(ts.position()) << " (local) " << std::endl;
    FrameChanger changer;
    FrameChanger::PlanePtr localPlane = changer.transformPlane( plane, *theVolume);
    std::cout << "The plane position is " << plane.position() << " (global) "
	      << localPlane->position() << " (local) " << std::endl;

    std::cout << "The initial distance to plane is " << plane.localZ( ts.position()) << std::endl;

    StraightLinePlaneCrossing cross( ts.position().basicVector(), ts.momentum().basicVector());
    std::pair<bool,double> res3 = cross.pathLength(plane);
    std::cout << "straight line distance " << res3.first << " " << res3.second << std::endl;
  }

  typedef RKAdaptiveSolver<double,RKOneCashKarpStep, 6>   Solver;
  typedef Solver::Vector                                  RKVector;

  GlobalPoint gpos( ts.position());
  GlobalVector gmom( ts.momentum());
  double startZ = plane.localZ(gpos);

  RKLocalFieldProvider field( fieldProvider());
  PathToPlane2Order pathLength( field, &field.frame());
  CartesianLorentzForce deriv(field, ts.charge());

  RKCartesianDistance dist;
  double eps = theTolerance;
  Solver solver;
  double stot = 0;
  PropagationDirection currentDirection = propagationDirection();

  // in magVolume frame
  RKVector start( CartesianStateAdaptor::rkstate( rkPosition(gpos), rkMomentum(gmom)));
  while (true) {
    CartesianStateAdaptor startState(start);

    std::pair<bool,double> path = pathLength( plane, startState.position(), 
					      startState.momentum(), 
					      (double) ts.charge(), currentDirection);
    if (!path.first) { 
      std::cout << "RKPropagatorInS: Path length calculation to plane failed!" << std::endl;
      std::cout << "...distance to plane " << plane.localZ( globalPosition(startState.position())) << std::endl;
      std::cout << "...Local starting position in volume " << startState.position() << std::endl;
      std::cout << "...Magnetic field " << field.inTesla( startState.position()) << std::endl;


      return GlobalParametersWithPath();
    }
    else {
      std::cout << "RKPropagatorInS: Path lenght to plane is " << path.second << std::endl;
    }

    double sstep = path.second;
    if ( std::abs(sstep) < eps) {
      std::cout << "On-surface accuracy not reached, but pathLength calculation says we are there! "
		<< std::endl << "path " << path.second << " distance to plane is " << startZ << std::endl;
      GlobalTrajectoryParameters res( gtpFromVolumeLocal( startState, ts.charge()));
      return GlobalParametersWithPath( res, stot);
    }

    std::cout << "RKPropagatorInS: Solving for " << sstep 
	      << " current distance to plane is " << startZ << std::endl;

    RKVector rkresult = solver( 0, start, sstep, deriv, dist, eps);
    stot += sstep;
    CartesianStateAdaptor cur( rkresult);
    double remainingZ = plane.localZ( globalPosition(cur.position()));
    if ( fabs(remainingZ) < eps) {
      std::cout << "On-surface accuracy reached! " << remainingZ << std::endl;
      GlobalTrajectoryParameters res( gtpFromVolumeLocal( cur, ts.charge()));
      return GlobalParametersWithPath( res, stot);
    }

    start = rkresult;
    if (remainingZ * startZ > 0) {
      std::cout << "Accuracy not reached yet, trying in same direction again " 
		<< remainingZ << std::endl;
    }
    else {
      std::cout << "Accuracy not reached yet, trying in opposite direction " 
		<< remainingZ << std::endl;
      currentDirection = invertDirection( currentDirection);
    }
    startZ = remainingZ;
  }
}

GlobalParametersWithPath
RKPropagatorInS::propagateParametersOnCylinder( const FreeTrajectoryState& ts, 
						const Cylinder& cyl) const
{
    typedef RKAdaptiveSolver<double,RKOneCashKarpStep, 6>   Solver;
    typedef Solver::Vector                                  RKVector;

    GlobalPoint gpos( ts.position());
    GlobalVector gmom( ts.momentum());
    LocalPoint pos(cyl.toLocal(gpos));
    LocalVector mom(cyl.toLocal(gmom));
    double startR = cyl.radius() - pos.perp();

    // std::cout << "RKPropagatorInS: starting from FTS " << ts << std::endl;

    RKLocalFieldProvider field( fieldProvider(cyl));
    // StraightLineCylinderCrossing pathLength( pos, mom, propagationDirection());
    CartesianLorentzForce deriv(field, ts.charge());

    RKCartesianDistance dist;
    double eps = theTolerance;
    Solver solver;
    double stot = 0;
    PropagationDirection currentDirection = propagationDirection();

    RKVector start( CartesianStateAdaptor::rkstate( pos.basicVector(), mom.basicVector()));
    while (true) {
      CartesianStateAdaptor startState(start);
      StraightLineCylinderCrossing pathLength( LocalPoint(startState.position()), 
					       LocalVector(startState.momentum()), 
					       currentDirection, eps);

      std::pair<bool,double> path = pathLength.pathLength( cyl);
      if (!path.first) { 
	std::cout << "RKPropagatorInS: Path length calculation to cylinder failed!" << std::endl;
	LocalPoint lpos( startState.position());
	std::cout << "Radius " << cyl.radius() << " pos.perp() " << lpos.perp() << std::endl;
	return GlobalParametersWithPath();
      }
      else {
	std::cout << "RKPropagatorInS: Path lenght to cylinder is " << path.second 
		  << " from point (R,z) " << startState.position().perp() 
		  << ", " << startState.position().z()
		  << " to R " << cyl.radius() 
		  << std::endl;
      }

      double sstep = path.second;
      if ( std::abs(sstep) < eps) {
	std::cout << "accuracy not reached, but pathLength calculation says we are there! "
	     << path.second << std::endl;

	GlobalTrajectoryParameters res( gtpFromLocal( startState.position(), 
						      startState.momentum(), 
						      ts.charge(), cyl));
	return GlobalParametersWithPath( res, stot);
      }

      std::cout << "RKPropagatorInS: Solving for " << sstep 
	   << " current distance to cylinder is " << startR << std::endl;

      RKVector rkresult = solver( 0, start, sstep, deriv, dist, eps);
      stot += sstep;
      CartesianStateAdaptor cur( rkresult);
      double remainingR = cyl.radius() - cur.position().perp();
      if ( fabs(remainingR) < eps) {
	std::cout << "Accuracy reached! " << remainingR << std::endl;
	GlobalTrajectoryParameters res( gtpFromLocal( cur.position(), 
						      cur.momentum(), 
						      ts.charge(), cyl));
	return GlobalParametersWithPath( res, stot);
      }

      start = rkresult;
      if (remainingR * startR > 0) {
	std::cout << "Accuracy not reached yet, trying in same direction again " 
	     << remainingR << std::endl;
      }
      else {
	std::cout << "Accuracy not reached yet, trying in opposite direction " 
	     << remainingR << std::endl;
	currentDirection = invertDirection( currentDirection);
      }
      startR = remainingR;
    }
}

TrajectoryStateOnSurface 
RKPropagatorInS::propagate(const FreeTrajectoryState& fts, const Plane& plane) const
{
  return propagateWithPath( fts, plane).first;
}

TrajectoryStateOnSurface
RKPropagatorInS::propagate( const FreeTrajectoryState& fts, const Cylinder& cyl) const
{
  return propagateWithPath( fts, cyl).first;
}

Propagator * RKPropagatorInS::clone() const
{
    return new RKPropagatorInS(*this);
}

GlobalTrajectoryParameters RKPropagatorInS::gtpFromLocal( const Basic3DVector<double>& lpos,
							  const Basic3DVector<double>& lmom,
							  TrackCharge ch, const Surface& surf) const
{
    return GlobalTrajectoryParameters( surf.toGlobal( LocalPoint( lpos)),
				       surf.toGlobal( LocalVector( lmom)), ch, theVolume);
}

RKLocalFieldProvider RKPropagatorInS::fieldProvider() const
{
  return RKLocalFieldProvider( *theVolume);
}

RKLocalFieldProvider RKPropagatorInS::fieldProvider( const Cylinder& cyl) const
{
  return RKLocalFieldProvider( *theVolume, cyl);
}

PropagationDirection RKPropagatorInS::invertDirection( PropagationDirection dir) const
{
  if (dir == anyDirection) return dir;
  return ( dir == alongMomentum ? oppositeToMomentum : alongMomentum);
}

Basic3DVector<double> RKPropagatorInS::rkPosition( const GlobalPoint& pos) const
{
  if (theVolume != 0) return theVolume->toLocal( pos).basicVector();
  else return pos.basicVector();
}

Basic3DVector<double> RKPropagatorInS::rkMomentum( const GlobalVector& mom) const
{
  if (theVolume != 0) return theVolume->toLocal( mom).basicVector();
  else return mom.basicVector();
}

GlobalPoint RKPropagatorInS::globalPosition( const Basic3DVector<double>& pos) const
{
  if (theVolume != 0) return theVolume->toGlobal( LocalPoint(pos));
  else return GlobalPoint(pos);
}

GlobalVector RKPropagatorInS::globalMomentum( const Basic3DVector<double>& mom) const

{
  if (theVolume != 0) return theVolume->toGlobal( LocalVector(mom));
  else return GlobalVector(mom);
}

GlobalTrajectoryParameters 
RKPropagatorInS::gtpFromVolumeLocal( const CartesianStateAdaptor& state, 
				     TrackCharge charge) const
{
  return GlobalTrajectoryParameters( globalPosition(state.position()), 
				     globalMomentum(state.momentum()), 
				     charge, theVolume);
}

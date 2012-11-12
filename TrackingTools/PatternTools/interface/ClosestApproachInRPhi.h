#ifndef _ClosestApproachInRPhi_H_
#define _ClosestApproachInRPhi_H_

#include "TrackingTools/PatternTools/interface/ClosestApproachOnHelices.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 

/** Given two trajectory states, computes the two points of closest approach 
 *  in the transverse plane for the helices extrapolated from these states. 
 *  1) computes the intersections of the circles in transverse plane. 
 *     Two cases: - circles have one or two intersection points; 
 *                - circles do not cross; the points used are 
 *                  the points of closest approach of the two circles. 
 *  2) computes the corresponding z-coordinates. In the case where 
 *     the circles have two intersections, the point for which 
 *     the z-coordinates on the 2 tracks are the closest is chosen. 
 */

class ClosestApproachInRPhi GCC11_FINAL : public ClosestApproachOnHelices {

public:

  ClosestApproachInRPhi() {status_ = false;}
  ~ClosestApproachInRPhi(){}

  virtual bool calculate(const TrajectoryStateOnSurface & sta, 
			 const TrajectoryStateOnSurface & stb);

  virtual bool calculate(const FreeTrajectoryState & sta,
			 const FreeTrajectoryState & stb);

  virtual bool status() const {return status_;}

  /**
   * Returns the two PCA on the trajectories.
   */
  virtual std::pair<GlobalPoint, GlobalPoint> points() const;

  /** Returns not only the points, but the full GlobalTrajectoryParemeters 
   *  at the points of closest approach */
  std::pair <GlobalTrajectoryParameters, GlobalTrajectoryParameters >
  trajectoryParameters () const;

  /** arithmetic mean of the two points of closest approach */
  virtual GlobalPoint crossingPoint() const;

  /** distance between the two points of closest approach in 3D */
  virtual float distance() const;

  /**
   *  Clone method
   */
  virtual ClosestApproachInRPhi * clone() const {
    return new ClosestApproachInRPhi(* this);
  }

private:

  bool compute(const TrackCharge & chargeA, 
	       const GlobalVector & momentumA, 
	       const GlobalPoint & positionA, 
	       const TrackCharge & chargeB, 
	       const GlobalVector & momentumB, 
	       const GlobalPoint & positionB) dso_internal;

  // given the old Parameters, and a new GlobalPoint,
  // we return the full new GlobalTrajectoryParameters at the
  // Point.
  static GlobalTrajectoryParameters
  newTrajectory( const GlobalPoint & newpt,
		 const GlobalTrajectoryParameters & oldpar, double bz)  dso_internal;

  // Computes center coordinates and unsigned radius of circle;
  static void circleParameters(const TrackCharge& charge, 
			       const GlobalVector& momemtum, 
			       const GlobalPoint& position, 
			       double& xc, double& yc, double& r,
			       double bz)  dso_internal;

  // Computes crossing points of 2 circles with centres (cx_i, cy_i) 
  // and unsigned radii r_i. 
  // Two cases: - circles have one or two intersection points; 
  //              return value = 1; 
  //            - circles do not cross; computes point of closest approach 
  //              on each circle; return value = 2;
  // if the calculation fails (e.g. concentric circles), return value = 0;

  static int transverseCoord(double cxa, double cya, double ra, 
			     double cxb, double cyb, double rb, 
			     double & xg1, double & yg1, 
			     double & xg2, double & yg2)  dso_internal;

  // Computes z-coordinate on helix at given transverse coordinates
  static double zCoord(const GlobalVector& mom, const GlobalPoint& pos, 
		       double r, double xc, double yc, double xg, double yg)  dso_internal; 


private:
  GlobalPoint posA, posB;
  GlobalTrajectoryParameters paramA, paramB;
  double bz;
  bool status_;
  
};

#endif

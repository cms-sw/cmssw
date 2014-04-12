#ifndef TR_StraightLine_Propagator_H_
#define TR_StraightLine_Propagator_H_

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

class MagneticField; 


/** As the name indicates, propagates track parameters according to
 *  a straight line model. Intended for test beams without magnetic field
 *  and similar cases.
 * \warning The errors are NOT propagated.
 */

class StraightLinePropagator GCC11_FINAL : public Propagator {

private: 

  typedef FreeTrajectoryState FTS;
  typedef TrajectoryStateOnSurface TSOS;

public:

  StraightLinePropagator( const MagneticField* field,
			  PropagationDirection aDir = alongMomentum) : 
    Propagator(aDir), theField(field) {}

  ~StraightLinePropagator() {}

  virtual TSOS propagate(const FreeTrajectoryState& fts,
			 const Surface& surface) const {
    return Propagator::propagate(fts, surface);
  }
  
  virtual TSOS propagate(const FreeTrajectoryState& fts, 
			 const Plane& plane) const {
    return propagateWithPath(fts,plane).first;
  }

  virtual TSOS propagate(const FreeTrajectoryState& fts, 
			 const Cylinder& cylinder) const {
    return propagateWithPath(fts,cylinder).first;
  }
  
  std::pair<TSOS,double> propagateWithPath(const FreeTrajectoryState& fts, 
				      const Surface& surface) const {
    return Propagator::propagateWithPath(fts,surface);
  }

  std::pair<TSOS,double> propagateWithPath(const FreeTrajectoryState& fts, 
				      const Plane& surface) const;

  std::pair<TSOS,double> propagateWithPath(const FreeTrajectoryState& fts, 
				      const Cylinder& surface) const;

  virtual StraightLinePropagator * clone() const {
    return new StraightLinePropagator(*this);
  }

 
  virtual const MagneticField* magneticField() const {return theField;}

private:

  const MagneticField* theField;

  // compute propagated state, with errors if needed
  TrajectoryStateOnSurface propagatedState(const FreeTrajectoryState& fts, 
					   const Surface& surface, 
					   const AlgebraicMatrix55& jacobian, 
					   const GlobalPoint& x, 
					   const GlobalVector& p) const;

  TrajectoryStateOnSurface propagatedState(const FreeTrajectoryState& fts, 
					   const Surface& surface, 
					   const AlgebraicMatrix55& jacobian, 
					   const LocalPoint& x, 
					   const LocalVector& p) const;


  // compute jacobian of transform
  AlgebraicMatrix55 jacobian(double& s) const;

  // compute propagated x and p and path s, return true when propagation is OK
  bool propagateParametersOnCylinder(const FreeTrajectoryState& fts, 
				     const Cylinder& cylinder, 
				     GlobalPoint& x, 
				     GlobalVector& p, 
				     double& s) const;

  // compute propagated x and p and path s, return true when propagation is OK
  bool propagateParametersOnPlane(const FreeTrajectoryState& fts, 
				  const Plane& plane, 
				  LocalPoint& x, 
				  LocalVector& p, 
				  double& s) const;

};

#endif

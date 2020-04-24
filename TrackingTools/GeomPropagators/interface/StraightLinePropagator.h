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

class StraightLinePropagator final : public Propagator {

private: 

  typedef FreeTrajectoryState FTS;
  typedef TrajectoryStateOnSurface TSOS;

public:

  StraightLinePropagator( const MagneticField* field,
			  PropagationDirection aDir = alongMomentum) : 
    Propagator(aDir), theField(field) {}

  ~StraightLinePropagator() override {}

  using Propagator::propagate;
  using Propagator::propagateWithPath;


  std::pair<TSOS,double> propagateWithPath(const FreeTrajectoryState& fts, 
				      const Plane& surface) const  override;

  std::pair<TSOS,double> propagateWithPath(const FreeTrajectoryState& fts, 
				      const Cylinder& surface) const  override;

  StraightLinePropagator * clone() const  override{
    return new StraightLinePropagator(*this);
  }

 
  const MagneticField* magneticField() const  override {return theField;}

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

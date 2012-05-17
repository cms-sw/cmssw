#ifndef AnalyticalCurvilinearJacobian_H
#define AnalyticalCurvilinearJacobian_H

#include "TrackingTools/AnalyticalJacobians/interface/CurvilinearJacobian.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h" 
#include "DataFormats/GeometryVector/interface/GlobalVector.h" 

/** \class AnalyticalCurvilinearJacobian 
 * Creating Jacobian of transformation within the curvilinear frame.
 * The basic functionality of this class is to provide the (analytical) Jacobian matrix
 * of the transformation within the curvilinear frame from the state defined
 * by globalParameters to the state defined by x and p. This Jacobian can then be used
 * to yield the corresponding error propagation. The current implementation
 * is based on the original derivations by W. Wittek. However, due to the implicit float
 * precision, two terms ((4,1) and (5,1)) have been modified in order to make the 
 * calculations more stable in a numerical sense.
 */

class GlobalTrajectoryParameters;

class AnalyticalCurvilinearJacobian : public CurvilinearJacobian {
 public:
  /// default constructor (for tests)
  AnalyticalCurvilinearJacobian() : theJacobian(AlgebraicMatrixID()) {}

  /// get Field at starting state (internally)
  AnalyticalCurvilinearJacobian(const GlobalTrajectoryParameters& globalParameters,
				const GlobalPoint& x, 
				const GlobalVector& p, 
				const double& s);
  /// new: give Field as a parameter
  AnalyticalCurvilinearJacobian(const GlobalTrajectoryParameters& globalParameters,
				const GlobalPoint& x, 
				const GlobalVector& p, 
				const GlobalVector& theFieldInInverseGeV, 
				const double& s);
  
  virtual ~AnalyticalCurvilinearJacobian() {}
  
  virtual const AlgebraicMatrix55& jacobian() const {return theJacobian;}
  virtual const AlgebraicMatrix jacobian_old() const {return asHepMatrix(theJacobian);}
public:
  /// result for non-vanishing curvature
  void computeFullJacobian (const GlobalTrajectoryParameters&,
			    const GlobalPoint&, const GlobalVector&, const GlobalVector&, 
			    const double& s);
  /// result for non-vanishing curvature and "small" step
  void computeInfinitesimalJacobian (const GlobalTrajectoryParameters&,
				     const GlobalPoint&, const GlobalVector&, const GlobalVector&, 
				     const double& s);
  /// straight line approximation
  void computeStraightLineJacobian (const GlobalTrajectoryParameters&,
				    const GlobalPoint&, const GlobalVector&, 
				    const double& s);

 private:
  
  AlgebraicMatrix55 theJacobian;

};  

#endif

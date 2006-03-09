#ifndef KinematicParametersError_H
#define KinematicParametersError_H

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToCartesian.h"

/**
 * Class to store the error matrix
 * for (x,y,z,p_x,p_y,p_z,m)
 * particle parametrization
 *
 * Kirill Prokofiev January 2003
 */


class KinematicParametersError{

public:
 KinematicParametersError()
 {vl = false;}

 KinematicParametersError(const AlgebraicSymMatrix& er):
                             theCovMatrix(er)
 {vl = true;}
 
/**
 * access methods
 */ 
 
 AlgebraicSymMatrix matrix() const
 {return theCovMatrix;}
 
 
 bool isValid() const
 {return vl;}

private:
 AlgebraicSymMatrix theCovMatrix;
 bool vl;
};
#endif


#ifndef KinematicParametersError_H
#define KinematicParametersError_H

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToCartesian.h"
#include "RecoVertex/KinematicFitPrimitives/interface/Matrices.h"

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

 KinematicParametersError(const AlgebraicSymMatrix77& er):
                             theCovMatrix(er)
 {vl = true;}
 
/**
 * access methods
 */ 
 
 AlgebraicSymMatrix77 matrix() const
 {return theCovMatrix;}
 
 
 bool isValid() const
 {return vl;}

private:
 AlgebraicSymMatrix77 theCovMatrix;
 bool vl;
};
#endif


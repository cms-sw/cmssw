#ifndef EventShapeVariables_h
#define EventShapeVariables_h

#include <vector>
#include "TMatrixDSym.h"
#include "DataFormats/Math/interface/Vector3D.h"


/**
   \class   EventShapeVariables EventShapeVariables.h "TopQuarkAnalysis/TopTools/interface/EventShapeVariables.h"

   \brief   Class for the calculation of event shape variables

   Class for the calculation of several event shape variables. Isotropy, sphericity,
   aplanarity and circularity are supported. The class has a vector of 3 dimentional 
   vectors as input which can be given in cartesian, cylindrical or polar coordinates. 
*/

class EventShapeVariables {

 public:
  /// contructor from XYZ coordinates
  explicit EventShapeVariables(const std::vector<math::XYZVector>& inputVectors);  
  /// contructor from rho eta phi coordinates
  explicit EventShapeVariables(const std::vector<math::RhoEtaPhiVector>& inputVectors);  
  /// contructor from r theta phi coordinates
  explicit EventShapeVariables(const std::vector<math::RThetaPhiVector>& inputVectors);  
  /// default destructir
  ~EventShapeVariables(){};

  /// the return value is 1 for spherical and 0 for events linear in r-phi. This function needs
  /// the number of steps to determined how fine the grnularity of the algorithm in phi should be
  double isotropy(const unsigned int& numberOfSteps = 1000) const;
  /// 1.5*(q1+q2) where 0<=q1<=q2<=q3 are the eigenvalues of the momemtum tensor 
  /// sum{pj[a]*pj[b]}/sum{pj**2} normalized to 1. Return values are 1 for spherical, 3/4 for 
  /// plane and 0 for linear events
  double sphericity()  const;
  /// 1.5*q1 where 0<=q1<=q2<=q3 are the eigenvalues of the momemtum tensor 
  /// sum{p_j[a]*p_j[b]}/sum{p_j**2} normalized to 1. Return values are 0.5 for spherical and 0 
  /// for plane and linear events
  double aplanarity()  const;
  /// the return value is 1 for spherical and 0 linear events in r-phi. This function needs the 
  /// number of steps to determined how fine the grnularity of the algorithm in phi should be
  double circularity(const unsigned int& numberOfSteps = 1000) const;
  
 private:
  /// helper function to fill the 3 dimensional momentum tensor from the inputVecotrs where 
  /// needed
  TMatrixDSym momentumTensor() const;

  /// cashing of input vectors
  std::vector<math::XYZVector> inputVectors_;
};

#endif


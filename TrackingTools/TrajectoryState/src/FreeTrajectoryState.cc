#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToCartesian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include <cmath>
#include<sstream>

// void bPoint(){}

void FreeTrajectoryState::missingError() const {
  std::stringstream form;
  form<< "FreeTrajectoryState: attempt to access errors when none available" <<
    "\nCurvilinear error valid/values :"<< theCurvilinearError.valid() << "\n" 
      <<  theCurvilinearError.matrix();
    edm::LogWarning("FreeTrajectoryState") << "(was exception) " << form.str();
    //  throw TrajectoryStateException(form.str());
    // bPoint();
}

// implementation of non-trivial methods of FreeTrajectoryState

// Warning: these methods violate constness

// convert curvilinear errors to cartesian
void FreeTrajectoryState::createCartesianError(CartesianTrajectoryError & aCartesianError) const{
  
  JacobianCurvilinearToCartesian curv2Cart(theGlobalParameters);
  const AlgebraicMatrix65& jac = curv2Cart.jacobian();

  aCartesianError = 
    ROOT::Math::Similarity(jac, theCurvilinearError.matrix());
}

// convert cartesian errors to curvilinear
void FreeTrajectoryState::createCurvilinearError(CartesianTrajectoryError const& aCartesianError) const{
  
  JacobianCartesianToCurvilinear cart2Curv(theGlobalParameters);
  const AlgebraicMatrix56& jac = cart2Curv.jacobian();
  
  theCurvilinearError = 
    ROOT::Math::Similarity(jac, aCartesianError.matrix());

} 


void FreeTrajectoryState::rescaleError(double factor) {
  if unlikely(!hasError()) return;
  bool zeroField = (parameters().magneticField().nominalValue()==0);  
  if unlikely(zeroField)  theCurvilinearError.zeroFieldScaling(factor*factor);
  else theCurvilinearError *= (factor*factor);
}








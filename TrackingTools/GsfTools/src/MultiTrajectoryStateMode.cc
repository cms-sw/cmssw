#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateMode.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianStateTransform.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"
#include "TrackingTools/GsfTools/interface/GaussianSumUtilities1D.h"

#include <iostream>

bool
MultiTrajectoryStateMode::momentumFromModeCartesian (const TrajectoryStateOnSurface tsos,
						     GlobalVector& momentum) const
{
  //
  // clear result vector and check validity of the TSOS
  //
  momentum = GlobalVector(0.,0.,0.);
  if ( !tsos.isValid() ) {
    edm::LogInfo("MultiTrajectoryStateMode") << "Cannot calculate mode from invalid TSOS";
    return false;
  }
  //  
  // 1D mode computation for px, py and pz
  // 
  std::vector<TrajectoryStateOnSurface> components(tsos.components());
  unsigned int numb = components.size();
  // vectors of components in x, y and z
  std::vector<SingleGaussianState1D> pxStates; pxStates.reserve(numb);
  std::vector<SingleGaussianState1D> pyStates; pyStates.reserve(numb);
  std::vector<SingleGaussianState1D> pzStates; pzStates.reserve(numb);
  for ( std::vector<TrajectoryStateOnSurface>::const_iterator ic=components.begin();
	ic!=components.end(); ++ic ) {
    GlobalVector mom(ic->globalMomentum());
    AlgebraicSymMatrix66 cov(ic->cartesianError().matrix());
    pxStates.push_back(SingleGaussianState1D(mom.x(),cov(3,3),ic->weight()));
    pyStates.push_back(SingleGaussianState1D(mom.y(),cov(4,4),ic->weight()));
    pzStates.push_back(SingleGaussianState1D(mom.z(),cov(5,5),ic->weight()));
  }
  //
  // transformation in 1D multi-states and creation of utility classes
  //
  MultiGaussianState1D pxState(pxStates);
  MultiGaussianState1D pyState(pyStates);
  MultiGaussianState1D pzState(pzStates);
  GaussianSumUtilities1D pxUtils(pxState);
  GaussianSumUtilities1D pyUtils(pyState);
  GaussianSumUtilities1D pzUtils(pzState);
  momentum = GlobalVector(pxUtils.mode().mean(),pyUtils.mode().mean(),pzUtils.mode().mean());
  return true;
}

// bool
// MultiTrajectoryStateMode::momentumFromModeLocal (const TrajectoryStateOnSurface tsos,
// 						 GlobalVector& momentum) const
// {
//   //
//   // clear result vector and check validity of the TSOS
//   //
//   momentum = GlobalVector(0.,0.,0.);
//   if ( !tsos.isValid() ) {
//     edm::LogInfo("MultiTrajectoryStateMode") << "Cannot calculate mode from invalid TSOS";
//     return false;
//   }
//   //  
//   // mode computation for local co-ordinates q/p, dx/dz, dy/dz
//   //
//   double qpMode(0);
//   double dxdzMode(0);
//   double dydzMode(0);
//   // first 3 elements of local parameters = q/p, dx/dz, dy/dz
//   for ( unsigned int iv=0; iv<3; ++iv ) {
//     MultiGaussianState1D state1D = MultiGaussianStateTransform::multiState1D(tsos,iv);
//     GaussianSumUtilities1D utils(state1D);
//     double result = utils.mode().mean();
//     if ( !utils.modeIsValid() )  result = utils.mean();
//     if ( iv==0 )  qpMode = result;
//     else if ( iv==1 )  dxdzMode = result;
//     else  dydzMode = result;
//   }
//   // local momentum vector from dx/dz, dy/dz and q/p + sign of local pz
//   LocalVector localP(dxdzMode,dydzMode,1.);
//   localP *= tsos.localParameters().pzSign()/fabs(qpMode);
//   // conversion to global
//   momentum = tsos.surface().toGlobal(localP);

//   GlobalVector momCart(0.,0.,0.);
//   momentumFromModeCartesian(tsos,momCart);
//   std::cout << "Comparison of modes: " 
// 	    << momentum.x() << " / " << momCart.x() << " ; "
// 	    << momentum.y() << " / " << momCart.y() << " ; "
// 	    << momentum.z() << " / " << momCart.z() << std::endl;
//   std::cout << tsos.localParameters().vector()[0] << " / " << 1./momentum.mag() 
// 	    << " / " << 1./momCart.mag() << std::endl;
//   std::cout << tsos.localParameters().vector()[1] << " / " << momentum.x()/momentum.z() 
// 	    << " / " << momCart.x()/momCart.z() << std::endl;
//   std::cout << tsos.localParameters().vector()[2] << " / " << momentum.y()/momentum.z() 
// 	    << " / " << momCart.y()/momCart.z() << std::endl;
//   return true;
// }


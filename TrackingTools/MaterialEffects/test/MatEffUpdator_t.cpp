// rl, xi 0.0711444 0.000158792

#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "MagneticField/Engine/interface/MagneticField.h"

namespace {

  struct M5T : public  MagneticField {
    explicit M5T(double br) :  m(br,br,5.){}
    virtual GlobalVector inTesla (const GlobalPoint&) const {
      return m;
    }

    GlobalVector m;
  };

}

#include "FWCore/Utilities/interface/HRRealTime.h"
void st(){}
void en(){}

#include<iostream>

int main(int argc, char** argv) {
  double br=0.;
  if (argc>1) br=0.1;
  M5T const m(br);

  Basic3DVector<float>  axis(0.5,1.,1);
  
  Surface::RotationType rot(axis,0.5*M_PI);
  std::cout << rot << std::endl;

  Surface::PositionType pos( 0., 0., 0.);

  Plane plane(pos,rot);
  LocalTrajectoryParameters tpl(-1./3.5, 1.,1., 0.,0.,1.);
  GlobalVector mg = plane.toGlobal(tpl.momentum());
  GlobalTrajectoryParameters tpg(pos,mg,-1., &m);
  double curv =   tpg.transverseCurvature();
  std::cout << curv << " " <<  mg.mag() << std::endl;
  std::cout << tpg.position() << " " << tpg.momentum() << std::endl;

  GlobalTrajectoryParameters tpg0(tpg);

  //HelixForwardPlaneCrossing::PositionType zero(0.,0.,0.); 
  GlobalPoint zero(0.,0.,0.); 
  std::cout << std::endl;

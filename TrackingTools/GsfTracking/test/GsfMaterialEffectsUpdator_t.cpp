// test of  GsfMaterialEffectsUpdator


#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"



#include "TrackingTools/GsfTracking/interface/GsfBetheHeitlerUpdator.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"

#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "MagneticField/Engine/interface/MagneticField.h"

namespace {

  struct M5T : public  MagneticField {
    M5T() :  m(0.,0.,5.){}
    virtual GlobalVector inTesla (const GlobalPoint&) const {
      return m;
    }

    GlobalVector m;
  };

}


#include "FWCore/Utilities/interface/HRRealTime.h"
#include<iostream>
#include<vector>


void st(){}
void en(){}




int main(int argc, char * arg[]) {

  std::string file("BetheHeitler_cdfmom_nC6_O5.par");
  if (argc<2) {
    std::cerr << "parameter file not given in input: default used" << std::endl;
  } else
    file = arg[1];

  std::cout << "using file " << file << std::endl;


  GsfBetheHeitlerUpdator bhu(file,0); 
  
  GsfMaterialEffectsUpdator * meu = &bhu;


  Basic3DVector<float>  axis(0.5,1.,1);
  
  Surface::RotationType rot(axis,0.5*M_PI);

  Surface::PositionType pos( 0., 0., 0.);

  BoundPlane::BoundPlanePointer plane = BoundPlane::build(pos,rot, RectangularPlaneBounds(1.,1.,1));
  plane->setMediumProperties(MediumProperties(0.1,0.3));

  LocalTrajectoryParameters tp(1., 1.,1., 0.,0.,0.);
  
  M5T const m; 
  
  
  TrajectoryStateOnSurface tsos(tp,*plane, &m, SurfaceSideDefinition::beforeSurface);
  
  st();
  meu->updateState(tsos,alongMomentum);
  en();

  return 0;

}

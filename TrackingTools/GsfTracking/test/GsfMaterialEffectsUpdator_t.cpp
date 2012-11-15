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
  plane->setMediumProperties(MediumProperties(1.1,1.3));
  M5T const m; 

  edm::HRTimeDiffType totT=0;
  int n=0;
  double neverKnow=0;
  bool printIt=true;
  for (int i=0; i!=100000; ++i) {
    LocalTrajectoryParameters tp(1.+0.01*i, 1.,1., 0.,0.,1.);
    LocalTrajectoryError lerr(1.,1.,0.1,0.1,0.1);
    
    TrajectoryStateOnSurface tsos(tp,lerr,*plane, &m, SurfaceSideDefinition::beforeSurface);
    if (printIt) {
      std::cout << tsos.globalMomentum() << std::endl;
      std::cout << tsos.localError().matrix() << std::endl;
      std::cout << tsos.weight() << std::endl;
    }
    st();
    totT -= edm::hrRealTime();
    meu->updateState(tsos,alongMomentum);
    totT +=edm::hrRealTime();
    ++n;
    en();
    neverKnow+=tsos.globalMomentum().perp();
    if (printIt) {
      std::cout << tsos.globalMomentum() << std::endl;
      std::cout << tsos.localError().matrix() << std::endl;
      std::cout << tsos.weight() << std::endl;
      printIt=false;
    }
  }

  std::cout << "\nupdate  time " << double(totT)/double(n) << std::endl;
  return neverKnow!=0 ? 0 : 20;

}

#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
double PixelRecoUtilities::
  longitudinalBendingCorrection( double radius, double pt)
{
  double invCurv = bendingRadius(pt);
  if ( invCurv == 0. ) return 0.;
  return  radius/6. * radius*radius/(2.*invCurv * 2.*invCurv);
}


// void PixelRecoUtilities:: MaginTesla( const edm::EventSetup& iSetup){
//   edm::ESHandle<MagneticField> pSetup;
//   iSetup.get<IdealMagneticFieldRecord>().get(pSetup);
//   // mgfieldininversegev=pSetup->inTesla(GlobalPoint(0,0,0))* 2.99792458e-3;
// }
float PixelRecoUtilities::fieldInInvGev() 
{  
 
  //MP da capire come accedere al B 
  static float theInvField = 
      1./fabs(  4*3e-3);
  return theInvField;
}



// #include "Utilities/UI/interface/SimpleConfigurable.h"
// #include <stdio.h>

// TimingReport::Item * PixelRecoUtilities::initTiming(
//     string name, int onLevel)
// {
//   static SimpleConfigurable<int> level(0,"TkTrackingRegions:timingLevel");
//   char Lev[] = "9999 ";
//   if (onLevel >= 0 && onLevel < 9999) sprintf(Lev,"(%d) ", onLevel);

//   TimingReport::Item * item = &(*TimingReport::current())[string(Lev)+name];
//   item->switchCPU(false);
//   if (onLevel > level) item->switchOn(false);
//   return item;
// }

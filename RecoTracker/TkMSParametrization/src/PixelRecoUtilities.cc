#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
double PixelRecoUtilities::
longitudinalBendingCorrection( double radius, double pt,const edm::EventSetup& iSetup)
{
  double invCurv = bendingRadius(pt,iSetup);
  if ( invCurv == 0. ) return 0.;
  return  radius/6. * radius*radius/(2.*invCurv * 2.*invCurv);
}


// void PixelRecoUtilities:: MaginTesla( const edm::EventSetup& iSetup){
//   edm::ESHandle<MagneticField> pSetup;
//   iSetup.get<IdealMagneticFieldRecord>().get(pSetup);
//   // mgfieldininversegev=pSetup->inTesla(GlobalPoint(0,0,0))* 2.99792458e-3;
// }
float PixelRecoUtilities::fieldInInvGev(const edm::EventSetup& iSetup) 
{  
  edm::ESHandle<MagneticField> pSetup;
  iSetup.get<IdealMagneticFieldRecord>().get(pSetup);
  //MP da capire come accedere al B 
  static float theInvField = 
    1./fabs(pSetup->inTesla(GlobalPoint(0,0,0)).z()  *2.99792458e-3);
  return theInvField;
}


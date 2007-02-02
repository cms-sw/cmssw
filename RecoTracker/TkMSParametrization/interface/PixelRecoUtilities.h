#ifndef PixelRecoUtilities_H
#define PixelRecoUtilities_H
#include "Geometry/Vector/interface/GlobalVector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <string>

/** \namespace PixelRecoUtilities
 *  Small utility funcions used during seed generation 
 */


namespace PixelRecoUtilities {

  /** Magnetic field intensity in units of cm/GeV.
   *  The value is cached in a static variable, so the actual
   *  MagneticField is accessed only once.
   */
  float fieldInInvGev(const edm::EventSetup& iSetup);
  //  void MaginTesla(const edm::EventSetup& c);
  /** gives bending radius in magnetic field, 
   *  pT in GeV, magnetic field taken at (0,0,0) 
   */
  template <typename T>
    T bendingRadius(T pt,const edm::EventSetup& iSetup) {return pt*fieldInInvGev(iSetup);}

  /** gives transverse curvature (=1/radius of curvature) in magnetic field, 
   *  pT in GeV, magnetic field taken at (0,0,0) 
   */
  template <typename T>
  T curvature(T InversePt,const edm::EventSetup& iSetup) {return InversePt/fieldInInvGev(iSetup);}

  /** inverse pt from curvature **/
  template <typename T>
    T inversePt (T curvature,const edm::EventSetup& iSetup) {return curvature*fieldInInvGev(iSetup);}


  /** distance between stright line propagation and helix
   *  r_stright_line = radius+longitudinalBendingCorrection(radius,pt)
   */
  double longitudinalBendingCorrection( double radius, double pt,const edm::EventSetup& iSetup); 

  /** 
   *  initialize TimingReport item in a way used in TkTrackingRegions. 
   *  Assigned a name and activate by .orcarc if 
   *  TkTrackingRegions:timerLevel is >= switchOnLevel
   */  
  //  TimingReport::Item * initTiming(string name, int switchOnLevel); 

}
 
#endif

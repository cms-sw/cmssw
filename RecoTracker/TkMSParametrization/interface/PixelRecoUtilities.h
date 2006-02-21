#ifndef PixelRecoUtilities_H
#define PixelRecoUtilities_H
#include "Geometry/Vector/interface/GlobalVector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <string>
using namespace std;
/** \namespace PixelRecoUtilities
 *  Small utility funcions used during seed generation 
 */


namespace PixelRecoUtilities {

  /** Magnetic field intensity in units of cm/GeV.
   *  The value is cached in a static variable, so the actual
   *  MagneticField is accessed only once.
   */
  float fieldInInvGev();
  //  void MaginTesla(const edm::EventSetup& c);
  /** gives bending radius in magnetic field, 
   *  pT in GeV, magnetic field taken at (0,0,0) 
   */
  template <typename T>
  T bendingRadius(T pt) {return pt*fieldInInvGev();}

  /** gives transverse curvature (=1/radius of curvature) in magnetic field, 
   *  pT in GeV, magnetic field taken at (0,0,0) 
   */
  template <typename T>
  T curvature(T InversePt) {return InversePt/fieldInInvGev();}

  /** inverse pt from curvature **/
  template <typename T>
    T inversePt (T curvature) {return curvature*fieldInInvGev();}


  /** distance between stright line propagation and helix
   *  r_stright_line = radius+longitudinalBendingCorrection(radius,pt)
   */
  double longitudinalBendingCorrection( double radius, double pt); 

  /** 
   *  initialize TimingReport item in a way used in TkTrackingRegions. 
   *  Assigned a name and activate by .orcarc if 
   *  TkTrackingRegions:timerLevel is >= switchOnLevel
   */  
  //  TimingReport::Item * initTiming(string name, int switchOnLevel); 

}
 
#endif

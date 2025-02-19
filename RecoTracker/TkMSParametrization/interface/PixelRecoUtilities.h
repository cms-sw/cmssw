#ifndef PixelRecoUtilities_H
#define PixelRecoUtilities_H
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <string>

/** \namespace PixelRecoUtilities
 *  Small utility funcions used during seed generation 
 */


namespace PixelRecoUtilities {

  /** The magnetic field at 0,0,0
   */
  struct FieldAt0 {
    FieldAt0(const edm::EventSetup& es);
    float fieldInInvGev;
  };

  /** Magnetic field intensity in units of cm/GeV.
   *  The value is cached in a static variable, so the actual
   *  MagneticField is accessed only once.
   */
  inline float fieldInInvGev(const edm::EventSetup& iSetup) {
    static  FieldAt0 fieldAt0(iSetup);
    return fieldAt0.fieldInInvGev;
  }
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
  inline double longitudinalBendingCorrection( double radius, double pt,const edm::EventSetup& iSetup) {
    double invCurv = bendingRadius(pt,iSetup);
    if ( invCurv == 0. ) return 0.;
    return  radius/6. * radius*radius/(2.*invCurv * 2.*invCurv);
  }


}
 
#endif

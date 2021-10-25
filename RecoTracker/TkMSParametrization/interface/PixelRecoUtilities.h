#ifndef PixelRecoUtilities_H
#define PixelRecoUtilities_H
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "MagneticField/Engine/interface/MagneticField.h"

/** \namespace PixelRecoUtilities
 *  Small utility funcions used during seed generation 
 */

namespace PixelRecoUtilities {
  /** gives bending radius in magnetic field, 
   *  pT in GeV, magnetic field taken at (0,0,0) 
   */
  template <typename T>
  T bendingRadius(T pt, const MagneticField& field) {
    return pt * field.inverseBzAtOriginInGeV();
  }

  /** gives transverse curvature (=1/radius of curvature) in magnetic field, 
   *  pT in GeV, magnetic field taken at (0,0,0) 
   */
  template <typename T>
  T curvature(T InversePt, const MagneticField& field) {
    return InversePt / field.inverseBzAtOriginInGeV();
  }

  /** inverse pt from curvature **/
  template <typename T>
  T inversePt(T curvature, const MagneticField& field) {
    return curvature * field.inverseBzAtOriginInGeV();
  }

  /** distance between stright line propagation and helix
   *  r_stright_line = radius+longitudinalBendingCorrection(radius,pt)
   */
  inline double longitudinalBendingCorrection(double radius, double pt, const MagneticField& field) {
    double invCurv = bendingRadius(pt, field);
    if (invCurv == 0.)
      return 0.;
    return radius / 6. * radius * radius / (2. * invCurv * 2. * invCurv);
  }

}  // namespace PixelRecoUtilities

#endif

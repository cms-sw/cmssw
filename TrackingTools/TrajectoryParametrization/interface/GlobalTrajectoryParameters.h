#ifndef _TRACKER_GLOBALTRAJECTORYPARAMETERS_H_
#define _TRACKER_GLOBALTRAJECTORYPARAMETERS_H_

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/TrajectoryState/interface/TrackCharge.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

class MagneticField;
/** Class providing access to a set of relevant parameters of a trajectory
 *  in the global, Cartesian frame. The basic data members used to calculate
 *  these parameters are the charge and global position and momentum.
 */

class GlobalTrajectoryParameters {
public:
// construct
  GlobalTrajectoryParameters() :
    theField(nullptr), 
    theX(), theP(), 
    theCharge(0)
  {}  // we must initialize cache to non-NAN to avoid FPE

  /** Constructing class from global position, global momentum and charge.
   */

  GlobalTrajectoryParameters(
                            const GlobalPoint& aX,
                            const GlobalVector& aP,
                            TrackCharge aCharge,
                            const MagneticField* fieldProvider) :
    theField(fieldProvider),
    theX(aX), theP(aP),
    theCharge(aCharge)
    {setCache();}

  GlobalTrajectoryParameters(const GlobalPoint& aX,
                             const GlobalVector& aP,
                             TrackCharge aCharge,
                             const MagneticField* fieldProvider,
                             GlobalVector fieldValue):
    theField(fieldProvider),
    theX(aX), theP(aP),
    cachedMagneticField(fieldValue),
    theCharge(aCharge)
    {}



  /** Constructing class from global position, direction (unit length) 
   *  and transverse curvature. The fourth int argument is dummy, 
   *  it serves only to distinguish
   *  this constructor from the one above.
   */
  GlobalTrajectoryParameters(const GlobalPoint& aX,
                             const GlobalVector& direction,
                             float transverseCurvature, int, 
			     const MagneticField* fieldProvider);


  GlobalTrajectoryParameters(const GlobalPoint& aX,
                             const GlobalVector& direction,
                             float transverseCurvature, int,
                             const MagneticField* fieldProvider,  
                             GlobalVector fieldValue);


  /** Global position.
   */
  
  GlobalPoint position() const {
    return theX;
  }

  /** Global momentum.
   */
  
  GlobalVector momentum() const {
    return theP;
  }

  GlobalVector direction() const {
    return theP.unit();
  }

  /** Charge q of particle, either +1 or -1.
   */

  TrackCharge charge() const {
    return theCharge;
  }

  /** Charge divided by (magnitude of) momentum, i.e. q/p.
   */  

  float signedInverseMomentum() const {
    return theCharge/theP.mag();
  }

  /** Charge divided by transverse momentum, i.e. q/p_T.
   */ 

  float signedInverseTransverseMomentum() const {
    return theCharge/theP.perp();
  }

  /** Transverse curvature kappa (which is the inverse radius of curvature in the transverse plane) 
   *  in cm^{-1}. Sign convention is such that positive kappa means
   *  counterclockwise rotation of the track with respect to the global z-axis.
   */

  float transverseCurvature() const {
    return -2.99792458e-3f * signedInverseTransverseMomentum() * cachedMagneticField.z();
  }

  /** Vector whose first three elements are the global position coordinates and
   *  whose last three elements are the global momentum coordinates.
   */

  AlgebraicVector6 vector() const {
    return AlgebraicVector6(theX.x(),theX.y(),theX.z(),theP.x(),theP.y(),theP.z());
  }

 
  GlobalVector magneticFieldInInverseGeV( const GlobalPoint& x) const; 
  GlobalVector magneticFieldInInverseGeV() const {
    return 2.99792458e-3f * cachedMagneticField;
  }

  GlobalVector magneticFieldInTesla() const {
    return cachedMagneticField;
  }

  const MagneticField& magneticField() const {return *theField;}

private:
  void setCache();

private:
  const MagneticField* theField;
  GlobalPoint theX;
  GlobalVector theP;
  GlobalVector cachedMagneticField;
  signed char  theCharge;

};

#endif

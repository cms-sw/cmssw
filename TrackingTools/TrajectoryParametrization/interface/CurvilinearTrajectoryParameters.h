#ifndef _TRACKER_CURVILINEARTRAJECTORYPARAMETERS_H_
#define _TRACKER_CURVILINEARTRAJECTORYPARAMETERS_H_


#include "DataFormats/TrajectoryState/interface/TrackCharge.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include <cmath>



/** \class CurvilinearTrajectoryParameters
 * Class providing access to a set of relevant parameters of a trajectory in a Curvilinear frame. The set consists of the following paramters: \
 * q/p: charged particles: charge(plus or minus one) divided by magnitude of momentum
 *      neutral particles: inverse magnitude of momentum
 * lambda: the helix dip angle (pi/2 minus theta(polar angle)), defined in the global frame
 * phi: the angle of inclination with the global x-axis in the transverse (global xy) plane
 * xT: transverse position  in the global xy plane and it points left when looking into the direction of the track
 * yT: transverse position that forms a right-handed frame with xT and zT
 *
 * Note that the frame is tangent to the track at the point of definition, with Z_T parallel to the track
*/



class CurvilinearTrajectoryParameters {
 public:
  
  /// default constructor
  
  CurvilinearTrajectoryParameters() {}
  

  /**Constructor from vector of parameters
   *Expects a vector of parameters as defined above. For charged particles he charge will be determined by\ the sign of the first element. For neutral particles the last argument should be false,
   *  in which case the charge of the first element will be neglected.
   *
   */

  CurvilinearTrajectoryParameters(const AlgebraicVector5& v, bool charged = true);

  /**Constructor from vector of parameters
   *Expects a vector of parameters as defined above. For charged particles the charge will be determined by the sign of the first element. For neutral particles the last argument should be false, 
   *  in which case the charge of the first element will be neglected.
   *
   */


  /**Constructor from individual  curvilinear parameters
   *Expects parameters as defined above.
   */

  CurvilinearTrajectoryParameters(double aQbp, double alambda, double aphi, double axT, double ayT, bool charged = true);

  
  /**Constructor from a global vector, global point and track charge
   *
   */
  CurvilinearTrajectoryParameters(const GlobalPoint& aX,const GlobalVector& aP,TrackCharge aCharge);

  /// access to the charge
  TrackCharge charge() const {return theCharge;}

  /// access to the Signed Inverse momentum q/p (zero for neutrals)
  double signedInverseMomentum() const {
    return charge()==0 ? 0. : theQbp;
  }
  
  
  /**Vector of parameters with signed inverse momentum.
   *
   *Vector of parameters as defined above, with the first element q/p.
   */
  AlgebraicVector5 vector() const ;
    

  double Qbp() const {    return theQbp;  }
  double lambda() const {   return thelambda;  }
  double phi() const {    return thephi;  }
  double xT() const {   return thexT;  }
  double yT() const {    return theyT;  }

  bool updateP(double dP);

 private:
  double theQbp;
  double thelambda;
  double thephi;
  double thexT;
  double theyT;

  TrackCharge theCharge;
};


#endif

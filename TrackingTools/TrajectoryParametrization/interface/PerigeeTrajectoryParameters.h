#ifndef PerigeeTrajectoryParameters_H
#define PerigeeTrajectoryParameters_H
/**
 *  Class providing access to the <i> Perigee</i> parameters of a trajectory.
 *  These parameters consist of <BR>
 *  rho : charged particles: transverse curvature (signed) <BR>
 *        neutral particles: inverse magnitude of transverse momentum <BR>
 *  theta, phi,
 *  transverse impact parameter (signed), longitudinal i.p.
 */

#include "DataFormats/TrajectoryState/interface/TrackCharge.h"
#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

class PerigeeTrajectoryParameters
{

public:

  PerigeeTrajectoryParameters() {}



  explicit PerigeeTrajectoryParameters(const AlgebraicVector5 &aVector, bool charged = true):
        theVector(aVector) {
    if ( charged )
      theCharge = theVector[0]>0 ? -1 : 1;
    else
      theCharge = 0;
  }

  PerigeeTrajectoryParameters(double aCurv, double aTheta, double aPhi,
  			      double aTip, double aLip, bool charged = true) {
    theVector[0] = aCurv;
    theVector[1] = aTheta;
    theVector[2] = aPhi;
    theVector[3] = aTip;
    theVector[4] = aLip;

    if ( charged )
      theCharge = aCurv>0 ? -1 : 1;
    else
      theCharge = 0;
  }

  /**
   * The charge
   */

  TrackCharge charge() const {return theCharge;}

  /**
   * The signed transverse curvature
   */

  double transverseCurvature() const {return ((charge()!=0)?theVector[0]:0.);}

  /**
   * The theta angle
   */

  double theta() const {return theVector[1];}

  /**
   * The phi angle
   */

  double phi() const {return theVector[2];}

  /**
   * The (signed) transverse impact parameter
   */

  double transverseImpactParameter() const {return theVector[3];}

  /**
   * The longitudinal impact parameter
   */

  double longitudinalImpactParameter() const {return theVector[4];}

  /**
   * returns the perigee parameters as a vector.
   * The order of the parameters are: <BR>
   *  transverse curvature (signed), theta, phi,
   *  transverse impact parameter (signed), longitudinal i.p.
   */
   const AlgebraicVector5 & vector() const { return theVector;}


private:
  AlgebraicVector5 theVector;
  TrackCharge theCharge;
};
#endif

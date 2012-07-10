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

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

class PerigeeTrajectoryParameters
{

public:

  PerigeeTrajectoryParameters() {}

  explicit PerigeeTrajectoryParameters(const AlgebraicVector &aVector, bool charged = true):
       theCurv(aVector[0]), theTheta(aVector[1]), thePhi(aVector[2]),
       theTip(aVector[3]), theLip(aVector[4]), theVector(asSVector<5>(aVector)), 
       vectorIsAvailable(true)
  {
    if ( charged )
      theCharge = theCurv>0 ? -1 : 1;
    else
      theCharge = 0;
  }

  explicit PerigeeTrajectoryParameters(const AlgebraicVector5 &aVector, bool charged = true):
       theCurv(aVector[0]), theTheta(aVector[1]), thePhi(aVector[2]),
       theTip(aVector[3]), theLip(aVector[4]), theVector(aVector), 
       vectorIsAvailable(true)
  {
    if ( charged )
      theCharge = theCurv>0 ? -1 : 1;
    else
      theCharge = 0;
  }

  PerigeeTrajectoryParameters(double aCurv, double aTheta, double aPhi,
  			      double aTip, double aLip, bool charged = true):
    theCurv(aCurv), theTheta(aTheta), thePhi(aPhi), theTip(aTip), theLip(aLip),
    vectorIsAvailable(false)
  {
    if ( charged )
      theCharge = theCurv>0 ? -1 : 1;
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

  double transverseCurvature() const {return ((charge()!=0)?theCurv:0.);}

  /**
   * The theta angle
   */

  double theta() const {return theTheta;}

  /**
   * The phi angle
   */

  double phi() const {return thePhi;}

  /**
   * The (signed) transverse impact parameter
   */

  double transverseImpactParameter() const {return theTip;}

  /**
   * The longitudinal impact parameter
   */

  double longitudinalImpactParameter() const {return theLip;}

  /**
   * returns the perigee parameters as a vector.
   * The order of the parameters are: <BR>
   *  transverse curvature (signed), theta, phi,
   *  transverse impact parameter (signed), longitudinal i.p.
   */
  const AlgebraicVector  vector_old() const { return asHepVector(theVector); }

  const AlgebraicVector5 & vector() const
  {
    if (!vectorIsAvailable) {
      //theVector = AlgebraicVector5();
      theVector[0] = theCurv;
      theVector[1] = theTheta;
      theVector[2] = thePhi;
      theVector[3] = theTip;
      theVector[4] = theLip;
      vectorIsAvailable = true;
    }
    return theVector;
  }


private:
  double theCurv, theTheta, thePhi, theTip, theLip;
  TrackCharge theCharge;
  mutable AlgebraicVector5 theVector;
  mutable bool vectorIsAvailable;

};
#endif

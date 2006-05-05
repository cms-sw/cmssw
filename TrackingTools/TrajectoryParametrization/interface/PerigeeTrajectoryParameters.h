#ifndef PerigeeTrajectoryParameters_H
#define PerigeeTrajectoryParameters_H

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "DataFormats/TrackReco/interface/PerigeeParameters.h"

/**
 *  Class providing access to the <i> Perigee</i> parameters of a trajectory.
 *  These parameters consist of <BR>
 *  rho : charged particles: transverse curvature (signed) <BR>
 *        neutral particles: inverse magnitude of transverse momentum <BR>
 *  theta, phi,
 *  transverse impact parameter (signed), longitudinal i.p.
 */

class PerigeeTrajectoryParameters
{

public:

  PerigeeTrajectoryParameters() {}

  explicit PerigeeTrajectoryParameters(AlgebraicVector aVector, double aPT, 
  				bool charged = true):
       theCurv(aVector[0]), theTheta(aVector[1]), thePhi(aVector[2]),
       theTip(aVector[3]), theLip(aVector[4]), pT(aPT), theVector(aVector), 
       vectorIsAvailable(true)
  {
    if ( charged )
      theCharge = theCurv>0 ? -1 : 1;
    else
      theCharge = 0;
  }

  PerigeeTrajectoryParameters(double aCurv, double aTheta, double aPhi,
  			      double aTip, double aLip, double aPT, bool charged = true):
    theCurv(aCurv), theTheta(aTheta), thePhi(aPhi), theTip(aTip), theLip(aLip),
    pT(aPT), vectorIsAvailable(false)
  {
    if ( charged )
      theCharge = theCurv>0 ? -1 : 1;
    else
      theCharge = 0;
  }

  PerigeeTrajectoryParameters(const reco::perigee::Parameters & perigeePar) :
    theCurv(perigeePar.transverseCurvature()), theTheta(perigeePar.theta()),
    thePhi(perigeePar.phi0()), theTip(perigeePar.d0()), theLip(perigeePar.dz()),
    pT(perigeePar.pt()), vectorIsAvailable(false)
  {
    theCharge = theCurv>0 ? -1 : 1;
  }

  operator reco::perigee::Parameters() const
  {
    return reco::perigee::Parameters(theCurv, theTheta, thePhi, theTip, theLip, pT);
  }

  /**
   * The charge
   */

  TrackCharge charge() const {return theCharge;}

  /**
   * The transverse momentum
   */

  double pt() const {return pT;}

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

  const AlgebraicVector & vector() const
  {
    if (!vectorIsAvailable) {
      theVector = AlgebraicVector(5);
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
  double theCurv, theTheta, thePhi, theTip, theLip, pT;
  TrackCharge theCharge;
  mutable AlgebraicVector theVector;
  mutable bool vectorIsAvailable;

};
#endif

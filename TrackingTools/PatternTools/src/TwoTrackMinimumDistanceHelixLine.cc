#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistanceHelixLine.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>

using namespace std;

bool TwoTrackMinimumDistanceHelixLine::updateCoeffs()
{
  bool isFirstALine = firstGTP->charge() == 0. || firstGTP->magneticField().inTesla(firstGTP->position()).z() == 0.;
  bool isSecondALine = secondGTP->charge() == 0. || secondGTP->magneticField().inTesla(secondGTP->position()).z() == 0.;
  if (isFirstALine && !isSecondALine ) {
    theL= firstGTP;
    theH= secondGTP;
  } else if (!isFirstALine && isSecondALine) {
    theH= firstGTP;
    theL= secondGTP;
  } else {
    edm::LogWarning ("TwoTrackMinimumDistanceHelixLine")
      << "Error in track charge: "
      << "One of the tracks has to be charged, and the other not." << endl
      << "Track Charges: "<<firstGTP->charge() << " and " <<secondGTP->charge();
    return true;
  }

  Hn = theH->momentum().mag();
  Ln = theL->momentum().mag();

  if ( Hn == 0. || Ln == 0. )
  {
    edm::LogWarning ("TwoTrackMinimumDistanceHelixLine")
      << "Momentum of input trajectory is zero.";
    return true;
  };

  GlobalPoint lOrig = theL->position();
  GlobalPoint hOrig = theH->position();
  posDiff = GlobalVector((lOrig - hOrig).basicVector());
  X = posDiff.x();
  Y = posDiff.y();
  Z = posDiff.z();
  theLp = theL->momentum();
  px = theLp.x(); px2 = px*px;
  py = theLp.y(); py2 = py*py;
  pz = theLp.z(); pz2 = pz*pz;
  
  const double Bc2kH = theH->magneticField().inTesla(hOrig).z() * 2.99792458e-3;
//   MagneticField::inInverseGeV ( hOrig ).z();

  if ( Bc2kH == 0. )
  {
    edm::LogWarning ("TwoTrackMinimumDistanceHelixLine")
      << "Magnetic field at point " << hOrig << " is zero.";
    return true;
  };

  theh= - Hn / (theH->charge() * Bc2kH ) *
     sqrt( 1 - ( ( (theH->momentum().z()*theH->momentum().z()) / (Hn*Hn) )));

  thetanlambdaH = - theH->momentum().z() / ( theH->charge() * Bc2kH * theh);

  thePhiH0 = theH->momentum().phi();
  thesinPhiH0= sin(thePhiH0);
  thecosPhiH0= cos(thePhiH0);

  aa = (X + theh*thesinPhiH0)*(py2 + pz2) - px*(py*Y + pz*Z);
  bb = (Y - theh*thecosPhiH0)*(px2 + pz2) - py*(px*X + pz*Z);
  cc = pz*theh*thetanlambdaH;
  dd = theh* px *py;
  ee = theh*(px2 - py2);
  ff = (px2 + py2)*theh*thetanlambdaH*thetanlambdaH;

  baseFct = thetanlambdaH * (Z*(px2+py2) - pz*(px*X + py*Y));
  baseDer = - ff;
  return false;
}

bool TwoTrackMinimumDistanceHelixLine::oneIteration(
    double & thePhiH, double & fct, double & derivative ) const
{

  double thesinPhiH = sin(thePhiH);
  double thecosPhiH = cos(thePhiH);

	// Fonction of which the root is to be found:

  fct = baseFct;
  fct -= ff*(thePhiH - thePhiH0);
  fct += thecosPhiH * aa;
  fct += thesinPhiH * bb;
  fct += cc*(thePhiH - thePhiH0)*(px * thecosPhiH + py * thesinPhiH);
  fct += cc * (px * (thesinPhiH - thesinPhiH0) - py * (thecosPhiH - thecosPhiH0));
  fct += dd * (thesinPhiH* (thesinPhiH - thesinPhiH0) - 
  		thecosPhiH*(thecosPhiH - thecosPhiH0));
  fct += ee * thecosPhiH * thesinPhiH;

	  // Its derivative:

  derivative = baseDer;
  derivative += - thesinPhiH * aa;
  derivative += thecosPhiH * bb;
  derivative += cc*(thePhiH - thePhiH0)*(py * thecosPhiH - px * thesinPhiH);
  derivative += 2* cc*(px * thecosPhiH + py * thesinPhiH);
  derivative += dd *(4 * thecosPhiH * thesinPhiH - thecosPhiH * thesinPhiH0 - 
			thesinPhiH * thecosPhiH0);
  derivative += ee * (thecosPhiH*thecosPhiH-thesinPhiH*thesinPhiH);

  return false;
}

bool TwoTrackMinimumDistanceHelixLine::calculate(
    const GlobalTrajectoryParameters & theFirstGTP,
    const GlobalTrajectoryParameters & theSecondGTP, const float qual )
{
  pointsUpdated = false;
  firstGTP  = (GlobalTrajectoryParameters *) &theFirstGTP;
  secondGTP = (GlobalTrajectoryParameters *) &theSecondGTP;

  if ( updateCoeffs () )
  {
    return true;
  };

  double fctVal, derVal, dPhiH;
  thePhiH = thePhiH0;
  
  double x1=thePhiH0-M_PI, x2=thePhiH0+M_PI;
  for (int j=1; j<=themaxiter; ++j) { 
    oneIteration(thePhiH, fctVal, derVal);
    dPhiH=fctVal/derVal;
    thePhiH -= dPhiH;
    if ((x1-thePhiH)*(thePhiH-x2) < 0.0) {
      LogDebug ("TwoTrackMinimumDistanceHelixLine")
        << "Jumped out of brackets in root finding. Will be moved closer.";
      thePhiH += (dPhiH*0.8);
    }
    if (fabs(dPhiH) < qual) {return false;}
  }
  LogDebug ("TwoTrackMinimumDistanceHelixLine")
    <<"Number of steps exceeded. Has not converged.";
  return true;
}

double TwoTrackMinimumDistanceHelixLine::firstAngle()  const
{
  if (firstGTP==theL) return theL->momentum().phi();
  else return thePhiH;
}

double TwoTrackMinimumDistanceHelixLine::secondAngle() const
{
  if (secondGTP==theL) return theL->momentum().phi();
  else return thePhiH;
}

pair <GlobalPoint, GlobalPoint> TwoTrackMinimumDistanceHelixLine::points()
    const 
{
  if (!pointsUpdated)finalPoints();
  if (firstGTP==theL) 
    return pair<GlobalPoint, GlobalPoint> (linePoint, helixPoint);
  else return pair<GlobalPoint, GlobalPoint> (helixPoint, linePoint);
}

pair <double, double> TwoTrackMinimumDistanceHelixLine::pathLength() const
{
  if (!pointsUpdated)finalPoints();
  if (firstGTP==theL) 
    return pair<double, double> (linePath, helixPath);
  else return pair<double, double> (helixPath, linePath);
}

void TwoTrackMinimumDistanceHelixLine::finalPoints() const
{
  helixPoint = GlobalPoint (
      theH->position().x() + theh * ( sin ( thePhiH) - thesinPhiH0 ),
      theH->position().y() + theh * ( - cos ( thePhiH) + thecosPhiH0 ),
      theH->position().z() + theh * ( thetanlambdaH * ( thePhiH- thePhiH0 ))
  );
  helixPath = ( thePhiH- thePhiH0 ) * (theh*sqrt(1+thetanlambdaH*thetanlambdaH)) ;

  GlobalVector diff((theL->position() -helixPoint).basicVector());
  tL = ( - diff.dot(theLp)) / (Ln*Ln);
  linePoint = GlobalPoint (
      theL->position().x() + tL * px,
      theL->position().y() + tL * py,
      theL->position().z() + tL * pz );
  linePath = tL * theLp.mag();
  pointsUpdated = true;
}

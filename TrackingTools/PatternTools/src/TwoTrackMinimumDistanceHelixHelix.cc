#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistanceHelixHelix.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

using namespace std;

TwoTrackMinimumDistanceHelixHelix::TwoTrackMinimumDistanceHelixHelix():
theH(nullptr), theG(nullptr), pointsUpdated(false), themaxjump(20),thesingjacI(1./0.1), themaxiter(4)
{ }

TwoTrackMinimumDistanceHelixHelix::~TwoTrackMinimumDistanceHelixHelix() {}

bool TwoTrackMinimumDistanceHelixHelix::updateCoeffs(
    const GlobalPoint & gpH, const GlobalPoint & gpG ) {

  const double Bc2kH = theH->magneticField().inInverseGeV(gpH).z();
  const double Bc2kG = theG->magneticField().inInverseGeV(gpG).z();
  const double Ht= theH->momentum().perp();
  const double Gt= theG->momentum().perp();
  //  thelambdaH=asin ( theH->momentum().z() / Hn );
  
  if ( Ht == 0. || Gt == 0. ) {
    edm::LogWarning ("TwoTrackMinimumDistanceHelixHelix")
      << "transverse momentum of input trajectory is zero.";
    return true;
  };
  
  if ( theH->charge() == 0. || theG->charge() == 0. ) {
    edm::LogWarning ("TwoTrackMinimumDistanceHelixHelix")
      << "charge of input track is zero.";
    return true;
  };
  
  if ( Bc2kG == 0.) {
    edm::LogWarning ("TwoTrackMinimumDistanceHelixHelix")
      << "magnetic field at point " << gpG << " is zero.";
    return true;
  };
  
  if ( Bc2kH == 0. ) {
    edm::LogWarning ("TwoTrackMinimumDistanceHelixHelix")
      << "magnetic field at point " << gpH << " is zero.";
    return true;
  };
  
  theh= Ht / (theH->charge() * Bc2kH );
  thesinpH0= - theH->momentum().y() / ( Ht );
  thecospH0= - theH->momentum().x() / ( Ht );
  thetanlambdaH = - theH->momentum().z() / ( Ht);
  thepH0 = atan2 ( thesinpH0 , thecospH0 );
  
  // thelambdaG=asin ( theG->momentum().z()/( Gn ) );
  
  theg= Gt / (theG->charge() * Bc2kG );
  thesinpG0= - theG->momentum().y() / ( Gt);
  thecospG0= - theG->momentum().x() / (Gt);
  thetanlambdaG = - theG->momentum().z() / ( Gt);
  thepG0 = atan2 ( thesinpG0 , thecospG0 );
  
  thea=theH->position().x() - theG->position().x() + theg * thesinpG0 -
    theh * thesinpH0;
  theb=theH->position().y() - theG->position().y() - theg * thecospG0 +
    theh * thecospH0;
  thec1=  theh * thetanlambdaH * thetanlambdaH;
  thec2= -theg * thetanlambdaG * thetanlambdaG;
  thed1= -theg * thetanlambdaG * thetanlambdaH;
  thed2=  theh * thetanlambdaG * thetanlambdaH;
  thee1= thetanlambdaH * ( theH->position().z() - theG->position().z() -
			   theh * thepH0 * thetanlambdaH + theg * thepG0 * thetanlambdaG );
  thee2= thetanlambdaG * ( theH->position().z() - theG->position().z() -
			   theh * thepH0 * thetanlambdaH + theg * thepG0 * thetanlambdaG );
  return false;
}

bool TwoTrackMinimumDistanceHelixHelix::oneIteration(double & dH, double & dG ) {
  thesinpH=sin(thepH);
  thecospH=cos(thepH);
  thesinpG=sin(thepG);
  thecospG=cos(thepG);
  
  const double A11= theh * ( - thesinpH * ( thea - theg * thesinpG ) +
			     thecospH * ( theb + theg * thecospG ) + thec1);
  if (A11 < 0) { return true; };
  const double A22= -theg * (- thesinpG * ( thea + theh * thesinpH ) +
			     thecospG*( theb - theh * thecospH ) + thec2);
  if (A22 < 0) { return true; };
  const double A12= theh * (-theg * thecospG * thecospH -
			    theg * thesinpH * thesinpG + thed1);
  const double A21= -theg * (theh * thecospG * thecospH +
			     theh * thesinpH * thesinpG + thed2);
  const double detaI = 1./(A11 * A22 - A12 * A21);
  const double z1=theh * ( thecospH * ( thea - theg * thesinpG ) + thesinpH *
			   ( theb + theg*thecospG ) + thec1 * thepH + thed1 * thepG + thee1);
  const double z2=-theg * (thecospG * ( thea + theh * thesinpH ) + thesinpG *
			   ( theb - theh*thecospH ) + thec2 * thepG + thed2 * thepH + thee2);
  
  dH=( z1 * A22 - z2 * A12 ) * detaI;
  dG=( z2 * A11 - z1 * A21 ) * detaI;
  if ( fabs(detaI) > thesingjacI ) { return true; };
  
  thepH -= dH;
  thepG -= dG;
  return false;
}

/*
bool TwoTrackMinimumDistanceHelixHelix::parallelTracks() const {
  return (fabs(theH->momentum().x() - theG->momentum().x()) < .00000001 )
    && (fabs(theH->momentum().y() - theG->momentum().y()) < .00000001 )
    && (fabs(theH->momentum().z() - theG->momentum().z()) < .00000001 )
    && (theH->charge()==theG->charge()) 
    ;
}
*/

bool TwoTrackMinimumDistanceHelixHelix::calculate(
						  const GlobalTrajectoryParameters & G,
						  const GlobalTrajectoryParameters & H, const float qual ) {
  pointsUpdated = false;
  theG= &G;
  theH= &H;
  bool retval=false;
  
  if ( updateCoeffs ( theG->position(), theH->position() ) ){
    finalPoints();
    return true;
  }
  
  thepG = thepG0;
  thepH = thepH0;
  
  int counter=0;
  double pH=0; double pG=0;
  do {
    retval=oneIteration ( pG, pH );
    if ( edm::isNotFinite(pG) || edm::isNotFinite(pH) ) retval=true;
    if ( counter++>themaxiter ) retval=true;
  } while ( (!retval) && ( fabs(pG) > qual || fabs(pH) > qual ));
  if ( fabs ( theg * ( thepG - thepG0 ) ) > themaxjump ) retval=true;
  if ( fabs ( theh * ( thepH - thepH0 ) ) > themaxjump ) retval=true;

  finalPoints();

  return retval;
}


void TwoTrackMinimumDistanceHelixHelix::finalPoints() {
  if (pointsUpdated) return;
  GlobalVector tmpG( sin(thepG) - thesinpG0,
		   - cos(thepG) + thecospG0,
		   thetanlambdaG * ( thepG- thepG0 ) 
		   );
  pointG = theG->position() + theg * tmpG;
  pathG = ( thepG- thepG0 ) * (theg*sqrt(1+thetanlambdaG*thetanlambdaG)) ;

  GlobalVector tmpH( sin(thepH) - thesinpH0,
		   - cos(thepH) + thecospH0,
		   thetanlambdaH * ( thepH- thepH0 ) 
		   );
  pointH = theH->position() + theh * tmpH;
  pathH = ( thepH- thepH0 ) * (theh*sqrt(1+thetanlambdaH*thetanlambdaH)) ;

  pointsUpdated = true;
}


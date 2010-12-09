#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistanceHelixHelix.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

namespace {
  inline GlobalPoint operator - ( const GlobalPoint & a, const GlobalPoint & b ){
    return GlobalPoint ( a.x() - b.x(), a.y() - b.y(), a.z() - b.z() );
  }

  inline GlobalPoint operator + ( const GlobalPoint & a, const GlobalPoint & b ){
    return GlobalPoint ( a.x() + b.x(), a.y() + b.y(), a.z() + b.z() );
  }

  inline GlobalPoint operator / ( const GlobalPoint & a, const double b ){
    return GlobalPoint ( a.x() / b, a.y() / b, a.z() / b );
  }

  inline GlobalPoint operator * ( const GlobalPoint & a, const double b ){
    return GlobalPoint ( a.x() * b, a.y() * b, a.z() * b );
  }

  inline GlobalPoint operator * ( const double b , const GlobalPoint & a ){
    return GlobalPoint ( a.x() * b, a.y() * b, a.z() * b );
  }

  inline double square ( const double s ) { return s*s; }
}

TwoTrackMinimumDistanceHelixHelix::TwoTrackMinimumDistanceHelixHelix():
theH(), theG(), pointsUpdated(false), themaxjump(20),thesingjac(.1), themaxiter(4)
{ }

TwoTrackMinimumDistanceHelixHelix::~TwoTrackMinimumDistanceHelixHelix() {}

bool TwoTrackMinimumDistanceHelixHelix::updateCoeffs(
    const GlobalPoint & gpH, const GlobalPoint & gpG )
{
//  const double Bc2kH = MagneticField::inInverseGeV ( gpH ).z();
  const double Bc2kH = theH->magneticField().inTesla(gpH).z() * 2.99792458e-3;
//  const double Bc2kG = MagneticField::inInverseGeV ( gpG ).z();
  const double Bc2kG = theG->magneticField().inTesla(gpG).z() * 2.99792458e-3;
  const double Hn= theH->momentum().mag();
  const double Gn= theG->momentum().mag();
  thelambdaH=asin ( theH->momentum().z() / Hn );

  if ( Hn == 0. || Gn == 0. )
  {
    edm::LogWarning ("TwoTrackMinimumDistanceHelixHelix")
      << "momentum of input trajectory is zero.";
    return true;
  };

  if ( theH->charge() == 0. || theG->charge() == 0. )
  {
    edm::LogWarning ("TwoTrackMinimumDistanceHelixHelix")
      << "charge of input track is zero.";
    return true;
  };

  if ( Bc2kG == 0. )
  {
    edm::LogWarning ("TwoTrackMinimumDistanceHelixHelix")
      << "magnetic field at point " << gpG << " is zero.";
    return true;
  };

  if ( Bc2kH == 0. )
  {
    edm::LogWarning ("TwoTrackMinimumDistanceHelixHelix")
      << "magnetic field at point " << gpH << " is zero.";
    return true;
  };

  theh= Hn / (theH->charge() * Bc2kH ) *
     sqrt( 1 - square ( ( theH->momentum().z() / Hn )));
  thesinpH0= - theH->momentum().y() / ( theH->charge() * Bc2kH * theh );
  thecospH0= - theH->momentum().x() / ( theH->charge() * Bc2kH * theh );
  thetanlambdaH = - theH->momentum().z() / ( theH->charge() * Bc2kH * theh);
  thepH0 = atan2 ( thesinpH0 , thecospH0 );
  thelambdaG=asin ( theG->momentum().z()/( Gn ) );
  theg= Gn / (theG->charge() * Bc2kG ) *
    sqrt( 1 - square ( ( theG->momentum().z() / Gn )));
  thesinpG0= - theG->momentum().y() / ( theG->charge()* Bc2kG * theg );
  thecospG0= - theG->momentum().x() / ( theG->charge() * Bc2kG * theg );
  thetanlambdaG = - theG->momentum().z() / ( theG->charge() * Bc2kG * theg);

  thepG0 = atan2 ( thesinpG0 , thecospG0 );

  thea=theH->position().x() - theG->position().x() + theg * thesinpG0 -
     theh * thesinpH0;
  theb=theH->position().y() - theG->position().y() - theg * thecospG0 +
     theh * thecospH0;
  thec1= theh * thetanlambdaH * thetanlambdaH;
  thec2= -theg * thetanlambdaG * thetanlambdaG;
  thed1= -theg * thetanlambdaG * thetanlambdaH;
  thed2= theh * thetanlambdaG * thetanlambdaH;
  thee1= thetanlambdaH * ( theH->position().z() - theG->position().z() -
       theh * thepH0 * thetanlambdaH + theg * thepG0 * thetanlambdaG );
  thee2= thetanlambdaG * ( theH->position().z() - theG->position().z() -
       theh * thepH0 * thetanlambdaH + theg * thepG0 * thetanlambdaG );
  return false;
}

bool TwoTrackMinimumDistanceHelixHelix::oneIteration(
    double & dH, double & dG ) const
{
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
  const double deta = A11 * A22 - A12 * A21;
  const double z1=theh * ( thecospH * ( thea - theg * thesinpG ) + thesinpH *
      ( theb + theg*thecospG ) + thec1 * thepH + thed1 * thepG + thee1);
  const double z2=-theg * (thecospG * ( thea + theh * thesinpH ) + thesinpG *
      ( theb - theh*thecospH ) + thec2 * thepG + thed2 * thepH + thee2);
  
  dH=( z1 * A22 - z2 * A12 ) / deta;
  dG=( z2 * A11 - z1 * A21 ) / deta;
  if ( fabs(deta) < thesingjac ) { return true; };

  thepH -= dH;
  thepG -= dG;
  return false;
}

inline bool TwoTrackMinimumDistanceHelixHelix::parallelTracks() const
{
  bool retval=false;
  if (fabs(theH->momentum().x() - theG->momentum().x()) < .00000001 )
  if (fabs(theH->momentum().y() - theG->momentum().y()) < .00000001 )
  if (fabs(theH->momentum().z() - theG->momentum().z()) < .00000001 )
  if (theH->charge()==theG->charge()) retval=true;
  return retval;
}


bool TwoTrackMinimumDistanceHelixHelix::calculate(
    const GlobalTrajectoryParameters & G,
    const GlobalTrajectoryParameters & H, const float qual )
{
  pointsUpdated = false;
  theG= (GlobalTrajectoryParameters *) &G;
  theH= (GlobalTrajectoryParameters *) &H;
  bool retval=false;

  if ( updateCoeffs ( theG->position(), theH->position() ) )
  {
    return true;
  };

  thepG = thepG0;
  thepH = thepH0;

  int counter=0;
  double pH=0; double pG=0;
  do {
    retval=oneIteration ( pG, pH );
    if ( std::isinf(pG) || std::isinf(pH) ) retval=true;
    if ( counter++>themaxiter ) retval=true;
  } while ( (!retval) && ( fabs(pG) > qual || fabs(pH) > qual ));
  if ( fabs ( theg * ( thepG - thepG0 ) ) > themaxjump ) retval=true;
  if ( fabs ( theh * ( thepH - thepH0 ) ) > themaxjump ) retval=true;
  return retval;
}

double TwoTrackMinimumDistanceHelixHelix::firstAngle()  const
{
  return thepG;
}

double TwoTrackMinimumDistanceHelixHelix::secondAngle() const
{
  return thepH;
}

void TwoTrackMinimumDistanceHelixHelix::finalPoints() const
{
  pointG = GlobalPoint (
      theG->position().x() + theg * ( sin ( thepG) - thesinpG0) ,
      theG->position().y() + theg * ( - cos ( thepG) + thecospG0 ),
      theG->position().z() + theg * ( thetanlambdaG * ( thepG- thepG0 ))
  );
  pathG = ( thepG- thepG0 ) * (theg*sqrt(1+thetanlambdaG*thetanlambdaG)) ;

  pointH = GlobalPoint (
      theH->position().x() + theh * ( sin ( thepH) - thesinpH0 ),
      theH->position().y() + theh * ( - cos ( thepH) + thecospH0 ),
      theH->position().z() + theh * ( thetanlambdaH * ( thepH- thepH0 ))
  );
  pathH = ( thepH- thepH0 ) * (theh*sqrt(1+thetanlambdaH*thetanlambdaH)) ;
  pointsUpdated = true;
}

pair <double, double> TwoTrackMinimumDistanceHelixHelix::pathLength() const
{
  if (!pointsUpdated)finalPoints();
  return pair <double, double> ( pathG, pathH);
}

pair <GlobalPoint, GlobalPoint> TwoTrackMinimumDistanceHelixHelix::points()
    const 
{
  if (!pointsUpdated)finalPoints();
  return pair<GlobalPoint, GlobalPoint> (pointG, pointH);
}


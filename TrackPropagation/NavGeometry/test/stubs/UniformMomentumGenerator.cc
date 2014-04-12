#include "TrackPropagation/NavGeometry/test/stubs/UniformMomentumGenerator.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

Basic3DVector<double> UniformMomentumGenerator::operator()() const
{
    double phiMin = -Geom::pi();
    double phiMax =  Geom::pi();
    double thetaMin = 0;
    double thetaMax = Geom::pi();

  double aPhi = CLHEP::RandFlat::shoot(phiMin,phiMax);
  double aTheta = CLHEP::RandFlat::shoot(thetaMin,thetaMax);
  double aP = CLHEP::RandFlat::shoot(thePmin,thePmax);
    
//   cout << "UniformMomentumGenerator: P= " << aP
//        << " theta= " << aTheta << " phi= " << aPhi << endl;

  Basic3DVector<double> result(aP*sin(aTheta)*cos(aPhi),
			       aP*sin(aTheta)*sin(aPhi),
			       aP*cos(aTheta));
  return result;
}

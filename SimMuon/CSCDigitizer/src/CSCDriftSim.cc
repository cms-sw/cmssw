#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimMuon/CSCDigitizer/src/CSCDriftSim.h"
#include "SimMuon/CSCDigitizer/src/CSCDetectorHit.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Transform3D.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include <cmath>
#include <iostream>

static const int N_INTEGRAL_STEPS = 700;

CSCDriftSim::CSCDriftSim() 
: bz(0.), // should make these local variables
  ycell(0.),
  zcell(0.),
  dNdEIntegral(N_INTEGRAL_STEPS, 0.),
  STEP_SIZE(0.01),
  ELECTRON_DIFFUSION_COEFF(0.0161),
  theMagneticField(0)
{
  // just initialize avalanche sim.  There has to be a better
  // way to take the integral of a function!
  double sum = 0.;
  int i;
  for(i = 0; i < N_INTEGRAL_STEPS; ++i) {
    if(i > 1) {
      double xx = STEP_SIZE * (double(i) - 0.5 );
      double dNdE = pow( xx, 0.38) * exp(-1.38*xx);

      sum += dNdE;
    }
    // store this value in the map
    dNdEIntegral[i] = sum;
  }

  // now normalize the whole map
  for(i =  0; i < N_INTEGRAL_STEPS; ++i) {
    dNdEIntegral[i] /= sum;
  }
}


CSCDetectorHit 
CSCDriftSim::getWireHit(const Local3DPoint & pos, const CSCLayer * layer,
                          int nearestWire, const PSimHit & simHit) {

  const CSCChamberSpecs * specs = layer->chamber()->specs();
  const CSCLayerGeometry * geom = layer->geometry();
  HepPoint3D clusterPos(pos.x(), pos.y(), pos.z());
  LogDebug("CSCDriftSim") << "Ionization cluster at: " <<  pos; 
  // set the coordinate system with the x-axis along the nearest wire,
  // with the origin in the center of the chamber, on that wire.
  HepTranslateY3D yShift(-1.*geom->yOfWire(nearestWire));
  HepRotateZ3D rotation(-1.*geom->wireAngle());
  clusterPos = yShift * clusterPos;
  clusterPos = rotation * clusterPos;
  
  GlobalPoint globalPosition = layer->surface().toGlobal(pos);
  assert(theMagneticField != 0);
 
  //  bz = theMagneticField->inTesla(globalPosition).z() * 10.;

  // We need magnetic field in _local_ coordinates
  // Interface now allows access in kGauss directly.
  bz = layer->toLocal( theMagneticField->inKGauss( globalPosition ) ).z();

  // these subroutines label the coordinates in GEANT coords...
  ycell = clusterPos.z() / specs->anodeCathodeSpacing();
  zcell = 2.*clusterPos.y() / specs->wireSpacing();

  LogDebug("CSCDriftSim") << "bz " << bz <<" avgDrift " << avgDrift()
       << " wireAngle " << geom->wireAngle()
       << " ycell " << ycell << " zcell " << zcell;

  double avgPathLength, pathSigma, avgDriftTime, driftTimeSigma;
  static const float B_FIELD_CUT = 15.f;
  if(fabs(bz) < B_FIELD_CUT) {
    avgPathLength  = avgPathLengthLowB();
    pathSigma      = pathSigmaLowB();
    avgDriftTime   = avgDriftTimeLowB();
    driftTimeSigma = driftTimeSigmaLowB();
  }
  else {
    avgPathLength  = avgPathLengthHighB();
    pathSigma      = pathSigmaHighB();
    avgDriftTime   = avgDriftTimeHighB();
    driftTimeSigma = driftTimeSigmaHighB();
  }

  // electron drift path length 
  double pathLength = std::max(RandGaussQ::shoot(avgPathLength, pathSigma), 0.);

  // electron drift distance along the anode wire, including diffusion
  double diffusionSigma = ELECTRON_DIFFUSION_COEFF * sqrt(pathLength);
  double x = clusterPos.x() + RandGaussQ::shoot(avgDrift(), driftSigma())
                            + RandGaussQ::shoot(0., diffusionSigma);

  // electron drift time
  double driftTime  = std::max(RandGaussQ::shoot(avgDriftTime, driftTimeSigma), 0.);

  //@@ Parameters which should be defined outside the code
  // f_att is the fraction of drift electrons lost due to attachment
  static const double f_att = 0.5;
  static const double f_collected = 0.82;

  // Avalanche charge, with fluctuation ('avalancheCharge()' is the fluctuation generator!)
  double charge = avalancheCharge() * f_att * f_collected * specs->gasGain() * e_SI * 1.e15;

  float t = simHit.tof() + driftTime;
  LogDebug("CSCDriftSim") << "CSCDriftSim: tof = " << simHit.tof() << 
      " driftTime = " << driftTime <<
      " MEDH = " << CSCDetectorHit(nearestWire, charge, x, t, &simHit);
  return CSCDetectorHit(nearestWire, charge, x, t, &simHit);
}

// Generate avalanche fluctuation
#include <algorithm>
double CSCDriftSim::avalancheCharge() {
  double returnVal = 0.;
  // pick a random value along the dNdE integral
  double x = RandFlat::shoot();
  size_t i;
  size_t isiz = dNdEIntegral.size();
  /*
  for(i = 0; i < isiz-1; ++i) {
    if(dNdEIntegral[i] > x) break;
  }
  */
  // return position of first element with a value >= x
  std::vector<double>::const_iterator p=lower_bound(dNdEIntegral.begin(),dNdEIntegral.end(),x);
  if (p==dNdEIntegral.end()) i=isiz-1;
  else i = p-dNdEIntegral.begin();
			      

  // now extrapolate between values
  if( i ==  isiz-1 ) {
    //edm::LogInfo("CSCDriftSim") << "Funky integral in CSCDriftSim " << x;
    returnVal = STEP_SIZE * double(i) * dNdEIntegral[i];
  }
  else {
    double x1 = dNdEIntegral[i];
    double x2 = dNdEIntegral[i+1];
    returnVal = STEP_SIZE * (double(i) + (x-x1)/(x2-x1)); 
  }
  LogDebug("CSCDriftSim") << "avalanche fluc " << returnVal << "  " << x ;

  return returnVal;  
}


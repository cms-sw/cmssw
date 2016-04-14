#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPM.h"

#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandFlat.h"

#include <cmath>
#include <cassert>

using std::vector;
//345678911234567892123456789312345678941234567895123456789612345678971234567898
HcalSiPM::HcalSiPM(int nCells, double tau) :
  theCellCount(nCells), theSiPM(nCells,1.), theTauInv(1.0/tau),
  theCrossTalk(0.), theTempDep(0.), theLastHitTime(-1.) {

  assert(theCellCount>0);
  resetSiPM();
}

HcalSiPM::~HcalSiPM() {
}

int HcalSiPM::hitCells(CLHEP::HepRandomEngine* engine, unsigned int photons, unsigned int integral) const {
  //don't need to do zero or negative photons.
  if (photons < 1) return 0;
  if (integral >= theCellCount) return 0;

  //normalize by theCellCount to remove dependency on SiPM size and pixel density.
  if ((theCrossTalk > 0.) && (theCrossTalk < 1.)) {
    CLHEP::RandPoissonQ randPoissonQ(*engine, photons/(1.-theCrossTalk)-photons);
    photons += randPoissonQ.fire();
  }
  double x(photons/double(theCellCount));
  double prehit(integral/double(theCellCount));

  //calculate the width and mean of the distribution for a given x
  double mean(1. - std::exp(-x));
  double width2(std::exp(-x)*(1-(1+x)*std::exp(-x)));

  //you can't hit more than everything.
  if (mean > 1.) mean = 1.;

  //convert back to absolute pixels
  mean *= (1-prehit)*theCellCount;
  width2 *= (1-prehit)*theCellCount;

  double npe;
  while (true) {
    npe = CLHEP::RandGaussQ::shoot(engine, mean, std::sqrt(width2 + (mean*prehit)));
    if ((npe > -0.5) && (npe <= theCellCount-integral))
      return int(npe + 0.5);
  }
}

double HcalSiPM::hitCells(CLHEP::HepRandomEngine* engine, unsigned int pes, double tempDiff,
			  double photonTime) {
  // response to light impulse with pes input photons.  The return is the number
  // of micro-pixels hit.  If a fraction other than 0. is supplied then the
  // micro-pixel doesn't fully discharge.  The tempDiff is the temperature 
  // difference from nominal and is used to modify the relative strength of a
  // hit pixel.  Pixels which are fractionally charged return a fractional
  // number of hit pixels.

  if ((theCrossTalk > 0.) && (theCrossTalk < 1.)) {
    CLHEP::RandPoissonQ randPoissonQ(*engine, pes/(1. - theCrossTalk) - pes);
    pes += randPoissonQ.fire();
  }

  unsigned int pixel;
  double sum(0.), hit(0.);
  for (unsigned int pe(0); pe < pes; ++pe) {
    pixel = CLHEP::RandFlat::shootInt(engine, theCellCount);
    hit = (theSiPM[pixel] < 0.) ? 1.0 :
      (cellCharge(photonTime - theSiPM[pixel]));
    sum += hit*(1 + (tempDiff*theTempDep));
    theSiPM[pixel] = photonTime;
  }

  theLastHitTime = photonTime;

  return sum;
}

double HcalSiPM::totalCharge(double time) const {
  // sum of the micro-pixels.  NP is a fully charged device.
  // 0 is a fullly depleted device.
  double tot(0.), hit(0.);
  for(unsigned int i=0; i<theCellCount; ++i)  {
    hit = (theSiPM[i] < 0.) ? 1. : cellCharge(time - theSiPM[i]);
    tot += hit;
  }
  return tot;
}

// void HcalSiPM::recoverForTime(double time, double dt) {
//   // apply the RC recover model to the pixels for time.  If dt is not
//   // positive then tau/5 will be used for dt.
//   if (dt <= 0.)
//     dt = 1.0/(theTauInv*5.);
//   for (double t = 0; t < time; t += dt) {
//     expRecover(dt);
//   }
// }

void HcalSiPM::setNCells(int nCells) {
  assert(nCells>0);
  theCellCount = nCells;
  theSiPM.resize(nCells);
  resetSiPM();
}

void HcalSiPM::setCrossTalk(double xTalk) {
  // set the cross-talk probability

  if((xTalk < 0) || (xTalk >= 1)) {
    theCrossTalk = 0.;
  } else {
    theCrossTalk = xTalk;
  }   

}

void HcalSiPM::setTemperatureDependence(double dTemp) {
  // set the temperature dependence
  theTempDep = dTemp;
}

// void HcalSiPM::expRecover(double dt) {
//   // recover each micro-pixel using the RC model.  For this to work well.
//   // dt << tau (typically dt = 0.2*tau or less)
//   double newval;
  
//   for (unsigned int i=0; i<theCellCount; ++i) {
//     if (theSiPM[i] < 0.999) {
//       newval = theSiPM[i] + (1 - theSiPM[i])*dt*theTauInv;
//       theSiPM[i] = (newval > 0.99) ? 1.0 : newval;
//   }
// }

double HcalSiPM::cellCharge(double deltaTime) const {
  if (deltaTime <= 0.) return 0.;
  if (deltaTime > 10./theTauInv) return 1.;
  double result(1. - std::exp(-deltaTime*theTauInv));
  return (result > 0.99) ? 1.0 : result;
}

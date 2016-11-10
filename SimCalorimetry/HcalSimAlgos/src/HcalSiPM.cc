#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPM.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandFlat.h"
#include "TMath.h"
#include <cmath>
#include <cassert>
#include <utility>

using std::vector;
//345678911234567892123456789312345678941234567895123456789612345678971234567898
HcalSiPM::HcalSiPM(int nCells, double tau) :
  theCellCount(nCells), theSiPM(nCells,1.),
  theCrossTalk(0.), theTempDep(0.), theLastHitTime(-1.), nonlin(0)
{
  setTau(tau);
  assert(theCellCount>0);
  resetSiPM();
}

HcalSiPM::~HcalSiPM() {
}

//================================================================================
//implementation of Borel-Tanner distribution
double HcalSiPM::Borel(unsigned int n, double lambda, unsigned int k){
  if(n<k) return 0;
  double dn = double(n);
  double dk = double(k);
  double dnk = dn-dk;
  double ldn = lambda * dn;
  double logb = -ldn + dnk*log(ldn) - TMath::LnGamma(dnk+1);
  double b=0;
  if (logb >= -20)  { // protect against underflow
    b=(dk/dn);
    if ((n-k)<100)
      b *= (exp(-ldn)*pow(ldn,dnk))/TMath::Factorial(n-k);
    else
      b *= exp( logb );
  }
  return b;
}

const HcalSiPM::cdfpair& HcalSiPM::BorelCDF(unsigned int k){
  // EPSILON determines the min and max # of xtalk cells that can be
  // simulated.
  static const double EPSILON = 1e-6;
  typename cdfmap::const_iterator it;
  it = borelcdfs.find(k);
  if (it == borelcdfs.end()) {
    vector<double> cdf;

    // Find the first n=k+i value for which cdf[i] > EPSILON
    unsigned int i;
    double b=0., sumb=0.;
    for (i=0; ; i++) {
      b = Borel(k+i,theCrossTalk,k);
      sumb += b;
      if (sumb >= EPSILON) break;
    }

    cdf.push_back(sumb);
    unsigned int borelstartn = i;

    // calculate cdf[i]
    for(++i; ; ++i){
      b = Borel(k+i,theCrossTalk,k);
      sumb += b;
      cdf.push_back(sumb);
      if (1-sumb < EPSILON) break;
    }

    it = (borelcdfs.emplace(k,make_pair(borelstartn, cdf))).first;
  }

  return it->second;
}

unsigned int HcalSiPM::addCrossTalkCells(CLHEP::HepRandomEngine* engine,
					 unsigned int in_pes) {
  const cdfpair& cdf = BorelCDF(in_pes);

  double U = CLHEP::RandFlat::shoot(engine);
  std::vector<double>::const_iterator up;
  up= std::lower_bound (cdf.second.cbegin(), cdf.second.cend(), U);

  LogDebug("HcalSiPM") << "cdf size = " << cdf.second.size()
		       << ", U = " << U
		       << ", in_pes = " << in_pes
		       << ", 2ndary_pes = " << (up-cdf.second.cbegin()+cdf.first);

  // returns the number of secondary pes produced
  return (up - cdf.second.cbegin() + cdf.first);
}

//================================================================================

double HcalSiPM::hitCells(CLHEP::HepRandomEngine* engine, unsigned int pes, double tempDiff,
			  double photonTime) {
  // response to light impulse with pes input photons.  The return is the number
  // of micro-pixels hit.  If a fraction other than 0. is supplied then the
  // micro-pixel doesn't fully discharge.  The tempDiff is the temperature 
  // difference from nominal and is used to modify the relative strength of a
  // hit pixel.  Pixels which are fractionally charged return a fractional
  // number of hit pixels.

  if ((theCrossTalk > 0.) && (theCrossTalk < 1.)) 
    pes += addCrossTalkCells(engine, pes);

  // Account for saturation - disabled in lieu of recovery model below
  //pes = nonlin->getPixelsFired(pes);

  //disable saturation/recovery model for bad tau values
  if(theTau<=0) return pes;

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

void HcalSiPM::setNCells(int nCells) {
  assert(nCells>0);
  theCellCount = nCells;
  theSiPM.resize(nCells);
  resetSiPM();
}

void HcalSiPM::setTau(double tau) { 
  theTau = tau;
  if(theTau > 0) theTauInv = 1./theTau;
  else theTauInv = 0;
}

void HcalSiPM::setCrossTalk(double xTalk) {
  // set the cross-talk probability

  double oldCrossTalk = theCrossTalk;

  if((xTalk < 0) || (xTalk >= 1)) {
    theCrossTalk = 0.;
  } else {
    theCrossTalk = xTalk;
  }   

  // Recalculate the crosstalk CDFs
  if (theCrossTalk != oldCrossTalk) {
    borelcdfs.clear();
    if (theCrossTalk > 0)
      for (int k=1; k<=100; k++)
	BorelCDF(k);
  }
}

void HcalSiPM::setTemperatureDependence(double dTemp) {
  // set the temperature dependence
  theTempDep = dTemp;
}

double HcalSiPM::cellCharge(double deltaTime) const {
  if (deltaTime <= 0.) return 0.;
  if (deltaTime*theTauInv > 10.) return 1.;
  double result(1. - std::exp(-deltaTime*theTauInv));
  return (result > 0.99) ? 1.0 : result;
}

void HcalSiPM::setSaturationPars(const std::vector<float>& pars)
{
  if (nonlin)
    delete nonlin;

  nonlin = new HcalSiPMnonlinearity(pars);
}

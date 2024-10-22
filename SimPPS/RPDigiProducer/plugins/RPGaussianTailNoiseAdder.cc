#include "SimPPS/RPDigiProducer/plugins/RPGaussianTailNoiseAdder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandBinomial.h"

#include <cmath>

using namespace std;

RPGaussianTailNoiseAdder::RPGaussianTailNoiseAdder(
    int numStrips, double theNoiseInElectrons, double theStripThresholdInE, CLHEP::HepRandomEngine &eng, int verbosity)
    : numStrips_(numStrips),
      theNoiseInElectrons(theNoiseInElectrons),
      theStripThresholdInE(theStripThresholdInE),
      rndEngine_(eng),
      verbosity_(verbosity) {
  strips_above_threshold_prob_ = std::erfc(theStripThresholdInE / sqrt(2.0) / theNoiseInElectrons) / 2;
}

simromanpot::strip_charge_map RPGaussianTailNoiseAdder::addNoise(const simromanpot::strip_charge_map &theSignal) {
  simromanpot::strip_charge_map the_strip_charge_map;

  // noise on strips with signal:
  for (simromanpot::strip_charge_map::const_iterator i = theSignal.begin(); i != theSignal.end(); ++i) {
    double noise = CLHEP::RandGauss::shoot(&(rndEngine_), 0.0, theNoiseInElectrons);
    the_strip_charge_map[i->first] = i->second + noise;
    if (verbosity_)
      edm::LogInfo("RPDigiProducer") << "noise added to signal strips: " << noise << "\n";
  }

  // noise on the other strips
  int strips_no_above_threshold =
      CLHEP::RandBinomial::shoot(&(rndEngine_), (long)numStrips_, strips_above_threshold_prob_);

  for (int j = 0; j < strips_no_above_threshold; j++) {
    int strip = CLHEP::RandFlat::shootInt(&(rndEngine_), numStrips_);
    if (the_strip_charge_map[strip] == 0) {
      the_strip_charge_map[strip] = 2 * theStripThresholdInE;
      //only binary decision later, no need to simulate the noise precisely,
      //enough to know that it exceeds the threshold
      if (verbosity_)
        edm::LogInfo("RPDigiProducer") << "nonsignal strips noise :" << strip << " " << the_strip_charge_map[strip]
                                       << "\n";
    }
  }

  return the_strip_charge_map;
}

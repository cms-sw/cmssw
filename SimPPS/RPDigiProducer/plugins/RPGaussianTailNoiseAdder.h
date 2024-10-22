#ifndef SimPPS_RPDigiProducer_RP_GAUSSIAN_TAIL_NOISE_ADDER_H
#define SimPPS_RPDigiProducer_RP_GAUSSIAN_TAIL_NOISE_ADDER_H

#include "SimPPS/RPDigiProducer/plugins/RPHitChargeConverter.h"
#include "SimPPS/RPDigiProducer/interface/RPSimTypes.h"

class RPGaussianTailNoiseAdder {
public:
  RPGaussianTailNoiseAdder(int numStrips,
                           double theNoiseInElectrons,
                           double theStripThresholdInE,
                           CLHEP::HepRandomEngine &eng,
                           int verbosity);
  simromanpot::strip_charge_map addNoise(const simromanpot::strip_charge_map &theSignal);

private:
  int numStrips_;
  double theNoiseInElectrons;
  double theStripThresholdInE;
  double strips_above_threshold_prob_;
  CLHEP::HepRandomEngine &rndEngine_;
  int verbosity_;
};

#endif  //SimPPS_RPDigiProducer_RP_GAUSSIAN_TAIL_NOISE_ADDER_H

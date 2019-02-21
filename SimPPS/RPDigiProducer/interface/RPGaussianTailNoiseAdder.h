#ifndef SimPPS_RPDigiProducer_RP_GAUSSIAN_TAIL_NOISE_ADDER_H
#define SimPPS_RPDigiProducer_RP_GAUSSIAN_TAIL_NOISE_ADDER_H

#include "SimPPS/RPDigiProducer/interface/RPHitChargeConverter.h"
#include "SimPPS/RPDigiProducer/interface/RPSimTypes.h"

class RPGaussianTailNoiseAdder
{
  public:
    RPGaussianTailNoiseAdder(int numStrips, double theNoiseInElectrons, 
        double theStripThresholdInE, int verbosity);
    SimRP::strip_charge_map addNoise(
        const SimRP::strip_charge_map &theSignal);
  private:
    SimRP::strip_charge_map the_strip_charge_map_;
    int numStrips_;
    double theNoiseInElectrons_;
    double theStripThresholdInE_;
    double strips_above_threshold_prob_;
    int verbosity_;
};

#endif  //SimPPS_RPDigiProducer_RP_GAUSSIAN_TAIL_NOISE_ADDER_H

#include <vector>
#include <iostream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimPPS/RPDigiProducer/plugins/RPDetDigitizer.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

RPDetDigitizer::RPDetDigitizer(const edm::ParameterSet &params,
                               CLHEP::HepRandomEngine &eng,
                               RPDetId det_id,
                               const edm::EventSetup &iSetup)
    : det_id_(det_id) {
  verbosity_ = params.getParameter<int>("RPVerbosity");
  numStrips_ = RPTopology().DetStripNo();
  theNoiseInElectrons = params.getParameter<double>("RPEquivalentNoiseCharge300um");
  theStripThresholdInE = params.getParameter<double>("RPVFATThreshold");
  noNoise_ = params.getParameter<bool>("RPNoNoise");
  misalignment_simulation_on_ = params.getParameter<bool>("RPDisplacementOn");
  links_persistence_ = params.getParameter<bool>("RPDigiSimHitRelationsPresistence");

  theRPGaussianTailNoiseAdder = std::make_unique<RPGaussianTailNoiseAdder>(
      numStrips_, theNoiseInElectrons, theStripThresholdInE, eng, verbosity_);
  theRPPileUpSignals = std::make_unique<RPPileUpSignals>(params, det_id_);
  theRPVFATSimulator = std::make_unique<RPVFATSimulator>(params, det_id_);
  theRPHitChargeConverter = std::make_unique<RPHitChargeConverter>(params, eng, det_id_);
  theRPDisplacementGenerator = std::make_unique<RPDisplacementGenerator>(params, det_id_, iSetup);
}

void RPDetDigitizer::run(const std::vector<PSimHit> &input,
                         const std::vector<int> &input_links,
                         std::vector<TotemRPDigi> &output_digi,
                         simromanpot::DigiPrimaryMapType &output_digi_links) {
  if (verbosity_)
    LogDebug("RPDetDigitizer ") << det_id_ << " received input.size()=" << input.size() << "\n";
  theRPPileUpSignals->reset();

  bool links_persistence_checked = links_persistence_ && input_links.size() == input.size();

  int input_size = input.size();
  for (int i = 0; i < input_size; ++i) {
    simromanpot::strip_charge_map the_strip_charge_map;
    if (misalignment_simulation_on_)
      the_strip_charge_map = theRPHitChargeConverter->processHit(theRPDisplacementGenerator->displace(input[i]));
    else
      the_strip_charge_map = theRPHitChargeConverter->processHit(input[i]);

    if (verbosity_)
      LogDebug("RPHitChargeConverter ") << det_id_ << " returned hits=" << the_strip_charge_map.size() << "\n";
    if (links_persistence_checked)
      theRPPileUpSignals->add(the_strip_charge_map, input_links[i]);
    else
      theRPPileUpSignals->add(the_strip_charge_map, 0);
  }

  const simromanpot::strip_charge_map &theSignal = theRPPileUpSignals->dumpSignal();
  simromanpot::strip_charge_map_links_type &theSignalProvenance = theRPPileUpSignals->dumpLinks();
  simromanpot::strip_charge_map afterNoise;
  if (noNoise_)
    afterNoise = theSignal;
  else
    afterNoise = theRPGaussianTailNoiseAdder->addNoise(theSignal);

  theRPVFATSimulator->ConvertChargeToHits(afterNoise, theSignalProvenance, output_digi, output_digi_links);
}

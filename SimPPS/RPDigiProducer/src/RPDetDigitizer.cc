#include <vector>
#include <iostream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimPPS/RPDigiProducer/interface/RPDetDigitizer.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"


RPDetDigitizer::RPDetDigitizer(const edm::ParameterSet &params, CLHEP::HepRandomEngine& eng, RPDetId det_id, const edm::EventSetup& iSetup)
  : params_(params), det_id_(det_id)
{
  verbosity_ = params.getParameter<int>("RPVerbosity");
  numStrips = RPTopology().DetStripNo();
  theNoiseInElectrons = params.getParameter<double>("RPEquivalentNoiseCharge300um");
  theStripThresholdInE = params.getParameter<double>("RPVFATThreshold");
  noNoise = params.getParameter<bool>("RPNoNoise");
  misalignment_simulation_on_ = params_.getParameter<bool>("RPDisplacementOn");
  _links_persistence = params.getParameter<bool>("RPDigiSimHitRelationsPresistence");

  theRPGaussianTailNoiseAdder = new RPGaussianTailNoiseAdder(numStrips, 
      theNoiseInElectrons, theStripThresholdInE, verbosity_);
  theRPPileUpSignals = new RPPileUpSignals(params_, det_id_);
  theRPVFATSimulator = new RPVFATSimulator(params_, det_id_);
  theRPHitChargeConverter = new RPHitChargeConverter(params_, eng, det_id_);
  theRPDisplacementGenerator = new RPDisplacementGenerator(params_, det_id_, iSetup);
}

RPDetDigitizer::~RPDetDigitizer()
{
  delete theRPGaussianTailNoiseAdder;
  delete theRPPileUpSignals;
  delete theRPVFATSimulator;
  delete theRPHitChargeConverter;
  delete theRPDisplacementGenerator;
}

void RPDetDigitizer::run(const std::vector<PSimHit> &input, const std::vector<int> &input_links, 
    std::vector<TotemRPDigi> &output_digi, 
    SimRP::DigiPrimaryMapType &output_digi_links) 
{
  if(verbosity_)
    std::cout<<"RPDetDigitizer "<<det_id_<<" received input.size()="<<input.size()<<std::endl;
  theRPPileUpSignals->reset();
  
  bool links_persistence_checked = _links_persistence && input_links.size()==input.size();
  
  int input_size = input.size();
  for (int i=0; i<input_size; ++i)
  {
    SimRP::strip_charge_map the_strip_charge_map;
    if(misalignment_simulation_on_)
      the_strip_charge_map = theRPHitChargeConverter->processHit(
            theRPDisplacementGenerator->Displace(input[i]));
    else
      the_strip_charge_map = theRPHitChargeConverter->processHit(input[i]);
      
    if(verbosity_)
      std::cout<<"RPHitChargeConverter "<<det_id_<<" returned hits="<<the_strip_charge_map.size()<<std::endl;
    if(links_persistence_checked)
      theRPPileUpSignals->add(the_strip_charge_map, input_links[i]);
    else
      theRPPileUpSignals->add(the_strip_charge_map, 0);
  }

  const SimRP::strip_charge_map &theSignal = theRPPileUpSignals->dumpSignal();
  SimRP::strip_charge_map_links_type &theSignalProvenance = theRPPileUpSignals->dumpLinks();
  SimRP::strip_charge_map afterNoise;
  if(noNoise)
    afterNoise = theSignal;
  else
    afterNoise = theRPGaussianTailNoiseAdder->addNoise(theSignal);

  theRPVFATSimulator->ConvertChargeToHits(afterNoise, theSignalProvenance, 
        output_digi, output_digi_links);
}

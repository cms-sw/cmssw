#include "SimPPS/RPDigiProducer/plugins/RPVFATSimulator.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>
#include <iostream>

RPVFATSimulator::RPVFATSimulator(const edm::ParameterSet &params, RPDetId det_id) : params_(params), det_id_(det_id) {
  threshold_ = params.getParameter<double>("RPVFATThreshold");
  dead_strip_probability_ = params.getParameter<double>("RPDeadStripProbability");
  dead_strips_simulation_on_ = params.getParameter<bool>("RPDeadStripSimulationOn");
  strips_no_ = RPTopology().DetStripNo();
  verbosity_ = params.getParameter<int>("RPVerbosity");
  links_persistence_ = params.getParameter<bool>("RPDigiSimHitRelationsPresistence");
}

void RPVFATSimulator::ConvertChargeToHits(const simromanpot::strip_charge_map &signals,
                                          simromanpot::strip_charge_map_links_type &theSignalProvenance,
                                          std::vector<TotemRPDigi> &output_digi,
                                          simromanpot::DigiPrimaryMapType &output_digi_links) {
  for (auto signal : signals) {
    //one threshold per hybrid
    unsigned short strip_no = signal.first;
    if (signal.second > threshold_ &&
        (!dead_strips_simulation_on_ || dead_strips_.find(strip_no) == dead_strips_.end())) {
      output_digi.push_back(TotemRPDigi(strip_no));
      if (links_persistence_) {
        output_digi_links.push_back(theSignalProvenance[strip_no]);
        if (verbosity_) {
          edm::LogInfo("RPVFatSimulator") << " digi links size=" << theSignalProvenance[strip_no].size() << "\n";
          for (auto &u : theSignalProvenance[strip_no]) {
            edm::LogInfo("RPVFatSimulator")
                << " digi: particle=" << u.first << " energy [electrons]=" << u.second << "\n";
          }
        }
      }
    }
  }

  if (verbosity_) {
    for (auto &i : output_digi) {
      edm::LogInfo("RPVFATSimulator") << i.stripNumber() << "\n";
    }
  }
}

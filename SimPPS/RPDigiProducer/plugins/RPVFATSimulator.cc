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
  for (simromanpot::strip_charge_map::const_iterator i = signals.begin(); i != signals.end(); ++i) {
    //one threshold per hybrid
    unsigned short strip_no = i->first;
    if (i->second > threshold_ && (!dead_strips_simulation_on_ || dead_strips_.find(strip_no) == dead_strips_.end())) {
      output_digi.push_back(TotemRPDigi(strip_no));
      if (links_persistence_) {
        output_digi_links.push_back(theSignalProvenance[strip_no]);
        if (verbosity_) {
          edm::LogInfo("RPVFatSimulator") << " digi links size=" << theSignalProvenance[strip_no].size() << "\n";
          for (unsigned int u = 0; u < theSignalProvenance[strip_no].size(); ++u) {
            edm::LogInfo("RPVFatSimulator")
                << " digi: particle=" << theSignalProvenance[strip_no][u].first
                << " energy [electrons]=" << theSignalProvenance[strip_no][u].second << "\n";
          }
        }
      }
    }
  }

  if (verbosity_) {
    for (unsigned int i = 0; i < output_digi.size(); ++i) {
      edm::LogInfo("RPVFATSimulator") << output_digi[i].stripNumber() << "\n";
    }
  }
}

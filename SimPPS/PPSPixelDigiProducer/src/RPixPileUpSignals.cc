#include "SimPPS/PPSPixelDigiProducer/interface/RPixPileUpSignals.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

RPixPileUpSignals::RPixPileUpSignals(const edm::ParameterSet &params, uint32_t det_id) : det_id_(det_id) {
  links_persistence_ = params.getParameter<bool>("CTPPSPixelDigiSimHitRelationsPersistence");
  verbosity_ = params.getParameter<int>("RPixVerbosity");
}

void RPixPileUpSignals::reset() {
  the_pixel_charge_piled_up_map_.clear();
  the_pixel_charge_piled_up_map_links_.clear();
}

void RPixPileUpSignals::add(const std::map<unsigned short, double> &charge_induced, int PSimHitIndex) {
  for (std::map<unsigned short, double>::const_iterator i = charge_induced.begin(); i != charge_induced.end(); ++i) {
    the_pixel_charge_piled_up_map_[i->first] += i->second;
    if (links_persistence_ && i->second > 0) {
      the_pixel_charge_piled_up_map_links_[i->first].push_back(std::pair<int, double>(PSimHitIndex, i->second));
      if (verbosity_) {
        edm::LogInfo("RPixPileUpSignals") << "Det id=" << det_id_ << " pixel=" << i->first << " charge=" << i->second;
      }
    }
  }
}

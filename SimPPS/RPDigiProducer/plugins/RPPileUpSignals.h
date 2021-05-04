#ifndef SimPPS_RPDigiProducer_RP_PILE_UP_SIGNALS_H
#define SimPPS_RPDigiProducer_RP_PILE_UP_SIGNALS_H

#include <map>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimPPS/RPDigiProducer/interface/RPSimTypes.h"

class RPPileUpSignals {
public:
  RPPileUpSignals(const edm::ParameterSet &params, RPDetId det_id);
  void reset();
  void add(const simromanpot::strip_charge_map &charge_induced, int PSimHitIndex);
  inline const simromanpot::strip_charge_map &dumpSignal() { return the_strip_charge_piled_up_map_; }
  inline simromanpot::strip_charge_map_links_type &dumpLinks() { return the_strip_charge_piled_up_map_links_; }

private:
  simromanpot::strip_charge_map the_strip_charge_piled_up_map_;
  simromanpot::strip_charge_map_links_type the_strip_charge_piled_up_map_links_;
  bool links_persistence_;
  RPDetId det_id_;
  bool verbosity_;
};

#endif  //SimPPS_RPDigiProducer_RP_PILE_UP_SIGNALS_H

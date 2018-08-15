#ifndef RP_VFAT_SIMULATION_H
#define RP_VFAT_SIMULATION_H

#include <set>

#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "DataFormats/CTPPSDigi/interface/RPDetTrigger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimPPS/RPDigiProducer/interface/RPSimTypes.h"

class RPVFATSimulator
{
  public:
    RPVFATSimulator(const edm::ParameterSet &params, RPDetId det_id);
    void ConvertChargeToHits(const SimRP::strip_charge_map &signals, 
          SimRP::strip_charge_map_links_type &theSignalProvenance, 
          std::vector<TotemRPDigi> &output_digi, 
          std::vector<RPDetTrigger> &output_trig, 
          SimRP::DigiPrimaryMapType &output_digi_links, 
          SimRP::TriggerPrimaryMapType &output_trig_links);
  private:
    typedef std::set<unsigned short, std::less<unsigned short> > dead_strip_set;
    void SetDeadStrips();
    const edm::ParameterSet &params_;
    RPDetId det_id_;
    double dead_strip_probability_;
    bool dead_strips_simulation_on_;
    dead_strip_set dead_strips_;
    int verbosity_;
    
    int trigger_mode_;
    unsigned short strips_no_;
    double threshold_;
    short strips_per_section_;
    SimRP::TriggerContainer the_trig_cont_;
    SimRP::TriggerContainerLinkMap the_trig_cont_links_;
    bool links_persistence_;
};

#endif


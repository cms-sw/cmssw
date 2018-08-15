
#include "SimPPS/RPDigiProducer/interface/RPPileUpSignals.h"
#include <iostream>


RPPileUpSignals::RPPileUpSignals(const edm::ParameterSet & params, RPDetId det_id)
 : det_id_(det_id)
{
  links_persistence_ = params.getParameter<bool>("RPDigiSimHitRelationsPresistence");
  verbosity_ = params.getParameter<int>("RPVerbosity");
}

void RPPileUpSignals::reset()
{
  the_strip_charge_piled_up_map_.clear();
  the_strip_charge_piled_up_map_links_.clear();
}

void RPPileUpSignals::add(const SimRP::strip_charge_map &charge_induced, int PSimHitIndex)
{
  for(SimRP::strip_charge_map::const_iterator i=charge_induced.begin();
      i!=charge_induced.end(); ++i)
  {
    the_strip_charge_piled_up_map_[i->first]+=i->second;
    if(links_persistence_ && i->second>0)
    {
      the_strip_charge_piled_up_map_links_[i->first].push_back(std::pair<int, double>(PSimHitIndex, i->second));
      if(verbosity_)
      {
        std::cout<<"Det id="<<det_id_<<" strip="<<i->first<<" charge="<<i->second<<std::endl;
      }
    }
  }
}

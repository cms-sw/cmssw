#include "SimPPS/RPDigiProducer/interface/RPVFATSimulator.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"
#include <vector>
#include "TRandom.h"
#include <iostream>


RPVFATSimulator::RPVFATSimulator(const edm::ParameterSet &params, RPDetId det_id)
 : params_(params), det_id_(det_id)
{
  trigger_mode_ = params_.getParameter<int>("RPVFATTriggerMode");
  threshold_ = params.getParameter<double>("RPVFATThreshold");
  dead_strip_probability_ = params.getParameter<double>("RPDeadStripProbability");
  dead_strips_simulation_on_ = params.getParameter<bool>("RPDeadStripSimulationOn");
  strips_no_ = RPTopology().DetStripNo();
  verbosity_ = params.getParameter<int>("RPVerbosity");
  links_persistence_ = params.getParameter<bool>("RPDigiSimHitRelationsPresistence");
  
  if(dead_strips_simulation_on_)
    SetDeadStrips();

  //trigger_mode: 0=no trigger, 1=one sector per chip, 2=4 sectors, 3=8 sectors, 4=gem mode (not implemented)
  switch(trigger_mode_)
  {
    case 0: 
      strips_per_section_ = 0;
      break;
    case 1:
      strips_per_section_ = 128; //since we have 4 chips
      break;
    case 2:
      strips_per_section_ = 128/4; //since we have 4 chips
      break;
    case 3:
      strips_per_section_ = 128/8; //since we have 4 chips
      break;
    default:
      strips_per_section_ = 0;
  }
}

void RPVFATSimulator::ConvertChargeToHits(const SimRP::strip_charge_map &signals, 
    SimRP::strip_charge_map_links_type &theSignalProvenance, 
    std::vector<TotemRPDigi> &output_digi, std::vector<RPDetTrigger> &output_trig, 
    SimRP::DigiPrimaryMapType &output_digi_links, 
    SimRP::TriggerPrimaryMapType &output_trig_links)
{
  the_trig_cont_.clear();
  the_trig_cont_links_.clear();
  for(SimRP::strip_charge_map::const_iterator i=signals.begin(); 
        i!=signals.end(); ++i)
  {
    //one threshold per hybrid
    unsigned short strip_no = i->first;
    if(i->second > threshold_ && (!dead_strips_simulation_on_ 
          || dead_strips_.find(strip_no)==dead_strips_.end() ))
    {
      output_digi.push_back(TotemRPDigi(strip_no));
      if(links_persistence_)
      {
        output_digi_links.push_back(theSignalProvenance[strip_no]);
        if(verbosity_)
        {
          std::cout<<"digi links size="<<theSignalProvenance[strip_no].size()<<std::endl;
          for(unsigned int u=0; u<theSignalProvenance[strip_no].size(); ++u)
          {
            std::cout<<"   digi: particle="<<theSignalProvenance[strip_no][u].first<<" energy [electrons]="<<theSignalProvenance[strip_no][u].second<<std::endl;
          }
        }
      }
      
      if(strips_per_section_)
      {
        int det_trig_section = strip_no/strips_per_section_;
        the_trig_cont_.insert(det_trig_section);
        
        if(links_persistence_)
        {
          std::vector< std::pair<int, double> >::const_iterator j=theSignalProvenance[strip_no].begin();
          std::vector< std::pair<int, double> >::const_iterator end=theSignalProvenance[strip_no].end();
          for(; j!=end; ++j)
          {
            the_trig_cont_links_[det_trig_section][j->first]+=j->second;
          }
        }
      }
    }
  }
  
  for(SimRP::TriggerContainer::const_iterator j=the_trig_cont_.begin();
      j!=the_trig_cont_.end(); ++j)
  {
    output_trig.push_back(RPDetTrigger(det_id_, *j));
    if(links_persistence_)
    {
      std::map<int, double>::const_iterator k=the_trig_cont_links_[*j].begin();
      std::map<int, double>::const_iterator end=the_trig_cont_links_[*j].end();
      std::vector<std::pair<int, double> > links_vector(k, end);
      output_trig_links.push_back(links_vector);
      if(verbosity_)
      {
        std::cout<<"trigger links size="<<links_vector.size()<<std::endl;
        for(unsigned int u=0; u<links_vector.size(); ++u)
        {
          std::cout<<"   trigger: particle="<<links_vector[u].first<<" energy [electrons]="<<links_vector[u].second<<std::endl;
        }
        std::cout<<std::endl;
      }
    }
  }
  if(verbosity_)
  {
    for(unsigned int i=0; i<output_digi.size(); ++i)
    {
      std::cout<<"VFAT Simulator "
          <<output_digi[i].getStripNumber()<<std::endl;
    }
  }
}

void RPVFATSimulator::SetDeadStrips()
{
  dead_strips_.clear();
  double dead_strip_number = gRandom->Binomial(strips_no_, dead_strip_probability_);
  
  for(int i=0; i<dead_strip_number; ++i)
  {
    dead_strips_.insert(gRandom->Integer(strips_no_));
  }
}


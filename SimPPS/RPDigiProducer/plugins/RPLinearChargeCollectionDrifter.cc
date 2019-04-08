#include "SimPPS/RPDigiProducer/plugins/RPLinearChargeCollectionDrifter.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <vector>

RPLinearChargeCollectionDrifter::RPLinearChargeCollectionDrifter
      (const edm::ParameterSet &params, RPDetId det_id)
{
  verbosity_ = params.getParameter<int>("RPVerbosity");
  GeV_per_electron_ = params.getParameter<double>("RPGeVPerElectron");
  charge_cloud_sigmas_vect_ = params.getParameter< std::vector<double> >("RPInterStripSmearing");
  det_thickness_ = RPTopology().DetThickness();
  det_id_=det_id;
}

simRP::charge_induced_on_surface RPLinearChargeCollectionDrifter::Drift
      (const simRP::energy_path_distribution &energy_deposition)
{
  simRP::charge_induced_on_surface temp_;
  temp_.resize(energy_deposition.size());
  for(unsigned int i=0; i<energy_deposition.size(); i++)
  {
    temp_[i].Position() = LocalPoint(energy_deposition[i].X(), energy_deposition[i].Y());
    temp_[i].Sigma() = getSigma_(energy_deposition[i].Z());  //befor charge_cloud_sigma_ used, now a vector of sigmas;
    temp_[i].Charge() = energy_deposition[i].Energy()/GeV_per_electron_;
    if(verbosity_)
    {
      edm::LogInfo("RPLinearChargeCollectionDrifter")<<det_id_<<" :"<<temp_[i].Position()<<" "
        <<temp_[i].Sigma()<<" "<<temp_[i].Charge()<<"\n";
    }
  }
  return temp_;
}

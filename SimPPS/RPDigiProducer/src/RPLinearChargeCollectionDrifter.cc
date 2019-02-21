#include "SimPPS/RPDigiProducer/interface/RPLinearChargeCollectionDrifter.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"
#include <iostream>
#include <vector>

RPLinearChargeCollectionDrifter::RPLinearChargeCollectionDrifter
      (const edm::ParameterSet &params, RPDetId det_id)
{
  verbosity_ = params.getParameter<int>("RPVerbosity");
  GeV_per_electron_ = params.getParameter<double>("RPGeVPerElectron");
  charge_cloud_sigmas_vect_ = params.getParameter< std::vector<double> >("RPInterStripSmearing");
  det_thickness_ = RPTopology().DetThickness();
  _det_id=det_id;
}

SimRP::charge_induced_on_surface RPLinearChargeCollectionDrifter::Drift
      (const SimRP::energy_path_distribution &energy_deposition)
{
  _temp.resize(energy_deposition.size());
  for(unsigned int i=0; i<energy_deposition.size(); i++)
  {
    _temp[i].Position() = LocalPoint(energy_deposition[i].X(), energy_deposition[i].Y());
    _temp[i].Sigma() = GetSigma(energy_deposition[i].Z());  //befor charge_cloud_sigma_ used, now a vector of sigmas;
    _temp[i].Charge() = energy_deposition[i].Energy()/GeV_per_electron_;
    if(verbosity_)
    {
      std::cout<<"RPLinearChargeCollectionDrifter "<<_det_id<<" :"<<_temp[i].Position()<<" "
        <<_temp[i].Sigma()<<" "<<_temp[i].Charge()<<std::endl;
    }
  }
  return _temp;
}

#ifndef SimPPS_RPDigiProducer_RP_LINEAR_CHARGE_COLLECTION_DRIFTER_H
#define SimPPS_RPDigiProducer_RP_LINEAR_CHARGE_COLLECTION_DRIFTER_H

#include <vector>
#include <iostream>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimPPS/RPDigiProducer/interface/RPSimTypes.h"

class RPLinearChargeCollectionDrifter
{
  public:
    RPLinearChargeCollectionDrifter(const edm::ParameterSet &params, RPDetId det_id);
    SimRP::charge_induced_on_surface Drift(const SimRP::energy_path_distribution &energy_deposition);
  
  private:
    SimRP::charge_induced_on_surface _temp;
    std::vector<double> charge_cloud_sigmas_vect_;
    double GeV_per_electron_;
    int verbosity_;
    double det_thickness_;
    RPDetId _det_id;
    
    inline double GetSigma(double z)  //z - z position
    {
       if(charge_cloud_sigmas_vect_.size()==1)
         return charge_cloud_sigmas_vect_[0];
       
       double factor = (z/det_thickness_)*(charge_cloud_sigmas_vect_.size()-1);
       double lo_i = floor(factor);
       double hi_i = ceil(factor);
       if(lo_i==hi_i)
       {
         return charge_cloud_sigmas_vect_[(int)factor];
       }
       else
       {
         double lo_weight = hi_i-factor;
         double hi_weight = factor - lo_i;
         
         return charge_cloud_sigmas_vect_[(int)lo_i]*lo_weight 
                + charge_cloud_sigmas_vect_[(int)hi_i]*hi_weight;
       }
    }
};

#endif  //SimPPS_RPDigiProducer_RP_LINEAR_CHARGE_COLLECTION_DRIFTER_H

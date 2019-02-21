#ifndef SimPPS_PPSPixelDigiProducer_RPix_LINEAR_CHARGE_COLLECTION_DRIFTER_H
#define SimPPS_PPSPixelDigiProducer_RPix_LINEAR_CHARGE_COLLECTION_DRIFTER_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimPPS/PPSPixelDigiProducer/interface/RPixSignalPoint.h"
#include "SimPPS/PPSPixelDigiProducer/interface/RPixEnergyDepositUnit.h"

class RPixLinearChargeCollectionDrifter
{
public:
  RPixLinearChargeCollectionDrifter(const edm::ParameterSet &params, uint32_t det_id);
  std::vector<RPixSignalPoint> Drift(const std::vector<RPixEnergyDepositUnit>  &energy_deposition);

private:
  std::vector<RPixSignalPoint>  _temp;

  std::vector<double> charge_cloud_sigmas_vect_; 
  double GeV_per_electron_;
  int verbosity_;
  double det_thickness_;
  uint32_t _det_id;
    
  inline double GetSigma(double z) 
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

#endif  

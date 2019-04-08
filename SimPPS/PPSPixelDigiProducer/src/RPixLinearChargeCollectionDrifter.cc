#include "SimPPS/PPSPixelDigiProducer/interface/RPixLinearChargeCollectionDrifter.h"
#include "Geometry/VeryForwardGeometry/interface/CTPPSPixelTopology.h"
#include <iostream>
#include <vector>

RPixLinearChargeCollectionDrifter::RPixLinearChargeCollectionDrifter
(const edm::ParameterSet &params, uint32_t det_id)
{
  verbosity_ = params.getParameter<int>("RPixVerbosity");

  GeV_per_electron_ = params.getParameter<double>("RPixGeVPerElectron");
  charge_cloud_sigmas_vect_ = params.getParameter< std::vector<double> >("RPixInterSmearing");
  det_thickness_ = CTPPSPixelTopology().detThickness();
  det_id_=det_id;
}

std::vector<RPixSignalPoint> RPixLinearChargeCollectionDrifter::Drift
(const std::vector<RPixEnergyDepositUnit> &energy_deposition)
// convert an energy deposit in a point and in a charge of electrons n=E/3.61 (eV)
{
  temp_.resize(energy_deposition.size());
  for(unsigned int i=0; i<energy_deposition.size(); i++)
    {
      temp_[i].Position() = LocalPoint(energy_deposition[i].X(), energy_deposition[i].Y());

      temp_[i].Sigma() = getSigma_(energy_deposition[i].Z()); 
      temp_[i].Charge() = energy_deposition[i].Energy()/GeV_per_electron_;
      if(verbosity_>1)
	{
	  edm::LogInfo("RPixLinearChargeCollectionDrifter")<<det_id_<<" :"<<temp_[i].Position()<<" "<<temp_[i].Sigma()<<" "<<temp_[i].Charge();
	}
    }
  return temp_;
}

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
  _det_id=det_id;
}

std::vector<RPixSignalPoint> RPixLinearChargeCollectionDrifter::Drift
(const std::vector<RPixEnergyDepositUnit> &energy_deposition)
// convert an energy deposit in a point and in a charge of electrons n=E/3.61 (eV)
{
  _temp.resize(energy_deposition.size());
  for(unsigned int i=0; i<energy_deposition.size(); i++)
    {
      _temp[i].Position() = LocalPoint(energy_deposition[i].X(), energy_deposition[i].Y());

      _temp[i].Sigma() = GetSigma(energy_deposition[i].Z()); 
      _temp[i].Charge() = energy_deposition[i].Energy()/GeV_per_electron_;
      if(verbosity_>1)
	{
	  edm::LogInfo("RPixLinearChargeCollectionDrifter")<<_det_id<<" :"<<_temp[i].Position()<<" "<<_temp[i].Sigma()<<" "<<_temp[i].Charge();
	}
    }
  return _temp;
}

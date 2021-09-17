#include "SimPPS/RPDigiProducer/plugins/RPLinearChargeCollectionDrifter.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <vector>

RPLinearChargeCollectionDrifter::RPLinearChargeCollectionDrifter(const edm::ParameterSet &params, RPDetId det_id) {
  verbosity_ = params.getParameter<int>("RPVerbosity");
  GeV_per_electron_ = params.getParameter<double>("RPGeVPerElectron");
  charge_cloud_sigmas_vect_ = params.getParameter<std::vector<double> >("RPInterStripSmearing");
  det_thickness_ = RPTopology().DetThickness();
  det_id_ = det_id;
}

simromanpot::charge_induced_on_surface RPLinearChargeCollectionDrifter::Drift(
    const simromanpot::energy_path_distribution &energy_deposition) {
  simromanpot::charge_induced_on_surface temp_;
  temp_.resize(energy_deposition.size());
  for (unsigned int i = 0; i < energy_deposition.size(); i++) {
    temp_[i].setPosition(LocalPoint(energy_deposition[i].Position().x(), energy_deposition[i].Position().y()));
    temp_[i].setSigma(
        getSigma(energy_deposition[i].Position().z()));  //befor charge_cloud_sigma_ used, now a vector of sigmas;
    temp_[i].setCharge(energy_deposition[i].Energy() / GeV_per_electron_);
    if (verbosity_) {
      edm::LogInfo("RPLinearChargeCollectionDrifter")
          << det_id_ << " :" << temp_[i].Position() << " " << temp_[i].Sigma() << " " << temp_[i].Charge() << "\n";
    }
  }
  return temp_;
}
double RPLinearChargeCollectionDrifter::getSigma(double z) {
  if (charge_cloud_sigmas_vect_.size() == 1)
    return charge_cloud_sigmas_vect_[0];

  double factor = (z / det_thickness_) * (charge_cloud_sigmas_vect_.size() - 1);
  double lo_i = floor(factor);
  double hi_i = ceil(factor);
  if (lo_i == hi_i) {
    return charge_cloud_sigmas_vect_[(int)factor];
  } else {
    double lo_weight = hi_i - factor;
    double hi_weight = factor - lo_i;

    return charge_cloud_sigmas_vect_[(int)lo_i] * lo_weight + charge_cloud_sigmas_vect_[(int)hi_i] * hi_weight;
  }
}

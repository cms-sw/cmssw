#include "SimPPS/PPSPixelDigiProducer/interface/RPixLinearChargeCollectionDrifter.h"

RPixLinearChargeCollectionDrifter::RPixLinearChargeCollectionDrifter(const edm::ParameterSet &params,
                                                                     uint32_t det_id,
                                                                     const PPSPixelTopology &ppt) {
  verbosity_ = params.getParameter<int>("RPixVerbosity");

  GeV_per_electron_ = params.getParameter<double>("RPixGeVPerElectron");
  charge_cloud_sigmas_vect_ = params.getParameter<std::vector<double> >("RPixInterSmearing");
  det_thickness_ = ppt.getThickness();
  det_id_ = det_id;
}

std::vector<RPixSignalPoint> RPixLinearChargeCollectionDrifter::Drift(
    const std::vector<RPixEnergyDepositUnit> &energy_deposition) {
  // convert an energy deposit in a point and in a charge of electrons n=E/3.61 (eV)
  temp_.resize(energy_deposition.size());
  for (unsigned int i = 0; i < energy_deposition.size(); i++) {
    temp_[i].setPosition(LocalPoint(energy_deposition[i].Position().x(), energy_deposition[i].Position().y()));
    temp_[i].setSigma(getSigma_(energy_deposition[i].Position().z()));
    temp_[i].setCharge(energy_deposition[i].Energy() / GeV_per_electron_);
    if (verbosity_ > 1) {
      edm::LogInfo("PPS") << "RPixLinearChargeCollectionDrifter " << det_id_ << " :" << temp_[i].Position() << " "
                          << temp_[i].Sigma() << " " << temp_[i].Charge();
    }
  }
  return temp_;
}
double RPixLinearChargeCollectionDrifter::getSigma_(double z) {
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

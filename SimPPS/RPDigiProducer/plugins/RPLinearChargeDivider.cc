#include "SimPPS/RPDigiProducer/plugins/RPLinearChargeDivider.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

RPLinearChargeDivider::RPLinearChargeDivider(const edm::ParameterSet& params,
                                             CLHEP::HepRandomEngine& eng,
                                             RPDetId det_id)
    : params_(params), rndEngine_(eng), det_id_(det_id) {
  verbosity_ = params.getParameter<int>("RPVerbosity");

  fluctuate_ = std::make_unique<SiG4UniversalFluctuation>();

  // To Run APV in peak instead of deconvolution mode, which degrades the time resolution.
  //use: SimpleConfigurable<bool> SiLinearChargeDivider::peakMode(false,"SiStripDigitizer:APVpeakmode");

  // To Enable interstrip Landau fluctuations within a cluster.
  //use: SimpleConfigurable<bool> SiLinearChargeDivider::fluctuateCharge(true,"SiStripDigitizer:LandauFluctuations");
  fluctuateCharge_ = params.getParameter<bool>("RPLandauFluctuations");

  // Number of segments per strip into which charge is divided during
  // simulation. If large, precision of simulation improves.
  //to do so: SimpleConfigurable<int> SiLinearChargeDivider::chargeDivisionsPerStrip(10,"SiStripDigitizer:chargeDivisionsPerStrip");
  chargedivisionsPerStrip_ = params.getParameter<int>("RPChargeDivisionsPerStrip");
  chargedivisionsPerThickness_ = params.getParameter<int>("RPChargeDivisionsPerThickness");

  // delta cutoff in MeV, has to be same as in OSCAR (0.120425 MeV corresponding // to 100um range for electrons)
  //        SimpleConfigurable<double>  SiLinearChargeDivider::deltaCut(0.120425,
  deltaCut_ = params.getParameter<double>("RPDeltaProductionCut");

  RPTopology rp_det_topol;
  pitch_ = rp_det_topol.DetPitch();
  thickness_ = rp_det_topol.DetThickness();
}

RPLinearChargeDivider::~RPLinearChargeDivider() {}

simromanpot::energy_path_distribution RPLinearChargeDivider::divide(const PSimHit& hit) {
  LocalVector direction = hit.exitPoint() - hit.entryPoint();
  if (direction.z() > 10 || direction.x() > 200 || direction.y() > 200) {
    the_energy_path_distribution_.clear();
    return the_energy_path_distribution_;
  }

  int NumberOfSegmentation_y = (int)(1 + chargedivisionsPerStrip_ * fabs(direction.y()) / pitch_);
  int NumberOfSegmentation_z = (int)(1 + chargedivisionsPerThickness_ * fabs(direction.z()) / thickness_);
  int NumberOfSegmentation = std::max(NumberOfSegmentation_y, NumberOfSegmentation_z);

  double eLoss = hit.energyLoss();  // Eloss in GeV

  the_energy_path_distribution_.resize(NumberOfSegmentation);

  if (fluctuateCharge_) {
    int pid = hit.particleType();
    double momentum = hit.pabs();
    double length = direction.mag();  // Track length in Silicon
    FluctuateEloss(pid, momentum, eLoss, length, NumberOfSegmentation, the_energy_path_distribution_);
    for (int i = 0; i < NumberOfSegmentation; i++) {
      the_energy_path_distribution_[i].setPosition(hit.entryPoint() +
                                                   double((i + 0.5) / NumberOfSegmentation) * direction);
    }
  } else {
    for (int i = 0; i < NumberOfSegmentation; i++) {
      the_energy_path_distribution_[i].setPosition(hit.entryPoint() +
                                                   double((i + 0.5) / NumberOfSegmentation) * direction);
      the_energy_path_distribution_[i].setEnergy(eLoss / (double)NumberOfSegmentation);
    }
  }

  if (verbosity_) {
    edm::LogInfo("RPLinearChargeDivider") << det_id_ << " charge along the track:\n";
    double sum = 0;
    for (unsigned int i = 0; i < the_energy_path_distribution_.size(); i++) {
      edm::LogInfo("RPLinearChargeDivider")
          << the_energy_path_distribution_[i].Position().x() << " " << the_energy_path_distribution_[i].Position().y()
          << " " << the_energy_path_distribution_[i].Position().z() << " " << the_energy_path_distribution_[i].Energy()
          << "\n";
      sum += the_energy_path_distribution_[i].Energy();
    }
    edm::LogInfo("RPLinearChargeDivider") << "energy dep. sum=" << sum << "\n";
  }

  return the_energy_path_distribution_;
}

void RPLinearChargeDivider::FluctuateEloss(int pid,
                                           double particleMomentum,
                                           double eloss,
                                           double length,
                                           int NumberOfSegs,
                                           simromanpot::energy_path_distribution& elossVector) {
  double particleMass = 139.6;  // Mass in MeV, Assume pion
  pid = std::abs(pid);
  if (pid != 211) {  // Mass in MeV
    if (pid == 11)
      particleMass = 0.511;
    else if (pid == 13)
      particleMass = 105.7;
    else if (pid == 321)
      particleMass = 493.7;
    else if (pid == 2212)
      particleMass = 938.3;
  }

  double segmentLength = length / NumberOfSegs;

  // Generate charge fluctuations.
  double de = 0.;
  double sum = 0.;
  double segmentEloss = (eloss * 1000) / NumberOfSegs;  //eloss in MeV
  for (int i = 0; i < NumberOfSegs; i++) {
    // The G4 routine needs momentum in MeV, mass in Mev, delta-cut in MeV,
    // track segment length in mm, segment eloss in MeV
    // Returns fluctuated eloss in MeV
    double deltaCutoff = deltaCut_;
    de = fluctuate_->SampleFluctuations(
             particleMomentum * 1000, particleMass, deltaCutoff, segmentLength, segmentEloss, &(rndEngine_)) /
         1000;  //convert to GeV
    elossVector[i].setEnergy(de);
    sum += de;
  }

  if (sum > 0.) {  // If fluctuations give eloss>0.
    // Rescale to the same total eloss
    double ratio = eloss / sum;
    for (int ii = 0; ii < NumberOfSegs; ii++)
      elossVector[ii].setEnergy(ratio * elossVector[ii].Energy());
  } else {  // If fluctuations gives 0 eloss
    double averageEloss = eloss / NumberOfSegs;
    for (int ii = 0; ii < NumberOfSegs; ii++)
      elossVector[ii].setEnergy(averageEloss);
  }
  return;
}

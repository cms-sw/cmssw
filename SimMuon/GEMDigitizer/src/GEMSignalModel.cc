#include "SimMuon/GEMDigitizer/interface/GEMSignalModel.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"
#include <cmath>
#include <utility>
#include <map>

GEMSignalModel::GEMSignalModel(const edm::ParameterSet& config) 
    : GEMDigiModel(config),
      averageEfficiency_(config.getParameter<double> ("averageEfficiency")),
      averageShapingTime_(config.getParameter<double> ("averageShapingTime")),
      timeResolution_(config.getParameter<double> ("timeResolution")),
      timeJitter_(config.getParameter<double> ("timeJitter")),
      signalPropagationSpeed_(config.getParameter<double> ("signalPropagationSpeed")),
      digitizeOnlyMuons_(config.getParameter<bool> ("digitizeOnlyMuons")),
      resolutionX_(config.getParameter<double>("resolutionX")) {}

GEMSignalModel::~GEMSignalModel() {}

void GEMSignalModel::simulate(const GEMEtaPartition* roll, 
                              const edm::PSimHitContainer& simHits, 
                              CLHEP::HepRandomEngine* engine, 
                              Strips& strips_, 
                              DetectorHitMap& detectorHitMap_) {
  bool digiMuon = false;
  bool digiElec = false;
  for (const auto& hit : simHits)
  {
    if (std::abs(hit.particleType()) != 13 && digitizeOnlyMuons_)
      continue;
    double elecEff = 0.;
    double partMom = hit.pabs();
    double checkMuonEff = CLHEP::RandFlat::shoot(engine, 0., 1.);
    double checkElecEff = CLHEP::RandFlat::shoot(engine, 0., 1.);
    if (std::abs(hit.particleType()) == 13 && checkMuonEff < averageEfficiency_)
      digiMuon = true;
    if (std::abs(hit.particleType()) != 13) //consider all non muon particles with gem efficiency to electrons
    {
      if (partMom <= 1.95e-03)
        elecEff = 1.7e-05 * std::exp(2.1 * partMom * 1000.);
      if (partMom > 1.95e-03 && partMom < 10.e-03)
        elecEff = 1.34 * log(7.96e-01 * partMom * 1000. - 5.75e-01)
            / (1.34 + log(7.96e-01 * partMom * 1000. - 5.75e-01));
      if (partMom > 10.e-03)
        elecEff = 1.;
      if (checkElecEff < elecEff)
        digiElec = true;
    }
   if (!(digiMuon || digiElec))
      continue;
    const int bx(getSimHitBx(&hit, engine));
    const std::vector<std::pair<int, int> >& cluster(simulateClustering(roll, &hit, bx, engine));
    for (const auto & digi : cluster)
    {
      detectorHitMap_.emplace(digi,&hit);
      strips_.emplace(digi);
    }
  }
}

int GEMSignalModel::getSimHitBx(const PSimHit* simhit, CLHEP::HepRandomEngine* engine)
{
  int bx = -999;
  const LocalPoint simHitPos(simhit->localPosition());
  const float tof(simhit->timeOfFlight());
  // random Gaussian time correction due to electronics jitter
  float randomJitterTime = CLHEP::RandGaussQ::shoot(engine, 0., timeJitter_);
  const GEMDetId id(simhit->detUnitId());
  const GEMEtaPartition* roll(geometry_->etaPartition(id));
  if (!roll)
  {
    throw cms::Exception("Geometry")<< "GEMSignalModel::getSimHitBx() - GEM simhit id does not match any GEM roll id: " << id << "\n";
    return 999;
  }
  if (roll->id().region() == 0)
  {
    throw cms::Exception("Geometry") << "GEMSignalModel::getSimHitBx() - this GEM id is from barrel, which cannot happen: " << roll->id() << "\n";
  }
  const double cspeed = 299792458;   // signal propagation speed in vacuum in [m/s]
  const int nstrips = roll->nstrips();
  float middleStrip = nstrips/2.;
  const LocalPoint& middleOfRoll = roll->centreOfStrip(middleStrip);
  const GlobalPoint& globMiddleRol = roll->toGlobal(middleOfRoll);
  double muRadius = sqrt(globMiddleRol.x()*globMiddleRol.x() + globMiddleRol.y()*globMiddleRol.y() +globMiddleRol.z()*globMiddleRol.z());
  double timeCalibrationOffset_ = (muRadius *1e+9)/(cspeed*1e+2); //[ns]

  const TrapezoidalStripTopology* top(dynamic_cast<const TrapezoidalStripTopology*> (&(roll->topology())));
  const float halfStripLength(0.5 * top->stripLength());
  const float distanceFromEdge(halfStripLength - simHitPos.y());

  // signal propagation speed in material in [cm/ns]
  double signalPropagationSpeedTrue = signalPropagationSpeed_ * cspeed * 1e-7;  // 1e+2 * 1e-9;

  // average time for the signal to propagate from the SimHit to the top of a strip
  const float averagePropagationTime(distanceFromEdge / signalPropagationSpeedTrue);
  // random Gaussian time correction due to the finite timing resolution of the detector
  float randomResolutionTime = CLHEP::RandGaussQ::shoot(engine, 0., timeResolution_);
  const float simhitTime(tof + averageShapingTime_ + randomResolutionTime + averagePropagationTime + randomJitterTime);
  float referenceTime = 0.;
  referenceTime = timeCalibrationOffset_ + halfStripLength / signalPropagationSpeedTrue + averageShapingTime_;
  const float timeDifference(simhitTime - referenceTime);
  // assign the bunch crossing
  bx = static_cast<int> (std::round((timeDifference) / 25.));

  // check time
  const bool debug(false);
  if (debug)
  {
    std::cout << "checktime " << "bx = " << bx << "\tdeltaT = " << timeDifference << "\tsimT =  " << simhitTime
        << "\trefT =  " << referenceTime << "\ttof = " << tof << "\tavePropT =  " << averagePropagationTime
        << "\taveRefPropT = " << halfStripLength / signalPropagationSpeedTrue << std::endl;
  }
  return bx;
}

std::vector<std::pair<int, int> > GEMSignalModel::simulateClustering(const GEMEtaPartition* roll,
                                                                     const PSimHit* simHit,
                                                                     const int bx,
                                                                     CLHEP::HepRandomEngine* engine) {

  const LocalPoint & hit_entry(simHit->entryPoint());
  const LocalPoint & hit_exit(simHit->exitPoint());

  LocalPoint start_point, end_point;
  if(hit_entry.x() < hit_exit.x()) {
    start_point = hit_entry;
    end_point = hit_exit;
  } else {
    start_point = hit_exit;
    end_point = hit_entry;
  }

  // Add Gaussian noise to the points towards outside. 
  float smeared_start_x = start_point.x() - std::abs(CLHEP::RandGaussQ::shoot(engine, 0, resolutionX_));
  float smeared_end_x = end_point.x() + std::abs(CLHEP::RandGaussQ::shoot(engine, 0, resolutionX_));

  LocalPoint smeared_start_point(smeared_start_x, start_point.y(), start_point.z());
  LocalPoint smeared_end_point(smeared_end_x, end_point.y(), end_point.z());

  int cluster_start = roll->strip(smeared_start_point);
  int cluster_end = roll->strip(smeared_end_point);

  std::vector< std::pair<int, int> > cluster;
  for (int strip = cluster_start; strip <= cluster_end; strip++) {
    cluster.emplace_back(strip, bx);
  }

  return cluster;
}

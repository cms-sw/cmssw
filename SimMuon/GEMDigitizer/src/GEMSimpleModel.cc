#include "SimMuon/GEMDigitizer/interface/GEMSimpleModel.h"
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

GEMSimpleModel::GEMSimpleModel(const edm::ParameterSet& config) :
GEMDigiModel(config)
, averageEfficiency_(config.getParameter<double> ("averageEfficiency"))
, averageShapingTime_(config.getParameter<double> ("averageShapingTime"))
, timeResolution_(config.getParameter<double> ("timeResolution"))
, timeJitter_(config.getParameter<double> ("timeJitter"))
, averageNoiseRate_(config.getParameter<double> ("averageNoiseRate"))
, signalPropagationSpeed_(config.getParameter<double> ("signalPropagationSpeed"))
, bxwidth_(config.getParameter<int> ("bxwidth"))
, minBunch_(config.getParameter<int> ("minBunch"))
, maxBunch_(config.getParameter<int> ("maxBunch"))
, digitizeOnlyMuons_(config.getParameter<bool> ("digitizeOnlyMuons"))
, doBkgNoise_(config.getParameter<bool> ("doBkgNoise"))
, doNoiseCLS_(config.getParameter<bool> ("doNoiseCLS"))
, fixedRollRadius_(config.getParameter<bool> ("fixedRollRadius"))
, simulateElectronBkg_(config.getParameter<bool> ("simulateElectronBkg"))
, instLumi_(config.getParameter<double>("instLumi"))
, rateFact_(config.getParameter<double>("rateFact"))
, referenceInstLumi_(config.getParameter<double>("referenceInstLumi"))
, resolutionX_(config.getParameter<double>("resolutionX"))
, GE11ElecBkgParam0_(config.getParameter<double>("GE11ElecBkgParam0"))
, GE11ElecBkgParam1_(config.getParameter<double>("GE11ElecBkgParam1"))
, GE11ElecBkgParam2_(config.getParameter<double>("GE11ElecBkgParam2"))
, GE21ElecBkgParam0_(config.getParameter<double>("GE21ElecBkgParam0"))
, GE21ElecBkgParam1_(config.getParameter<double>("GE21ElecBkgParam1"))
, GE21ElecBkgParam2_(config.getParameter<double>("GE21ElecBkgParam2"))
, GE11ModNeuBkgParam0_(config.getParameter<double>("GE11ModNeuBkgParam0"))
, GE11ModNeuBkgParam1_(config.getParameter<double>("GE11ModNeuBkgParam1"))
, GE11ModNeuBkgParam2_(config.getParameter<double>("GE11ModNeuBkgParam2"))
, GE21ModNeuBkgParam0_(config.getParameter<double>("GE11ModNeuBkgParam0"))
, GE21ModNeuBkgParam1_(config.getParameter<double>("GE11ModNeuBkgParam1"))
, GE21ModNeuBkgParam2_(config.getParameter<double>("GE11ModNeuBkgParam2"))
{
}

GEMSimpleModel::~GEMSimpleModel()
{
}

void GEMSimpleModel::setup()
{
  return;
}

void GEMSimpleModel::simulateSignal(const GEMEtaPartition* roll, const edm::PSimHitContainer& simHits, CLHEP::HepRandomEngine* engine)
{
  stripDigiSimLinks_.clear();
  detectorHitMap_.clear();
  stripDigiSimLinks_ = StripDigiSimLinks(roll->id().rawId());
  theGemDigiSimLinks_.clear();
  theGemDigiSimLinks_ = GEMDigiSimLinks(roll->id().rawId());
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

int GEMSimpleModel::getSimHitBx(const PSimHit* simhit, CLHEP::HepRandomEngine* engine)
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
    throw cms::Exception("Geometry")<< "GEMSimpleModel::getSimHitBx() - GEM simhit id does not match any GEM roll id: " << id << "\n";
    return 999;
  }
  if (roll->id().region() == 0)
  {
    throw cms::Exception("Geometry") << "GEMSimpleModel::getSimHitBx() - this GEM id is from barrel, which cannot happen: " << roll->id() << "\n";
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
  bx = static_cast<int> (std::round((timeDifference) / bxwidth_));

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

void GEMSimpleModel::simulateNoise(const GEMEtaPartition* roll, CLHEP::HepRandomEngine* engine)
{
  if (!doBkgNoise_)
    return;
  const GEMDetId& gemId(roll->id());
  const int nstrips(roll->nstrips());
  double trArea(0.0);
  double trStripArea(0.0);
  if (gemId.region() == 0)
  {
    throw cms::Exception("Geometry")
        << "GEMSynchronizer::simulateNoise() - this GEM id is from barrel, which cannot happen.";
  }
  const TrapezoidalStripTopology* top_(dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology())));
  const float striplength(top_->stripLength());
  trStripArea = (roll->pitch()) * striplength;
  trArea = trStripArea * nstrips;
  const int nBxing(maxBunch_ - minBunch_ + 1);
  const float rollRadius(fixedRollRadius_ ? top_->radius() : 
			 top_->radius() + CLHEP::RandFlat::shoot(engine, -1.*top_->stripLength()/2., top_->stripLength()/2.));

  //calculate noise from model
  double averageNeutralNoiseRatePerRoll = 0.;
  double averageNoiseElectronRatePerRoll = 0.;
  double averageNoiseRatePerRoll = 0.;
  if (gemId.station() == 1)
  {
    //simulate neutral background for GE1/1
    averageNeutralNoiseRatePerRoll = (GE11ModNeuBkgParam0_
				      + GE11ModNeuBkgParam1_ * rollRadius
				      + GE11ModNeuBkgParam2_ * rollRadius * rollRadius);    //simulate electron background for GE1/1
    if (simulateElectronBkg_)
      averageNoiseElectronRatePerRoll = (GE11ElecBkgParam0_
					 + GE11ElecBkgParam1_ * rollRadius
					 + GE11ElecBkgParam2_ * rollRadius * rollRadius);

    // Scale up/down for desired instantaneous lumi (reference is 5E34, double from config is in units of 1E34)
    averageNoiseRatePerRoll = averageNeutralNoiseRatePerRoll + averageNoiseElectronRatePerRoll;
    averageNoiseRatePerRoll *= instLumi_*rateFact_*1.0/referenceInstLumi_;
  }
  if (gemId.station() == 2)
  {
    //simulate neutral background for GE2/1
    averageNeutralNoiseRatePerRoll = (GE21ModNeuBkgParam0_
				      + GE21ModNeuBkgParam1_ * rollRadius
				      + GE21ModNeuBkgParam2_ * rollRadius * rollRadius);
    //simulate electron background for GE2/1
    if (simulateElectronBkg_)
      averageNoiseElectronRatePerRoll = (GE21ElecBkgParam0_
					 + GE21ElecBkgParam1_ * rollRadius
					 + GE21ElecBkgParam2_ * rollRadius * rollRadius);

    // Scale up/down for desired instantaneous lumi (reference is 5E34, double from config is in units of 1E34)
    averageNoiseRatePerRoll = averageNeutralNoiseRatePerRoll + averageNoiseElectronRatePerRoll;
    averageNoiseRatePerRoll *= instLumi_*rateFact_*1.0/referenceInstLumi_;
  }

  //simulate intrinsic noise
  if(simulateIntrinsicNoise_)
  {
    const double aveIntrinsicNoisePerStrip(averageNoiseRate_ * nBxing * bxwidth_ * trStripArea * 1.0e-9);
    for(int j = 0; j < nstrips; ++j)
    {
      CLHEP::RandPoissonQ randPoissonQ(*engine, aveIntrinsicNoisePerStrip);
      const int n_intrHits(randPoissonQ.fire());
    
      for (int k = 0; k < n_intrHits; k++ )
	{
        const int time_hit(static_cast<int>(CLHEP::RandFlat::shoot(engine, nBxing)) + minBunch_);
        std::pair<int, int> digi(k+1,time_hit);
        strips_.emplace(digi);
      }
    }
  }//end simulate intrinsic noise

  //simulate bkg contribution
  const double averageNoise(averageNoiseRatePerRoll * nBxing * bxwidth_ * trArea * 1.0e-9);
  CLHEP::RandPoissonQ randPoissonQ(*engine, averageNoise);
  const int n_hits(randPoissonQ.fire());
  for (int i = 0; i < n_hits; ++i)
  {
    const int centralStrip(static_cast<int> (CLHEP::RandFlat::shoot(engine, 1, nstrips)));
    const int time_hit(static_cast<int>(CLHEP::RandFlat::shoot(engine, nBxing)) + minBunch_);
    if (doNoiseCLS_)
    {
      std::vector < std::pair<int, int> > cluster_;
      cluster_.clear();
      cluster_.emplace_back(centralStrip, time_hit);
      int clusterSize((CLHEP::RandFlat::shoot(engine)) <= 0.53 ? 1 : 2);
      if (clusterSize == 2)
      {
        if(CLHEP::RandFlat::shoot(engine) < 0.5)
        {
          if (CLHEP::RandFlat::shoot(engine) < averageEfficiency_ && (centralStrip - 1 > 0))
            cluster_.emplace_back(centralStrip - 1, time_hit);
        }
        else
        {
          if (CLHEP::RandFlat::shoot(engine) < averageEfficiency_ && (centralStrip + 1 <= nstrips))
            cluster_.emplace_back(centralStrip + 1, time_hit);
        }
      }
      for (const auto& digi : cluster_)
      {
        strips_.emplace(digi);
      }
    } //end doNoiseCLS_
    else
    {
      strips_.emplace(centralStrip, time_hit);
    }
  }
  return;
}


std::vector<std::pair<int, int> > GEMSimpleModel::simulateClustering(
    const GEMEtaPartition* roll,
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

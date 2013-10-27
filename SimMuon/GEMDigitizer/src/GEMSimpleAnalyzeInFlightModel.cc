#include "SimMuon/GEMDigitizer/interface/GEMSimpleAnalyzeInFlightModel.h"

#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandGamma.h"
#include "CLHEP/Random/RandLandau.h"

#include <cmath>
#include <utility>
#include <map>

namespace
{
  // "magic" parameter for cosmics
  const double COSMIC_PAR(37.62);
}

GEMSimpleAnalyzeInFlightModel::GEMSimpleAnalyzeInFlightModel(const edm::ParameterSet& config) :
  GEMDigiModel(config)//
      , averageEfficiency_(config.getParameter<double> ("averageEfficiency"))//
      , averageShapingTime_(config.getParameter<double> ("averageShapingTime"))//
      , timeResolution_(config.getParameter<double> ("timeResolution"))//
      , timeJitter_(config.getParameter<double> ("timeJitter"))//
      , timeCalibrationOffset_(config.getParameter<double> ("timeCalibrationOffset"))//
      , averageNoiseRate_(config.getParameter<double> ("averageNoiseRate"))//
      , averageClusterSize_(config.getParameter<double> ("averageClusterSize"))//
      , signalPropagationSpeed_(config.getParameter<double> ("signalPropagationSpeed"))//
      , cosmics_(config.getParameter<bool> ("cosmics"))//
      , bxwidth_(config.getParameter<int> ("bxwidth"))//
      , minBunch_(config.getParameter<int> ("minBunch"))//
      , maxBunch_(config.getParameter<int> ("maxBunch"))//
      , digitizeOnlyMuons_(config.getParameter<bool> ("digitizeOnlyMuons"))//
      , cutElecMomentum_(config.getParameter<double> ("cutElecMomentum"))//
      , cutForCls_(config.getParameter<int> ("cutForCls"))//
      , neutronGammaRoll1_(config.getParameter<double> ("neutronGammaRoll1"))//
      , neutronGammaRoll2_(config.getParameter<double> ("neutronGammaRoll2"))//
      , neutronGammaRoll3_(config.getParameter<double> ("neutronGammaRoll3"))//
      , neutronGammaRoll4_(config.getParameter<double> ("neutronGammaRoll4"))//
      , neutronGammaRoll5_(config.getParameter<double> ("neutronGammaRoll5"))//
      , neutronGammaRoll6_(config.getParameter<double> ("neutronGammaRoll6"))//
      , neutronGammaRoll7_(config.getParameter<double> ("neutronGammaRoll7"))//
      , neutronGammaRoll8_(config.getParameter<double> ("neutronGammaRoll8"))//
{
  edm::Service<TFileService> fs;
  particleId_h = fs->make<TH1F> ("particleId", "particleId", 6000, -3000, 3000);
  energyLoss_el = fs->make<TH1F> ("energyLoss_el", "energyLoss_el", 1000, 0., 0.01);
  energyLoss_mu = fs->make<TH1F> ("energyLoss_mu", "energyLoss_mu", 1000, 0., 0.01);
  tof_el = fs->make<TH1F> ("tof_el", "tof_el", 2000, -100., 100.);
  tof_mu = fs->make<TH1F> ("tof_mu", "tof_mu", 2000, -100., 100.);
  pabs_el = fs->make<TH1F> ("pabs_el", "pabs_el", 50000, 0., 500.);
  pabs_mu = fs->make<TH1F> ("pabs_mu", "pabs_mu", 50000, 0., 500.);
  process_el = fs->make<TH1F> ("process_el", "process_el", 10, 0., 10.);
  process_mu = fs->make<TH1F> ("process_mu", "process_mu", 10, 0., 10.);
  cls_el = fs->make<TH1F> ("cls_el", "cls_el", 10, 0., 10.);
  cls_mu = fs->make<TH1F> ("cls_mu", "cls_mu", 10, 0., 10.);
  cls_all = fs->make<TH1F> ("cls_all", "cls_all", 10, 0., 10.);
  res_mu = fs->make<TH1F> ("res_mu", "res_mu", 10000, -5., 5.);
  res_el = fs->make<TH1F> ("res_el", "res_el", 10000, -5., 5.);
  res_all = fs->make<TH1F> ("res_all", "res_all", 10000, -5., 5.);
  bx_h = fs->make<TH1F> ("bx", "bx", 12, -6, 6);
  numbDigis = fs->make<TH1F> ("deltaStrip", "deltaStrip", 390, 0., 390);
  poisHisto = fs->make<TH1F> ("poisHisto", "poisHisto", 2000, 0., 20);
  noisyBX = fs->make<TH1F> ("noisyBX", "noisyBX", 20, -10., 10);

  selPsimHits = new std::vector<PSimHit>;

  res_mu1 = fs->make<TH1F> ("res_mu1", "res_mu1", 10000, -5., 5.);
  res_mu8 = fs->make<TH1F> ("res_mu8", "res_mu8", 10000, -5., 5.);
  bx_final = fs->make<TH1F> ("bx_final", "bx_final", 12, -6, 6);
  stripProfile = fs->make<TH1F> ("stripProfile", "stripProfile", 390, 0., 390);

}

GEMSimpleAnalyzeInFlightModel::~GEMSimpleAnalyzeInFlightModel()
{
  if (flat1_)
    delete flat1_;
  if (flat2_)
    delete flat2_;
  if (poisson_)
    delete poisson_;
  if (gauss1_)
    delete gauss1_;
  if (gauss2_)
    delete gauss2_;
  if (gamma1_)
    delete gamma1_;
  if (selPsimHits)
    delete selPsimHits;
}

void GEMSimpleAnalyzeInFlightModel::setRandomEngine(CLHEP::HepRandomEngine& eng)
{
  flat1_ = new CLHEP::RandFlat(eng);
  flat2_ = new CLHEP::RandFlat(eng);
  poisson_ = new CLHEP::RandPoissonQ(eng);
  gauss1_ = new CLHEP::RandGaussQ(eng);
  gauss2_ = new CLHEP::RandGaussQ(eng);
  gamma1_ = new CLHEP::RandGamma(eng);
  gamma1_ = new CLHEP::RandGamma(eng);
  landau1_ = new CLHEP::RandLandau(eng);
}

void GEMSimpleAnalyzeInFlightModel::setup()
{
  return;
}

void GEMSimpleAnalyzeInFlightModel::simulateSignal(const GEMEtaPartition* roll, const edm::PSimHitContainer& simHits)
{
  stripDigiSimLinks_.clear();
  detectorHitMap_.clear();
  stripDigiSimLinks_ = StripDigiSimLinks(roll->id().rawId());
  //  const Topology& topology(roll->specs()->topology());
  selPsimHits->clear();

  for (edm::PSimHitContainer::const_iterator hit1 = simHits.begin(); hit1 != simHits.end(); ++hit1)
  {
    particleId_h->Fill(hit1->particleType());
    if (!(abs(hit1->particleType()) == 13 || abs(hit1->particleType()) == 11))
      continue;
    if ((abs(hit1->particleType() != 13)) && (hit1->pabs() < cutElecMomentum_))
      continue;
    selPsimHits->push_back(*hit1);
  }

  int offset = selPsimHits->begin() == selPsimHits->end() ? 0 : 1;
  edm::PSimHitContainer::iterator hitEnd = selPsimHits->end();
  for (edm::PSimHitContainer::iterator hit1 = selPsimHits->begin(); hit1 != hitEnd - offset && hit1 != hitEnd; ++hit1)
  {
    const auto entry1(hit1->entryPoint());
    const Topology& topology(roll->specs()->topology());
    const int hitStrip1(topology.channel(entry1) + 1);

    for (edm::PSimHitContainer::iterator hit2 = hit1 + 1; hit2 != hitEnd;)
    {
      const auto entry2(hit2->entryPoint());
      const Topology& topology(roll->specs()->topology());
      const int hitStrip2(topology.channel(entry2) + 1);

      int deltaStrip = abs(hitStrip1 - hitStrip2);
      numbDigis->Fill(deltaStrip);
      if (deltaStrip < cutForCls_)
      {
        hit2 = selPsimHits->erase(hit2);
        hitEnd = selPsimHits->end();
      }
      else
      {
        ++hit2;
      }
    }
  }

  if ((selPsimHits->size() > 0))
  {
    //    std::cout << "---------------------------" << std::endl;
for  (const auto & hit: (*selPsimHits))
  {
    if (std::fabs(hit.particleType()) == 11)
    {
      energyLoss_el->Fill(hit.energyLoss());
      tof_el->Fill(hit.timeOfFlight());
      pabs_el->Fill(hit.pabs());
      process_el->Fill(hit.processType());
    }
    if (std::fabs(hit.particleType()) == 13)
    {
      energyLoss_mu->Fill(hit.energyLoss());
      tof_mu->Fill(hit.timeOfFlight());
      pabs_mu->Fill(hit.pabs());
      process_mu->Fill(hit.processType());
    }

    if (std::abs(hit.particleType()) != 13 && digitizeOnlyMuons_)
    continue;
    // Check GEM efficiency
    if (flat1_->fire(1) > averageEfficiency_)
    continue;
    const int bx(getSimHitBx(&hit));
    const std::vector<std::pair<int, int> > cluster(simulateClustering(roll, &hit, bx));
    for (auto & digi : cluster)
    {
      //       std::cout << hit.particleType() << "\t" << digi.first << std::endl;
      detectorHitMap_.insert(DetectorHitMap::value_type(digi,&hit));
      strips_.insert(digi);
      bx_final->Fill(digi.second);
      stripProfile->Fill(digi.first);
    }
  }
}
}

int GEMSimpleAnalyzeInFlightModel::getSimHitBx(const PSimHit* simhit)
{
  int bx = -999;
  const LocalPoint simHitPos(simhit->localPosition());
  const float tof(simhit->timeOfFlight());
  // random Gaussian time correction due to electronics jitter
  const float randomJitterTime(gauss1_->fire(0., timeJitter_));

  const GEMDetId id(simhit->detUnitId());
  const GEMEtaPartition* roll(geometry_->etaPartition(id));

  if (!roll)
  {
    throw cms::Exception("Geometry")
        << "GEMSimpleAnalyzeInFlightModel::getSimHitBx() - GEM simhit id does not match any GEM roll id: " << id
        << "\n";
    return 999;
  }

  if (roll->id().region() == 0)
  {
    throw cms::Exception("Geometry")
        << "GEMSimpleAnalyzeInFlightModel::getSimHitBx() - this GEM id is from barrel, which cannot happen: "
        << roll->id() << "\n";
  }

  const TrapezoidalStripTopology* top(dynamic_cast<const TrapezoidalStripTopology*> (&(roll->topology())));
  const float halfStripLength(0.5 * top->stripLength());
  const float distanceFromEdge(halfStripLength - simHitPos.y());

  // average time for the signal to propagate from the SimHit to the top of a strip
  const float averagePropagationTime(distanceFromEdge / signalPropagationSpeed_);
  // random Gaussian time correction due to the finite timing resolution of the detector
  const float randomResolutionTime(gauss2_->fire(0., timeResolution_));

  const float simhitTime(tof + averageShapingTime_ + randomResolutionTime + averagePropagationTime + randomJitterTime);
  const float referenceTime(timeCalibrationOffset_ + halfStripLength / signalPropagationSpeed_ + averageShapingTime_);
  const float timeDifference(cosmics_ ? (simhitTime - referenceTime) / COSMIC_PAR : simhitTime - referenceTime);

  // assign the bunch crossing
  bx = static_cast<int> (std::round((timeDifference) / bxwidth_));

  bx_h->Fill(bx);
  // check time
  const bool debug(false);
  if (debug)
  {
    std::cout << "checktime " << "bx = " << bx << "\tdeltaT = " << timeDifference << "\tsimT =  " << simhitTime
        << "\trefT =  " << referenceTime << "\ttof = " << tof << "\tavePropT =  " << averagePropagationTime
        << "\taveRefPropT = " << halfStripLength / signalPropagationSpeed_ << std::endl;
  }
  return bx;
}

void GEMSimpleAnalyzeInFlightModel::simulateNoise(const GEMEtaPartition* roll)
{
  const GEMDetId gemId(roll->id());
  int rollNumb = gemId.roll();
  const int nstrips(roll->nstrips());
  double area(0.0);

  if (gemId.region() == 0)
  {
    throw cms::Exception("Geometry")
        << "GEMSynchronizer::simulateNoise() - this GEM id is from barrel, which cannot happen.";
  }
  const TrapezoidalStripTopology* top_(dynamic_cast<const TrapezoidalStripTopology*> (&(roll->topology())));
  const float xmin((top_->localPosition(0.)).x());
  const float xmax((top_->localPosition((float) roll->nstrips())).x());
  const float striplength(top_->stripLength());
  area = striplength * (xmax - xmin);

  const int nBxing(maxBunch_ - minBunch_ + 1);
  double averageNoiseRatePerStrip;
  if (rollNumb == 1)
  {
    averageNoiseRatePerStrip = neutronGammaRoll1_ / roll->nstrips() + averageNoiseRate_;
  }
  if (rollNumb == 2)
  {
    averageNoiseRatePerStrip = neutronGammaRoll2_ / roll->nstrips() + averageNoiseRate_;
  }
  if (rollNumb == 3)
  {
    averageNoiseRatePerStrip = neutronGammaRoll3_ / roll->nstrips() + averageNoiseRate_;
  }
  if (rollNumb == 4)
  {
    averageNoiseRatePerStrip = neutronGammaRoll4_ / roll->nstrips() + averageNoiseRate_;
  }
  if (rollNumb == 5)
  {
    averageNoiseRatePerStrip = neutronGammaRoll5_ / roll->nstrips() + averageNoiseRate_;
  }
  if (rollNumb == 6)
  {
    averageNoiseRatePerStrip = neutronGammaRoll6_ / roll->nstrips() + averageNoiseRate_;
  }
  if (rollNumb == 7)
  {
    averageNoiseRatePerStrip = neutronGammaRoll7_ / roll->nstrips() + averageNoiseRate_;
  }
  if (rollNumb == 8)
  {
    averageNoiseRatePerStrip = neutronGammaRoll8_ / roll->nstrips() + averageNoiseRate_;
  }

  //  const double averageNoise(averageNoiseRate_ * nBxing * bxwidth_ * area * 1.0e-9);
  const double averageNoise(averageNoiseRatePerStrip * nBxing * bxwidth_ * area * 1.0e-9);
  const int n_hits(poisson_->fire(averageNoise));
  poisHisto->Fill(n_hits);

  for (int i = 0; i < n_hits; ++i)
  {
    const int strip(static_cast<int> (flat1_->fire(1, nstrips)));
    const int time_hit(static_cast<int> (flat2_->fire(nBxing)) + minBunch_);
    std::pair<int, int> digi(strip, time_hit);
    strips_.insert(digi);
    noisyBX->Fill(time_hit);
  }
  return;
}

std::vector<std::pair<int, int> > GEMSimpleAnalyzeInFlightModel::simulateClustering(const GEMEtaPartition* roll,
    const PSimHit* simHit, const int bx)
{
  const auto entry(simHit->entryPoint());
  const Topology& topology(roll->specs()->topology());
  const int centralStrip(topology.channel(entry) + 1);

  LocalPoint lsimHitEntry(entry);
  double deltaX(roll->centreOfStrip(centralStrip).x() - entry.x());

  // Add central digi to cluster vector
  std::vector<std::pair<int, int> > cluster_;
  cluster_.clear();
  cluster_.push_back(std::pair<int, int>(centralStrip, bx));

  // get the cluster size
  //    const int clusterSize(static_cast<int>(std::round(poisson_->fire(averageClusterSize_))));
  //  const int clusterSize(static_cast<int> (std::round(landau1_->fire()))   );
  const int clusterSize(static_cast<int> (std::round(gamma1_->fire(averageClusterSize_, averageClusterSize_))));

  if (std::fabs((*simHit).particleType()) == 13)
  {
    res_mu->Fill(deltaX);
    if (roll->id().roll() == 1)
      res_mu1->Fill(deltaX);
    if (roll->id().roll() == 8)
      res_mu8->Fill(deltaX);
    if (clusterSize < 1)
      cls_mu->Fill(1);
    else
      cls_mu->Fill(clusterSize);
  }
  if (std::fabs((*simHit).particleType()) == 11)
  {
    res_el->Fill(deltaX);
    if (clusterSize < 1)
      cls_el->Fill(1);
    else
      cls_el->Fill(clusterSize);
  }

  res_all->Fill(deltaX);
  if (clusterSize < 1)
    cls_all->Fill(1);
  else
    cls_all->Fill(clusterSize);

  if (clusterSize < 1)
    return cluster_;

  //odd cls
  if (clusterSize % 2 != 0)
  {
    int clsR = clusterSize - 1;
    cluster_.push_back(std::pair<int, int>(centralStrip, bx));
    for (int i = 1; i < clsR; ++i)
    {
      cluster_.push_back(std::pair<int, int>(centralStrip - i, bx));
      cluster_.push_back(std::pair<int, int>(centralStrip + i, bx));
    }
  }
  //even cls
  if (clusterSize % 2 == 0)
  {
    cluster_.push_back(std::pair<int, int>(centralStrip, bx));
    if (deltaX <= 0)
    {
      cluster_.push_back(std::pair<int, int>(centralStrip - 1, bx));
      int clsR = clusterSize - 2;
      for (int i = 1; i < clsR; ++i)
      {
        cluster_.push_back(std::pair<int, int>(centralStrip - 1 - i, bx));
        cluster_.push_back(std::pair<int, int>(centralStrip + i, bx));
      }
    }
    else
    {
      cluster_.push_back(std::pair<int, int>(centralStrip + 1, bx));
      int clsR = clusterSize - 2;
      for (int i = 1; i < clsR; ++i)
      {
        cluster_.push_back(std::pair<int, int>(centralStrip + 1 + i, bx));
        cluster_.push_back(std::pair<int, int>(centralStrip - i, bx));
      }
    }
  }

  //
  /*
   // Add the other digis to the cluster
   for (int cl = 0; cl < (clusterSize - 1) / 2; ++cl)
   {
   if (centralStrip - cl - 1 >= 1)
   cluster_.push_back(std::pair<int, int>(centralStrip - cl - 1, bx));
   if (centralStrip + cl + 1 <= roll->nstrips())
   cluster_.push_back(std::pair<int, int>(centralStrip + cl + 1, bx));
   }
   if (clusterSize % 2 == 0)
   {
   // insert the last strip according to the
   // simhit position in the central strip
   //    const double deltaX(roll->centreOfStrip(centralStrip).x() - entry.x());
   if (deltaX < 0.)
   {
   if (lstrip < roll->nstrips())
   {
   ++lstrip;
   cluster_.push_back(std::pair<int, int>(lstrip, bx));
   }
   }
   else
   {
   if (fstrip > 1)
   {
   --fstrip;
   cluster_.push_back(std::pair<int, int>(fstrip, bx));
   }
   }
   }
   */
  return cluster_;
}

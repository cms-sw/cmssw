#include "SimMuon/GEMDigitizer/src/GEMDigiModel.h"
#include "SimMuon/GEMDigitizer/src/GEMFactory.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "CondFormats/GEMObjects/interface/GEMStripNoise.h" 
#include "CondFormats/DataRecord/interface/GEMStripNoiseRcd.h" 
#include "CondFormats/GEMObjects/interface/GEMStripClustering.h" 
#include "CondFormats/DataRecord/interface/GEMStripClusteringRcd.h" 
#include "CondFormats/GEMObjects/interface/GEMStripEfficiency.h" 
#include "CondFormats/DataRecord/interface/GEMStripEfficiencyRcd.h" 
#include "CondFormats/GEMObjects/interface/GEMStripTiming.h" 
#include "CondFormats/DataRecord/interface/GEMStripTimingRcd.h" 

#include <utility>
#include <set>
#include <iostream>


GEMDigiModel::GEMDigiModel(const edm::ParameterSet& config)
  : digiModelString_(config.getParameter<std::string>("digiModelString"))
{ 
  if (digiModelString_=="Detailed")
  {
    throw cms::Exception("GEMDigiModel")
      << "GEMDigiModel::GEMDigiModel() - this GEMDigiModel is not available yet.";
  }
  timingModel_ = GEMTimingFactory::get()->create("GEMTiming" + digiModelString_, config);
  noiseModel_ = GEMNoiseFactory::get()->create("GEMNoise" + digiModelString_, config);
  clusteringModel_ = GEMClusteringFactory::get()->create("GEMClustering" + digiModelString_, config);
  efficiencyModel_ = GEMEfficiencyFactory::get()->create("GEMEfficiency" + digiModelString_, config);
}


GEMDigiModel::~GEMDigiModel()
{
  if (timingModel_) delete timingModel_;
  if (noiseModel_) delete noiseModel_;
  if (clusteringModel_) delete clusteringModel_;
  if (efficiencyModel_) delete efficiencyModel_;
}


void
GEMDigiModel::setRandomEngine(CLHEP::HepRandomEngine& eng)
{
  timingModel_->setRandomEngine(eng);
  noiseModel_->setRandomEngine(eng);
  clusteringModel_->setRandomEngine(eng);
  efficiencyModel_->setRandomEngine(eng);
}


void 
GEMDigiModel::setUp(std::vector<GEMStripTiming::StripTimingItem> vTiming, 
		    std::vector<GEMStripNoise::StripNoiseItem> vNoise, 
		    std::vector<GEMStripClustering::StripClusteringItem> vCls, 
		    std::vector<GEMStripEfficiency::StripEfficiencyItem> vEff)
{
  timingModel_->setUp(vTiming);
  noiseModel_->setUp(vNoise);
  clusteringModel_->setUp(vCls);
  efficiencyModel_->setUp(vEff);
}


void 
GEMDigiModel::setGeometry(const GEMGeometry *geom)
{
  geometry_ = geom;
  timingModel_->setGeometry(geom);
  noiseModel_->setGeometry(geom);
  clusteringModel_->setGeometry(geom);
  efficiencyModel_->setGeometry(geom);
}


void 
GEMDigiModel::simulateSignal(const GEMEtaPartition* roll,
			     const edm::PSimHitContainer& simHits)
{
  stripDigiSimLinks_.clear();
  detectorHitMap_.clear();
  stripDigiSimLinks_ = StripDigiSimLinks(roll->id().rawId());

  for (const auto & hit: simHits)
  {
    if (std::abs(hit.particleType()) != 13) continue;
    // select only the good detIds according to the chosen efficiency model
    if (!efficiencyModel_->isGoodDetId(hit.detUnitId())) continue; 
    // get the bunch crossing according to the chosen timing model
    const int bx(timingModel_->getSimHitBx(&hit));
    // simulate the clustering according to the chosen clustering model
    const std::vector<std::pair<int,int> > cluster(clusteringModel_->simulateClustering(roll, &hit,bx));

    // FIXME
    for (auto & digi : cluster)
    {
      detectorHitMap_.insert(DetectorHitMap::value_type(digi,&hit));
      strips_.insert(digi);
    }
  }
}


void 
GEMDigiModel::simulateNoise(const GEMEtaPartition* roll)
{
  const GEMDetId gemId(roll->id());
  if (gemId.region() == 0)
  {
    throw cms::Exception("Geometry")
      << "GEMDigiModel::simulateNoise() - this GEM id is from barrel, which cannot happen.";
  }
  // get the noise collection according to the chosen noise model
  std::set< std::pair<int,int> > noiseCollection(noiseModel_->simulateNoise(roll));
  for (auto d : noiseCollection)
  {
    strips_.insert(d);
  }
}


void 
GEMDigiModel::fillDigis(const uint32_t detId, GEMDigiCollection& digis)
{
  for (auto d: strips_)
  {
    if (d.second == -999) continue;
    GEMDigi digi(d.first, d.second); // (strip, bx)
    digis.insertDigi(GEMDetId(detId), digi);
    addLinks(d.first, d.second);
  }
  strips_.clear();
}


void 
GEMDigiModel::addLinks(int strip, int bx)
{
  /*
  std::pair<unsigned int, int> digi(strip, bx);
  auto channelHitItr = detectorHitMap_.equal_range(digi);

  // find the fraction contribution for each SimTrack
  std::map<int, float> simTrackChargeMap;
  std::map<int, EncodedEventId> eventIdMap;
  float totalCharge(0);
  for(auto hitItr = channelHitItr.first; hitItr != channelHitItr.second; ++hitItr)
  {
    const PSimHit * hit(hitItr->second);
    // might be zero for unit tests and such
    if(!hit) continue;
    
    int simTrackId(hit->trackId());
    //float charge = hit->getCharge();
    const float charge(1.f);
    auto chargeItr = simTrackChargeMap.find(simTrackId);
    if( chargeItr == simTrackChargeMap.end() )
    {
      simTrackChargeMap[simTrackId] = charge;
      eventIdMap[simTrackId] = hit->eventId();
    }
    else 
    {
      chargeItr->second += charge;
    }
    totalCharge += charge;
  }
  
  for(auto &charge: simTrackChargeMap)
  {
    int simTrackId(charge.first);
    stripDigiSimLinks_.push_back(StripDigiSimLink(strip, simTrackId, eventIdMap[simTrackId], charge.second/totalCharge));
  }
  */
}



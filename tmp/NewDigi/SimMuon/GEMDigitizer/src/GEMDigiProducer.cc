#include "SimMuon/GEMDigitizer/src/GEMDigiProducer.h"
#include "SimMuon/GEMDigitizer/src/GEMDigiModel.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "CondFormats/GEMObjects/interface/GEMStripNoise.h" 
#include "CondFormats/DataRecord/interface/GEMStripNoiseRcd.h" 
#include "CondFormats/GEMObjects/interface/GEMStripClustering.h" 
#include "CondFormats/DataRecord/interface/GEMStripClusteringRcd.h" 
#include "CondFormats/GEMObjects/interface/GEMStripEfficiency.h" 
#include "CondFormats/DataRecord/interface/GEMStripEfficiencyRcd.h" 
#include "CondFormats/GEMObjects/interface/GEMStripTiming.h" 
#include "CondFormats/DataRecord/interface/GEMStripTimingRcd.h" 

#include <string>
#include <map>


GEMDigiProducer::GEMDigiProducer(const edm::ParameterSet& ps)
  : gemDigiModel_(new GEMDigiModel(ps))
  , inputCollection_(ps.getParameter<std::string>("inputCollection"))
  , digiModelString_(ps.getParameter<std::string>("digiModelString"))
{
  produces<GEMDigiCollection>();
  produces<StripDigiSimLinks>("GEM");

  const edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable())
  {
    throw cms::Exception("Configuration")
      << "GEMDigiProducer::GEMDigiProducer() - RandomNumberGeneratorService is not present in configuration file.\n"
      << "Add the service in the configuration file or remove the modules that require it.";
  }
  CLHEP::HepRandomEngine& eng(rng->getEngine());
  gemDigiModel_->setRandomEngine(eng);
}


GEMDigiProducer::~GEMDigiProducer()
{
  if (gemDigiModel_) delete gemDigiModel_;
}


void GEMDigiProducer::beginRun( edm::Run& iRun, const edm::EventSetup& iEventSetup)
{
  edm::ESHandle<GEMGeometry> hGeom;
  iEventSetup.get<MuonGeometryRecord>().get(hGeom);
  gemDigiModel_->setGeometry(&*hGeom);

  edm::ESHandle<GEMStripTiming> timingRcd;
  iEventSetup.get<GEMStripTimingRcd>().get(timingRcd);
  
  edm::ESHandle<GEMStripNoise> noiseRcd;
  iEventSetup.get<GEMStripNoiseRcd>().get(noiseRcd);
   
  edm::ESHandle<GEMStripClustering> clusterRcd;
  iEventSetup.get<GEMStripClusteringRcd>().get(clusterRcd);
  
  edm::ESHandle<GEMStripEfficiency> efficiencyRcd;
  iEventSetup.get<GEMStripEfficiencyRcd>().get(efficiencyRcd);
  
  std::vector<GEMStripTiming::StripTimingItem> vTiming;
  std::vector<GEMStripNoise::StripNoiseItem> vNoise;
  std::vector<GEMStripClustering::StripClusteringItem> vCluster;
  std::vector<GEMStripEfficiency::StripEfficiencyItem> vEfficiency;

//   if (digiModelString_ == "Detailed")
//   {
//     vTiming = timingRcd->getStripTimingVector(); 
//     vNoise = noiseRcd->getStripNoiseVector();	       
//     vCluster = clusterRcd->getClusterSizeVector(); 
//     vEfficiency = efficiencyRcd->getStripEfficiencyVector();		       
//   }
  gemDigiModel_->setUp(vTiming,vNoise,vCluster,vEfficiency);
}


void GEMDigiProducer::produce(edm::Event& iEvent, const edm::EventSetup& iEventSetup)
{
  // check for a valid geometry
  if (!gemDigiModel_->getGeometry())
  {
    throw cms::Exception("Configuration")
      << "GEMDigiProducer::produce() - No GEMGeometry present in the configuration file."
      << "Add the service in the configuration file or remove the modules that require it.";
  }

  // get mix collection
  edm::Handle<CrossingFrame<PSimHit> > cf;
  iEvent.getByLabel("mix", inputCollection_, cf);
  
  std::auto_ptr<MixCollection<PSimHit> > hits(new MixCollection<PSimHit>(cf.product()));
  
  // Create empty output
  std::auto_ptr<GEMDigiCollection> digis(new GEMDigiCollection());
  std::auto_ptr<StripDigiSimLinks> digiSimLinks(new StripDigiSimLinks());

  // arrange the hits by eta partition
  std::map<uint32_t, edm::PSimHitContainer> hitMap;
  for(auto &hit: *hits)
  {
    hitMap[hit.detUnitId()].push_back(hit);
  }

  // simulate signal and noise for each eta partition
  const auto & etaPartitions(gemDigiModel_->getGeometry()->etaPartitions());
  for(auto &roll: etaPartitions)
  {
    const GEMDetId detId(roll->id());
    const uint32_t rawId(detId.rawId());
    const auto & simHits(hitMap[rawId]);

    LogDebug("GEMDigiProducer") 
      << "GEMDigiProducer: found " << simHits.size() << " hit(s) in eta partition" << rawId;
    
    gemDigiModel_->simulateSignal(roll, simHits);
    gemDigiModel_->simulateNoise(roll);
    //     gemDigiModel_->fillDigis(rawId, *digis);
    //     (*digiSimLinks).insert(gemDigiModel_->stripDigiSimLinks());
  }
  
  iEvent.put(digis);
  iEvent.put(digiSimLinks,"GEM");
}


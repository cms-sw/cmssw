#include "SimMuon/GEMDigitizer/src/GEMDigitizer.h"
#include "SimMuon/GEMDigitizer/src/GEMSimFactory.h"
#include "SimMuon/GEMDigitizer/src/GEMSim.h"
#include "SimMuon/GEMDigitizer/src/GEMSimSetUp.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"


GEMDigitizer::GEMDigitizer(const edm::ParameterSet& config, CLHEP::HepRandomEngine& eng)
{
  modelName_ = config.getParameter<std::string>("digiModel");
  gemSim_ = GEMSimFactory::get()->create(modelName_, config.getParameter<edm::ParameterSet>("digiModelConfig"));
  gemSim_->setRandomEngine(eng);
}


GEMDigitizer::~GEMDigitizer()
{
  if(gemSim_) delete gemSim_;
}


void GEMDigitizer::doAction(MixCollection<PSimHit> & simHits, 
                            GEMDigiCollection & digis,
                            StripDigiSimLinks & digiSimLinks)
{
  gemSim_->setGEMSimSetUp(simSetUp_);

  // arrange the hits by roll
  std::map<int, edm::PSimHitContainer> hitMap;
  for(auto &hit: simHits)
  {
    hitMap[hit.detUnitId()].push_back(hit);
  }

  if ( ! geometry_)
  {
    throw cms::Exception("Configuration")
      << "GEMDigitizer requires the GEMGeometry which is not present in the configuration file.\n"
         "You must add the service\n in the configuration file or remove the modules that require it.";
  }

  auto etaPartitions = geometry_->etaPartitions() ;
  for(auto &p: etaPartitions)
  {
    const auto & partSimHits = hitMap[p->id()];

    //LogDebug("GEMDigitizer") << "GEMDigitizer: found " << partSimHits.size() <<" hit(s) in the eta partition";
    
    gemSim_->simulate(p, partSimHits);
    gemSim_->simulateNoise(p);
    gemSim_->fillDigis(p->id(), digis);
    digiSimLinks.insert(gemSim_->stripDigiSimLinks());
  }
}


const GEMEtaPartition * GEMDigitizer::findDet(int detId) const
{
  assert(geometry_ != 0);
  return dynamic_cast<const GEMEtaPartition *>(geometry_->idToDetUnit(GEMDetId(detId)));
}


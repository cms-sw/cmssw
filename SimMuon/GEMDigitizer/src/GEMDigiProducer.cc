#include "SimMuon/GEMDigitizer/src/GEMDigiProducer.h"
#include "SimMuon/GEMDigitizer/src/GEMSimSetUp.h"
#include "SimMuon/GEMDigitizer/src/GEMSynchronizer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "CLHEP/Random/RandomEngine.h"

#include "CondFormats/DataRecord/interface/RPCStripNoisesRcd.h"
#include "CondFormats/RPCObjects/interface/RPCClusterSize.h"
#include "CondFormats/DataRecord/interface/RPCClusterSizeRcd.h"

#include <sstream>
#include <string>
#include <map>
#include <vector>


GEMDigiProducer::GEMDigiProducer(const edm::ParameterSet& ps)
{
  produces<GEMDigiCollection>();
  produces<StripDigiSimLinks>("GEM");

  //Name of Collection used for create the XF 
  collectionXF_ = ps.getParameter<std::string>("InputCollection");

  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable())
  {
   throw cms::Exception("Configuration")
     << "GEMDigiProducer::GEMDigiProducer() - RandomNumberGeneratorService is not present in configuration file.\n"
     << "Add the service in the configuration file or remove the modules that require it.";
  }
  CLHEP::HepRandomEngine& engine = rng->getEngine();

  gemSimSetUp_ =  new GEMSimSetUp(ps);
  digitizer_ = new GEMDigitizer(ps, engine);
}


GEMDigiProducer::~GEMDigiProducer()
{
  delete digitizer_;
  delete gemSimSetUp_;
}


void GEMDigiProducer::beginRun( const edm::Run& r, const edm::EventSetup& eventSetup)
{
  edm::ESHandle<GEMGeometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get( hGeom );
  const GEMGeometry *pGeom = &*hGeom;

  /*
  edm::ESHandle<RPCStripNoises> noiseRcd;
  eventSetup.get<RPCStripNoisesRcd>().get(noiseRcd);

  edm::ESHandle<RPCClusterSize> clsRcd;
  eventSetup.get<RPCClusterSizeRcd>().get(clsRcd);

  auto vnoise = noiseRcd->getVNoise();
  auto vcls = clsRcd->getCls();
  */
  std::vector<RPCStripNoises::NoiseItem> vnoise;
  std::vector<float> vcls;

  gemSimSetUp_->setGeometry( pGeom );
  if (vnoise.size()==0 && vcls.size()==0)
    gemSimSetUp_->setup();
  else
    gemSimSetUp_->setup(vnoise, vcls);
  
  digitizer_->setGeometry( pGeom );
  digitizer_->setGEMSimSetUp( gemSimSetUp_ );
}


void GEMDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{
  edm::Handle<CrossingFrame<PSimHit> > cf;
  e.getByLabel("mix", collectionXF_, cf);

  std::auto_ptr<MixCollection<PSimHit> > hits( new MixCollection<PSimHit>(cf.product()) );

  // Create empty output
  std::auto_ptr<GEMDigiCollection> pDigis(new GEMDigiCollection());
  std::auto_ptr<StripDigiSimLinks> digiSimLinks(new StripDigiSimLinks() );

  // run the digitizer
  digitizer_->digitize(*hits, *pDigis, *digiSimLinks);

  // store them in the event
  e.put(pDigis);
  e.put(digiSimLinks,"GEM");
}


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimMuon/CSCDigitizer/src/CSCDigiProducer.h"
#include "SimMuon/CSCDigitizer/src/CSCDigitizer.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"


CSCDigiProducer::CSCDigiProducer(const edm::ParameterSet& ps) {
  theDigitizer = new CSCDigitizer();
  produces<CSCWireDigiCollection>();
  produces<CSCStripDigiCollection>();
  produces<CSCComparatorDigiCollection>();

  theFakeHitsAreUsed = ps.getUntrackedParameter<bool>("useFakeHits", false);
}


CSCDigiProducer::~CSCDigiProducer() {
  delete theDigitizer;
}


void CSCDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {

  edm::Handle<edm::PSimHitContainer> hCscHits;
  e.getByLabel("r", "MuonCSCHits", hCscHits);
  edm::PSimHitContainer cscHits = *hCscHits;

  std::cout << "HITS SIZE " << cscHits.size() << std::endl;
  // Create empty output

  std::auto_ptr<CSCWireDigiCollection> pWireDigis(new CSCWireDigiCollection());
  std::auto_ptr<CSCStripDigiCollection> pStripDigis(new CSCStripDigiCollection());
  std::auto_ptr<CSCComparatorDigiCollection> pComparatorDigis(new CSCComparatorDigiCollection());

  // find the geometry & conditions for this event
  edm::ESHandle<TrackingGeometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get( hGeom );
  const TrackingGeometry *pGeom = &*hGeom;

  theDigitizer->setGeometry( pGeom );

  if(theFakeHitsAreUsed) {
    addFakeHits(pGeom, cscHits);
  }

  // find the magnetic field
  edm::ESHandle<MagneticField> magfield;
  eventSetup.get<IdealMagneticFieldRecord>().get(magfield);

  theDigitizer->setMagneticField(&*magfield);

  // run the digitizer
  theDigitizer->doAction(cscHits, *pWireDigis, *pStripDigis, *pComparatorDigis);


  // store them in the event
  e.put(pWireDigis);
  e.put(pStripDigis);
  e.put(pComparatorDigis);

}

void
CSCDigiProducer::addFakeHits(const TrackingGeometry * geom, 
                             edm::PSimHitContainer & hits) 
{
  
  Local3DPoint entry(0.,0.,-0.5);
  Local3DPoint exit(0.,0., 0.5);
  float pabs = 100.;
  double tof = 0.;
  double eloss = 0.0001;
  int particleType = 13;
  unsigned int trackId = 0;
  float theta = 0.;
  float phi = 0.;
  int firstDetId = geom->detIds().begin()->rawId();
  hits.push_back(PSimHit(entry, exit, pabs, tof, eloss, particleType, firstDetId, trackId, theta, phi));
}

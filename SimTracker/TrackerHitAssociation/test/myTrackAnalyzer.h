#ifndef myTrackAnalyzer_h
#define myTrackAnalyzer_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

//add simhit info
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
//--- for SimHit
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

//simtrack
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"


#include <iostream>
#include <string>
#include <map>
#include <set>

//using namespace edm;
using namespace std;

class SiStripHitAssociator;

class myTrackAnalyzer : public edm::EDAnalyzer {

  
 public:
  typedef map<const TrackingRecHit*, int > sim_id_map;
  sim_id_map SimIdMap;

  myTrackAnalyzer(const edm::ParameterSet& conf): conf_(conf) {}

  ~myTrackAnalyzer(){}

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);

 private:
const   edm::ParameterSet& conf_;

 };

#endif

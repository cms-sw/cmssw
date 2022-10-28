#ifndef myTrackAnalyzer_h
#define myTrackAnalyzer_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

//add simhit info
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
//--- for SimHit
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

//simtrack
//#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include <iostream>
#include <string>
#include <map>
#include <set>

class SiStripHitAssociator;

class myTrackAnalyzer : public edm::one::EDAnalyzer<> {
public:
  typedef std::map<const TrackingRecHit*, int> sim_id_map;
  sim_id_map SimIdMap;

  explicit myTrackAnalyzer(const edm::ParameterSet& conf);

  ~myTrackAnalyzer() override = default;

  void analyze(const edm::Event& event, const edm::EventSetup& setup) override;

private:
  TrackerHitAssociator::Config trackerHitAssociatorConfig_;
  const bool doPixel_, doStrip_;
  const edm::InputTag trackCollectionTag_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tokGeo_;
  const edm::EDGetTokenT<reco::TrackCollection> tokTrack_;
  const edm::EDGetTokenT<edm::SimTrackContainer> tokSimTk_;
  const edm::EDGetTokenT<edm::SimVertexContainer> tokSimVtx_;
};

#endif

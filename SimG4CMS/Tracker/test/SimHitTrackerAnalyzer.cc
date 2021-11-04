// -*- C++ -*-
//
// Package:    SimHitTrackerAnalyzer
// Class:      SimHitTrackerAnalyzer
//
/**\class SimHitTrackerAnalyzer SimHitTrackerAnalyzer.cc test/SimHitTrackerAnalyzer/src/SimHitTrackerAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tommaso Boccali
//         Created:  Tue Jul 26 08:47:57 CEST 2005
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
//
// class decleration
//

class SimHitTrackerAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit SimHitTrackerAnalyzer(const edm::ParameterSet&);
  ~SimHitTrackerAnalyzer() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------
  const std::string HepMCLabel;
  const std::string SimTkLabel;
  const std::string SimVtxLabel;
  const std::string SimHitLabel;
  edm::EDGetTokenT<edm::SimTrackContainer> tokSimTk_;
  edm::EDGetTokenT<edm::SimVertexContainer> tokSimVtx_;
  edm::EDGetTokenT<edm::PSimHitContainer> tokPixelBarrelHitsLowTof_;
  edm::EDGetTokenT<edm::PSimHitContainer> tokPixelBarrelHitsHighTof_;
  edm::EDGetTokenT<edm::PSimHitContainer> tokPixelEndcapHitsLowTof_;
  edm::EDGetTokenT<edm::PSimHitContainer> tokPixelEndcapHitsHighTof_;
  edm::EDGetTokenT<edm::PSimHitContainer> tokTIBHitsLowTof_;
  edm::EDGetTokenT<edm::PSimHitContainer> tokTIBHitsHighTof_;
  edm::EDGetTokenT<edm::PSimHitContainer> tokTIDHitsLowTof_;
  edm::EDGetTokenT<edm::PSimHitContainer> tokTIDHitsHighTof_;
  edm::EDGetTokenT<edm::PSimHitContainer> tokTOBHitsLowTof_;
  edm::EDGetTokenT<edm::PSimHitContainer> tokTOBHitsHighTof_;
  edm::EDGetTokenT<edm::PSimHitContainer> tokTECHitsLowTof_;
  edm::EDGetTokenT<edm::PSimHitContainer> tokTECHitsHighTof_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SimHitTrackerAnalyzer::SimHitTrackerAnalyzer(const edm::ParameterSet& iConfig)
    : HepMCLabel(iConfig.getUntrackedParameter("moduleLabelMC", std::string("FlatRandomPtGunProducer"))),
      SimTkLabel(iConfig.getUntrackedParameter("moduleLabelTk", std::string("g4SimHits"))),
      SimVtxLabel(iConfig.getUntrackedParameter("moduleLabelVtx", std::string("g4SimHits"))),
      SimHitLabel(iConfig.getUntrackedParameter("moduleLabelHit", std::string("g4SimHits"))) {
  //now do what ever initialization is needed
  tokSimTk_ = consumes<edm::SimTrackContainer>(SimTkLabel);
  tokSimVtx_ = consumes<edm::SimVertexContainer>(SimVtxLabel);
  tokPixelBarrelHitsLowTof_ =
      consumes<edm::PSimHitContainer>(edm::InputTag(SimHitLabel, "TrackerHitsPixelBarrelLowTof"));
  tokPixelBarrelHitsHighTof_ =
      consumes<edm::PSimHitContainer>(edm::InputTag(SimHitLabel, "TrackerHitsPixelBarrelHighTof"));
  tokPixelEndcapHitsLowTof_ =
      consumes<edm::PSimHitContainer>(edm::InputTag(SimHitLabel, "TrackerHitsPixelEndcapLowTof"));
  tokPixelEndcapHitsHighTof_ =
      consumes<edm::PSimHitContainer>(edm::InputTag(SimHitLabel, "TrackerHitsPixelEndcapHighTof"));
  tokTIBHitsLowTof_ = consumes<edm::PSimHitContainer>(edm::InputTag(SimHitLabel, "TrackerHitsTIBLowTof"));
  tokTIBHitsHighTof_ = consumes<edm::PSimHitContainer>(edm::InputTag(SimHitLabel, "TrackerHitsTIBHighTof"));
  tokTIDHitsLowTof_ = consumes<edm::PSimHitContainer>(edm::InputTag(SimHitLabel, "TrackerHitsTIDLowTof"));
  tokTIDHitsHighTof_ = consumes<edm::PSimHitContainer>(edm::InputTag(SimHitLabel, "TrackerHitsTIDHighTof"));
  tokTOBHitsLowTof_ = consumes<edm::PSimHitContainer>(edm::InputTag(SimHitLabel, "TrackerHitsTOBLowTof"));
  tokTOBHitsHighTof_ = consumes<edm::PSimHitContainer>(edm::InputTag(SimHitLabel, "TrackerHitsTOBHighTof"));
  tokTECHitsLowTof_ = consumes<edm::PSimHitContainer>(edm::InputTag(SimHitLabel, "TrackerHitsTECLowTof"));
  tokTECHitsHighTof_ = consumes<edm::PSimHitContainer>(edm::InputTag(SimHitLabel, "TrackerHitsTECHighTof"));
}

SimHitTrackerAnalyzer::~SimHitTrackerAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void SimHitTrackerAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::vector<PSimHit> theTrackerHits;
  std::vector<SimTrack> theSimTracks;
  std::vector<SimVertex> theSimVertexes;

  //   Handle<HepMCProduct> MCEvt;
  auto SimTk = iEvent.getHandle(tokSimTk_);
  auto SimVtx = iEvent.getHandle(tokSimVtx_);
  auto PixelBarrelHitsLowTof = iEvent.getHandle(tokPixelBarrelHitsLowTof_);
  auto PixelBarrelHitsHighTof = iEvent.getHandle(tokPixelBarrelHitsHighTof_);
  auto PixelEndcapHitsLowTof = iEvent.getHandle(tokPixelEndcapHitsLowTof_);
  auto PixelEndcapHitsHighTof = iEvent.getHandle(tokPixelEndcapHitsHighTof_);
  auto TIBHitsLowTof = iEvent.getHandle(tokTIBHitsLowTof_);
  auto TIBHitsHighTof = iEvent.getHandle(tokTIBHitsHighTof_);
  auto TIDHitsLowTof = iEvent.getHandle(tokTIDHitsLowTof_);
  auto TIDHitsHighTof = iEvent.getHandle(tokTIDHitsHighTof_);
  auto TOBHitsLowTof = iEvent.getHandle(tokTOBHitsLowTof_);
  auto TOBHitsHighTof = iEvent.getHandle(tokTOBHitsHighTof_);
  auto TECHitsLowTof = iEvent.getHandle(tokTECHitsLowTof_);
  auto TECHitsHighTof = iEvent.getHandle(tokTECHitsHighTof_);

  theSimTracks.insert(theSimTracks.end(), SimTk->begin(), SimTk->end());
  theSimVertexes.insert(theSimVertexes.end(), SimVtx->begin(), SimVtx->end());
  theTrackerHits.insert(theTrackerHits.end(), PixelBarrelHitsLowTof->begin(), PixelBarrelHitsLowTof->end());
  theTrackerHits.insert(theTrackerHits.end(), PixelBarrelHitsHighTof->begin(), PixelBarrelHitsHighTof->end());
  theTrackerHits.insert(theTrackerHits.end(), PixelEndcapHitsLowTof->begin(), PixelEndcapHitsLowTof->end());
  theTrackerHits.insert(theTrackerHits.end(), PixelEndcapHitsHighTof->begin(), PixelEndcapHitsHighTof->end());
  theTrackerHits.insert(theTrackerHits.end(), TIBHitsLowTof->begin(), TIBHitsLowTof->end());
  theTrackerHits.insert(theTrackerHits.end(), TIBHitsHighTof->begin(), TIBHitsHighTof->end());
  theTrackerHits.insert(theTrackerHits.end(), TIDHitsLowTof->begin(), TIDHitsLowTof->end());
  theTrackerHits.insert(theTrackerHits.end(), TIDHitsHighTof->begin(), TIDHitsHighTof->end());
  theTrackerHits.insert(theTrackerHits.end(), TOBHitsLowTof->begin(), TOBHitsLowTof->end());
  theTrackerHits.insert(theTrackerHits.end(), TOBHitsHighTof->begin(), TOBHitsHighTof->end());
  theTrackerHits.insert(theTrackerHits.end(), TECHitsLowTof->begin(), TECHitsLowTof->end());
  theTrackerHits.insert(theTrackerHits.end(), TECHitsHighTof->begin(), TECHitsHighTof->end());

  /*
   Hepmc::GenEvent * myGenEvent = new  HepMC::GenEvent(*(MCEvt->GetEvent()));
   
   for ( HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
	 p != myGenEvent->particles_end(); ++p ) {
     edm::LogInfo("TrackerSimInfoAnalyzer")<< "Particle type form MC = "<< abs((*p)->pdg_id()) ; 
     edm::LogInfo("TrackerSimInfoAnalyzer")<< "Particle momentum Pt form MC = "<< (*p)->momentum().perp() ;  
   }
   */

  for (std::vector<SimTrack>::iterator isimtk = theSimTracks.begin(); isimtk != theSimTracks.end(); ++isimtk) {
    edm::LogInfo("TrackerSimInfoAnalyzer") << " Track momentum  x = " << isimtk->momentum().x()
                                           << " y = " << isimtk->momentum().y() << " z = " << isimtk->momentum().z();
    edm::LogInfo("TrackerSimInfoAnalyzer") << " Track momentum Ptx = " << std::sqrt(isimtk->momentum().perp2());
  }

  for (std::vector<SimVertex>::iterator isimvtx = theSimVertexes.begin(); isimvtx != theSimVertexes.end(); ++isimvtx) {
    edm::LogInfo("TrackerSimInfoAnalyzer") << " Vertex position  x = " << isimvtx->position().x()
                                           << " y = " << isimvtx->position().y() << " z = " << isimvtx->position().z();
  }

  std::map<unsigned int, std::vector<PSimHit>, std::less<unsigned int> > SimHitMap;

  for (std::vector<PSimHit>::iterator isim = theTrackerHits.begin(); isim != theTrackerHits.end(); ++isim) {
    SimHitMap[(*isim).detUnitId()].push_back((*isim));
    edm::LogInfo("TrackerSimInfoAnalyzer")
        << " SimHit position  x = " << isim->localPosition().x() << " y = " << isim->localPosition().y()
        << " z = " << isim->localPosition().z();
    edm::LogInfo("TrackerSimInfoAnalyzer") << " SimHit DetID = " << isim->detUnitId();
    edm::LogInfo("TrackerSimInfoAnalyzer") << " Time of flight = " << isim->timeOfFlight();
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(SimHitTrackerAnalyzer);

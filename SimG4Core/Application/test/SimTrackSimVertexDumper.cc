// -*- C++ -*-
//
// Package:    SimTrackerDumper
// Class:      SimTrackSimVertexDumper
//
/*
 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
//

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"

class SimTrackSimVertexDumper : public edm::EDAnalyzer {
public:
  explicit SimTrackSimVertexDumper(const edm::ParameterSet&);
  ~SimTrackSimVertexDumper() override{};

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override{};
  void endJob() override{};

private:
  edm::InputTag hepmcLabel;
  edm::InputTag simtkLabel;
  edm::InputTag simvtxLabel;
  bool dumpHepMC;

  edm::EDGetTokenT<edm::HepMCProduct> hepmcToken;
  edm::EDGetTokenT<edm::SimTrackContainer> simTrackToken;
  edm::EDGetTokenT<edm::SimVertexContainer> simVertexToken;
};

SimTrackSimVertexDumper::SimTrackSimVertexDumper(const edm::ParameterSet& iConfig)
    : hepmcLabel(iConfig.getParameter<edm::InputTag>("moduleLabelHepMC")),
      simtkLabel(iConfig.getParameter<edm::InputTag>("moduleLabelTk")),
      simvtxLabel(iConfig.getParameter<edm::InputTag>("moduleLabelVtx")),
      dumpHepMC(iConfig.getUntrackedParameter<bool>("dumpHepMC", "false")) {
  hepmcToken = consumes<edm::HepMCProduct>(hepmcLabel);
  simTrackToken = consumes<edm::SimTrackContainer>(simtkLabel);
  simVertexToken = consumes<edm::SimVertexContainer>(simvtxLabel);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void SimTrackSimVertexDumper::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace HepMC;

  std::vector<SimTrack> theSimTracks;
  std::vector<SimVertex> theSimVertexes;

  edm::Handle<edm::HepMCProduct> MCEvt;
  edm::Handle<edm::SimTrackContainer> SimTk;
  edm::Handle<edm::SimVertexContainer> SimVtx;

  iEvent.getByToken(hepmcToken, MCEvt);
  const HepMC::GenEvent* evt = MCEvt->GetEvent();

  iEvent.getByToken(simTrackToken, SimTk);
  iEvent.getByToken(simVertexToken, SimVtx);

  theSimTracks.insert(theSimTracks.end(), SimTk->begin(), SimTk->end());
  theSimVertexes.insert(theSimVertexes.end(), SimVtx->begin(), SimVtx->end());

  std::cout << "\n SimVertex / SimTrack structure dump \n" << std::endl;
  std::cout << " SimVertex in the event = " << theSimVertexes.size() << std::endl;
  std::cout << " SimTracks in the event = " << theSimTracks.size() << std::endl;
  std::cout << "\n" << std::endl;
  for (unsigned int isimvtx = 0; isimvtx < theSimVertexes.size(); isimvtx++) {
    std::cout << "SimVertex " << isimvtx << " = " << theSimVertexes[isimvtx] << "\n" << std::endl;
    for (unsigned int isimtk = 0; isimtk < theSimTracks.size(); isimtk++) {
      if (theSimTracks[isimtk].vertIndex() >= 0 && std::abs(theSimTracks[isimtk].vertIndex()) == (int)isimvtx) {
        std::cout << "  SimTrack " << isimtk << " = " << theSimTracks[isimtk]
                  << " Track Id = " << theSimTracks[isimtk].trackId() << std::endl;

        // for debugging purposes
        if (dumpHepMC) {
          if (theSimTracks[isimtk].genpartIndex() != -1) {
            HepMC::GenParticle* part = evt->barcode_to_particle(theSimTracks[isimtk].genpartIndex());
            if (part) {
              std::cout << "  ---> Corresponding to HepMC particle " << *part << std::endl;
            } else {
              std::cout << " ---> Corresponding HepMC particle to barcode " << theSimTracks[isimtk].genpartIndex()
                        << " not in selected event " << std::endl;
            }
          }
        }
      }
    }
    std::cout << "\n" << std::endl;
  }

  for (std::vector<SimTrack>::iterator isimtk = theSimTracks.begin(); isimtk != theSimTracks.end(); ++isimtk) {
    if (isimtk->noVertex()) {
      std::cout << "SimTrack without an associated Vertex = " << *isimtk << std::endl;
    }
  }

  return;
}

#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in
DEFINE_FWK_MODULE(SimTrackSimVertexDumper);

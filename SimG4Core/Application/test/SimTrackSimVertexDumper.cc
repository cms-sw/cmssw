// system include files
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/ValidHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"

class SimTrackSimVertexDumper : public edm::one::EDAnalyzer<> {
public:
  explicit SimTrackSimVertexDumper(const edm::ParameterSet&);
  ~SimTrackSimVertexDumper() override{};

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override{};
  void endJob() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<edm::HepMCProduct> hepmcToken_;
  edm::EDGetTokenT<edm::SimTrackContainer> simTrackToken_;
  edm::EDGetTokenT<edm::SimVertexContainer> simVertexToken_;
  bool dumpHepMC_;
};

SimTrackSimVertexDumper::SimTrackSimVertexDumper(const edm::ParameterSet& iConfig)
    : hepmcToken_(consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("moduleLabelHepMC"))),
      simTrackToken_(consumes<edm::SimTrackContainer>(iConfig.getParameter<edm::InputTag>("moduleLabelTk"))),
      simVertexToken_(consumes<edm::SimVertexContainer>(iConfig.getParameter<edm::InputTag>("moduleLabelVtx"))),
      dumpHepMC_(iConfig.getUntrackedParameter<bool>("dumpHepMC")) {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void SimTrackSimVertexDumper::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace HepMC;

  std::vector<SimTrack> theSimTracks;
  std::vector<SimVertex> theSimVertexes;

  auto MCEvt = edm::makeValid(iEvent.getHandle(hepmcToken_));
  const HepMC::GenEvent* evt = MCEvt->GetEvent();

  auto SimTk = edm::makeValid(iEvent.getHandle(simTrackToken_));
  auto SimVtx = edm::makeValid(iEvent.getHandle(simVertexToken_));

  theSimTracks.insert(theSimTracks.end(), SimTk->begin(), SimTk->end());
  theSimVertexes.insert(theSimVertexes.end(), SimVtx->begin(), SimVtx->end());

  edm::LogPrint("DumpTkVtx") << "\n SimVertex / SimTrack structure dump \n";
  edm::LogPrint("DumpTkVtx") << " SimVertex in the event = " << theSimVertexes.size();
  edm::LogPrint("DumpTkVtx") << " SimTracks in the event = " << theSimTracks.size();
  edm::LogPrint("DumpTkVtx") << "\n";
  for (unsigned int isimvtx = 0; isimvtx < theSimVertexes.size(); isimvtx++) {
    edm::LogPrint("DumpTkVtx") << "SimVertex " << isimvtx << " = " << theSimVertexes[isimvtx] << "\n";
    for (unsigned int isimtk = 0; isimtk < theSimTracks.size(); isimtk++) {
      if (theSimTracks[isimtk].vertIndex() >= 0 && std::abs(theSimTracks[isimtk].vertIndex()) == (int)isimvtx) {
        edm::LogPrint("DumpTkVtx") << "  SimTrack " << isimtk << " = " << theSimTracks[isimtk]
                                   << " Track Id = " << theSimTracks[isimtk].trackId();

        // for debugging purposes
        if (dumpHepMC_) {
          if (theSimTracks[isimtk].genpartIndex() != -1) {
            HepMC::GenParticle* part = evt->barcode_to_particle(theSimTracks[isimtk].genpartIndex());
            if (part) {
              edm::LogPrint("DumpTkVtx") << "  ---> Corresponding to HepMC particle " << *part;
            } else {
              edm::LogPrint("DumpTkVtx") << " ---> Corresponding HepMC particle to barcode "
                                         << theSimTracks[isimtk].genpartIndex() << " not in selected event ";
            }
          }
        }
      }
    }
    edm::LogPrint("DumpTkVtx") << "\n";
  }

  for (std::vector<SimTrack>::iterator isimtk = theSimTracks.begin(); isimtk != theSimTracks.end(); ++isimtk) {
    if (isimtk->noVertex()) {
      edm::LogPrint("DumpTkVtx") << "SimTrack without an associated Vertex = " << *isimtk;
    }
  }

  return;
}

void SimTrackSimVertexDumper::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("moduleLabelHepMC", edm::InputTag("generatorSmeared"))
      ->setComment("Input generated HepMC event after vtx smearing");
  desc.add<edm::InputTag>("moduleLabelTk", edm::InputTag("g4SimHits"))
      ->setComment("Module for input SimTrack collection");
  desc.add<edm::InputTag>("moduleLabelVtx", edm::InputTag("g4SimHits"))
      ->setComment("Module for input SimVertex collection");
  desc.addUntracked<bool>("dumpHepMC", false);
  descriptions.add("simTrackSimVertexDumper", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in
DEFINE_FWK_MODULE(SimTrackSimVertexDumper);

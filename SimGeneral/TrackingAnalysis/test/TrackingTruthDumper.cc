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

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

class TrackingTruthDumper : public edm::one::EDAnalyzer<> {
public:
  explicit TrackingTruthDumper(const edm::ParameterSet&);
  ~TrackingTruthDumper() override{};

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override{};
  void endJob() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<TrackingParticleCollection> simTPToken_;
  edm::EDGetTokenT<TrackingVertexCollection> simTVToken_;

  bool dumpVtx_;
  bool dumpTk_;
};

TrackingTruthDumper::TrackingTruthDumper(const edm::ParameterSet& iConfig)
    : simTPToken_(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("moduleLabelTk"))),
      simTVToken_(consumes<TrackingVertexCollection>(iConfig.getParameter<edm::InputTag>("moduleLabelVtx"))),
      dumpVtx_(iConfig.getUntrackedParameter<bool>("dumpVtx")),
      dumpTk_(iConfig.getUntrackedParameter<bool>("dumpTk")) {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void TrackingTruthDumper::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto SimTk = edm::makeValid(iEvent.getHandle(simTPToken_));
  auto SimVtx = edm::makeValid(iEvent.getHandle(simTVToken_));

  edm::LogPrint("DumpTkVtx") << "\n SimVertex / SimTrack structure dump \n";
  edm::LogPrint("DumpTkVtx") << " SimVertex in the event = " << (*SimTk).size();
  edm::LogPrint("DumpTkVtx") << " SimTracks in the event = " << (*SimVtx).size();
  edm::LogPrint("DumpTkVtx") << "\n";
  size_t isimvtx(0);
  size_t isimtk(0);

  if (dumpVtx_) {
    for (const auto& vtx : *SimVtx) {
      edm::LogPrint("DumpTkVtx") << "TrackingVertex " << isimvtx << " = " << vtx << "\n";
      edm::LogPrint("DumpTkVtx") << "TPs of this vertex: \n";
      isimtk = 0;
      for (const auto& tk : vtx.daughterTracks()) {
        edm::LogPrint("DumpTkVtx") << "TrackingParticle " << isimtk << " = " << *tk << "\n";
        isimtk++;
      }
      edm::LogPrint("DumpTkVtx") << "\n";
      isimvtx++;
    }
  }

  if (dumpTk_) {
    isimtk = 0;
    for (const auto& tk : *SimTk) {
      bool isMerged = tk.g4Tracks().size() > 1;
      edm::LogPrint("DumpTkVtx") << "TrackingParticle " << isimtk << " isMerged " << isMerged << " = " << tk << "\n";
      isimtk++;
    }
  }

  return;
}

void TrackingTruthDumper::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("moduleLabelTk", edm::InputTag("mix", "MergedTrackTruth"))
      ->setComment("Module for input TrackingParticle collection");
  desc.add<edm::InputTag>("moduleLabelVtx", edm::InputTag("mix", "MergedTrackTruth"))
      ->setComment("Module for input TrackingVertex collection");
  desc.addUntracked<bool>("dumpVtx", true);
  desc.addUntracked<bool>("dumpTk", true);
  descriptions.add("trackingTruthDumper", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in
DEFINE_FWK_MODULE(TrackingTruthDumper);

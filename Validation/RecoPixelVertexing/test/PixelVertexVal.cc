#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "HepMC/GenEvent.h"
#include "HepMC/GenVertex.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "TDirectory.h"
#include "TFile.h"
#include "TTree.h"
#include <TH1.h>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>

using namespace std;

class PixelVertexVal : public edm::EDAnalyzer {
public:
  explicit PixelVertexVal(const edm::ParameterSet &conf);
  ~PixelVertexVal() override;
  void beginJob() override;
  void analyze(const edm::Event &ev, const edm::EventSetup &es) override;
  void endJob() override;

private:
  edm::ParameterSet conf_;
  int verbose_;
  std::string file_;
  std::map<std::string, TH1 *> h;
  edm::EDGetTokenT<reco::TrackCollection> trackCollectionToken_;
  edm::EDGetTokenT<reco::VertexCollection> vertexCollectionToken_;
  edm::EDGetTokenT<edm::SimVertexContainer> simVertexContainerToken_;
};

PixelVertexVal::PixelVertexVal(const edm::ParameterSet &conf)
    : verbose_(conf.getUntrackedParameter<unsigned int>("Verbosity",
                                                        0))  // How noisy?
      ,
      file_(conf.getUntrackedParameter<std::string>("HistoFile", "pixelVertexHistos.root")),
      h(),
      trackCollectionToken_(
          consumes<reco::TrackCollection>(edm::InputTag(conf.getParameter<std::string>("TrackCollection")))),
      vertexCollectionToken_(
          consumes<reco::VertexCollection>(edm::InputTag(conf.getParameter<std::string>("VertexCollection")))),
      simVertexContainerToken_(consumes<edm::SimVertexContainer>(conf.getParameter<edm::InputTag>("simG4"))) {
  edm::LogInfo("PixelVertexVal") << " CTOR";
}

PixelVertexVal::~PixelVertexVal() { edm::LogInfo("PixelVertexVal") << " DTOR"; }

void PixelVertexVal::beginJob() {
  // validation histos
  h["h_Nbvtx"] = new TH1F("h_nbvtx", "nb vertices in event", 16, 0., 16.);
  h["h_Nbtrks"] = new TH1F("h_Nbtrks", "nb tracks in PV", 100, 0., 100.);
  h["h_ResZ"] = new TH1F("resz", "residual z", 100, -0.1, 0.1);
  h["h_PullZ"] = new TH1F("pullz", "pull z", 100, -25., 25.);
  h["h_TrkRes"] = new TH1F("h_TrkRes", "h_TrkRes", 100, -0.2, 0.2);
  h["h_Eff"] = new TH1F("h_Etff", "h_Etff", 10, -1., 9.);
}

void PixelVertexVal::analyze(const edm::Event &ev, const edm::EventSetup &es) {
  if (verbose_ > 0)
    cout << "------------------------------------------------" << endl;
  cout << "*** PixelVertexVal, analyze event: " << ev.id() << endl;
  edm::Handle<reco::TrackCollection> trackCollection;
  ev.getByToken(trackCollectionToken_, trackCollection);
  const reco::TrackCollection tracks = *(trackCollection.product());

  reco::TrackRefVector trks;

  //  if (verbose_ > 0) {
  //    std::cout << *(trackCollection.provenance()) << std::endl;
  //    cout << "Reconstructed "<< tracks.size() << " tracks" << std::endl;
  //  }

  edm::Handle<reco::VertexCollection> vertexCollection;
  ev.getByToken(vertexCollectionToken_, vertexCollection);
  const reco::VertexCollection vertexes = *(vertexCollection.product());
  if (verbose_ > 0) {
    //    std::cout << *(vertexCollection.provenance()) << std::endl;
    cout << "Reconstructed " << vertexes.size() << " vertexes" << std::endl;
  }

  edm::Handle<edm::SimVertexContainer> simVtcs;
  ev.getByToken(simVertexContainerToken_, simVtcs);
  if (verbose_ > 0) {
    cout << "simulated vertices: " << simVtcs->size() << std::endl;
  }

  bool hasPV = (!simVtcs->empty());
  if (!hasPV)
    cout << "Event without PV!, skip" << endl;
  float z_PV = hasPV ? (*simVtcs)[0].position().z() : 0.;

  int nVertices = vertexes.size();
  h["h_Nbvtx"]->Fill(nVertices);
  if (nVertices > 0)
    h["h_Nbtrks"]->Fill(vertexes[0].tracksSize());

  int ivtx_matched = -1;
  float min_dist = 0.1;
  for (int idx = 0; idx < nVertices; idx++) {
    float dz = vertexes[idx].position().z() - z_PV;
    if (fabs(dz) < min_dist) {
      ivtx_matched = idx;
      min_dist = fabs(dz);
    }
  }
  h["h_Eff"]->Fill(ivtx_matched + 0.00001);

  if (ivtx_matched == 0) {
    const reco::Vertex &pv = vertexes[ivtx_matched];
    float dz = pv.position().z() - z_PV;
    h["h_ResZ"]->Fill(dz);
    h["h_PullZ"]->Fill(dz / pv.zError());

    for (reco::Vertex::trackRef_iterator it = pv.tracks_begin(); it != pv.tracks_end(); it++) {
      //    for (reco::Vertex::track_iterator it=pv.tracks_begin(); it !=
      //    pv.tracks_end(); it++) { for (reco::TrackRefVector::iterator
      //    it=pv.tracks_begin(); it != pv.tracks_end(); it++) {
      // h["h_TrkRes"]->Fill((*it)->dz());
      h["h_TrkRes"]->Fill((*it)->vertex().z() - pv.position().z());
    }
  }
}

void PixelVertexVal::endJob() {
  TFile rootFile(file_.c_str(), "RECREATE");
  for (std::map<std::string, TH1 *>::const_iterator ih = h.begin(); ih != h.end(); ++ih) {
    TH1 *histo = (*ih).second;
    histo->Write();
    delete histo;
  }
  rootFile.Close();
  h.clear();
}

DEFINE_FWK_MODULE(PixelVertexVal);

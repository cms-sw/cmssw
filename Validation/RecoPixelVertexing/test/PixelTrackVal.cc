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
#include "FWCore/Utilities/interface/isFinite.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenVertex.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "TFile.h"
#include "TObjArray.h"
#include <TH1.h>
#include <cmath>
#include <iostream>
#include <vector>

template <class T>
T sqr(T t) {
  return t * t;
}

class PixelTrackVal : public edm::EDAnalyzer {
public:
  explicit PixelTrackVal(const edm::ParameterSet &conf);
  ~PixelTrackVal() override;
  void beginJob() override;
  void analyze(const edm::Event &ev, const edm::EventSetup &es) override;
  void endJob() override;

private:
  int verbose_;
  std::string file_;
  TObjArray hList;
  edm::EDGetTokenT<reco::TrackCollection> trackCollectionToken_;
  edm::EDGetTokenT<edm::SimTrackContainer> simTrackContainerToken_;
  edm::EDGetTokenT<edm::SimVertexContainer> simVertexContainerToken_;
};

PixelTrackVal::PixelTrackVal(const edm::ParameterSet &conf)
    : verbose_(conf.getUntrackedParameter<unsigned int>("Verbosity",
                                                        0))  // How noisy?
      ,
      file_(conf.getUntrackedParameter<std::string>("HistoFile", "pixelTrackHistos.root")),
      hList(0),
      trackCollectionToken_(
          consumes<reco::TrackCollection>(edm::InputTag(conf.getParameter<std::string>("TrackCollection")))),
      simTrackContainerToken_(consumes<edm::SimTrackContainer>(conf.getParameter<edm::InputTag>("simG4"))),
      simVertexContainerToken_(consumes<edm::SimVertexContainer>(conf.getParameter<edm::InputTag>("simG4"))) {
  edm::LogInfo("PixelTrackVal") << " CTOR";
}

PixelTrackVal::~PixelTrackVal() { edm::LogInfo("PixelTrackVal") << " DTOR"; }

void PixelTrackVal::beginJob() {
  hList.Add(new TH1F("h_Pt", "h_Pt", 31, -2., 1.2));
  hList.Add(new TH1F("h_dR", "h_dR", 30, 0., 0.06));
  hList.Add(new TH1F("h_TIP", "h_TIP", 100, -0.1, 0.1));
  hList.Add(new TH1F("h_VtxZ", "h_VtxZ", 100, -0.1, 0.1));
  hList.Add(new TH1F("h_VtxZ_Pull", "h_VtxZ_Pull", 80, 0., 8));
  hList.Add(new TH1F("h_Nan", "Illegal values for x,y,z,xx,xy,xz,yy,yz,zz", 9, 0.5, 9.5));
  hList.SetOwner();
}

void PixelTrackVal::analyze(const edm::Event &ev, const edm::EventSetup &es) {
  std::cout << "*** PixelTrackVal, analyze event: " << ev.id() << std::endl;

  //------------------------ simulated tracks
  edm::Handle<reco::TrackCollection> trackCollection;
  ev.getByToken(trackCollectionToken_, trackCollection);
  const reco::TrackCollection tracks = *(trackCollection.product());

  typedef reco::TrackCollection::const_iterator IT;

  if (verbose_ > 0) {
    //    std::cout << *(trackCollection.provenance()) << std::endl;
    std::cout << "Reconstructed " << tracks.size() << " tracks" << std::endl;
  }

  for (unsigned int idx = 0; idx < tracks.size(); idx++) {
    const reco::Track *it = &tracks[idx];
    TH1 *h = static_cast<TH1 *>(hList.FindObject("h_Nan"));
    h->Fill(1., edm::isNotFinite(it->momentum().x()) * 1.);
    h->Fill(2., edm::isNotFinite(it->momentum().y()) * 1.);
    h->Fill(3., edm::isNotFinite(it->momentum().z()) * 1.);

    bool problem = false;
    int index = 3;
    for (int i = 0; i != 3; i++) {
      for (int j = i; j != 3; j++) {
        index++;
        static_cast<TH1 *>(hList.FindObject("h_Nan"))->Fill(index * 1., edm::isNotFinite(it->covariance(i, j)) * 1.);
        if (edm::isNotFinite(it->covariance(i, j)))
          problem = true;
        // in addition, diagonal element must be positive
        if (j == i && it->covariance(i, j) < 0) {
          static_cast<TH1 *>(hList.FindObject("h_Nan"))->Fill(index * 1., 1.);
          problem = true;
        }
      }
    }
    if (problem)
      std::cout << " *** PROBLEM **" << std::endl;

    if (verbose_ > 0) {
      std::cout << "\tmomentum: " << tracks[idx].momentum() << "\tPT: " << tracks[idx].pt() << std::endl;
      std::cout << "\tvertex: " << tracks[idx].vertex() << "\tTIP: " << tracks[idx].d0() << " +- "
                << tracks[idx].d0Error() << "\tZ0: " << tracks[idx].dz() << " +- " << tracks[idx].dzError()
                << std::endl;
      std::cout << "\tcharge: " << tracks[idx].charge() << std::endl;
    }
  }

  //------------------------ simulated vertices and tracks

  edm::Handle<edm::SimVertexContainer> simVtcs;
  ev.getByToken(simVertexContainerToken_, simVtcs);

  //   std::cout << "SimVertex " << simVtcs->size() << std::endl;
  //   for(edm::SimVertexContainer::const_iterator v=simVtcs->begin();
  //       v!=simVtcs->end(); ++v){
  //     std::cout << "simvtx " << std::setw(10) << std::setprecision(3)
  //         << v->position().x() << " " << v->position().y() << " " <<
  //         v->position().z() << " "
  //         << v->parentIndex() << " " << v->noParent() << " " << std::endl; }

  edm::Handle<edm::SimTrackContainer> simTrks;
  ev.getByToken(simTrackContainerToken_, simTrks);
  std::cout << "simtrks " << simTrks->size() << std::endl;

  //-------------- association
  // matching cuts from Marcin
  float detaMax = 0.012;
  float dRMax = 0.025;
  typedef edm::SimTrackContainer::const_iterator IP;
  for (IP p = simTrks->begin(); p != simTrks->end(); p++) {
    if ((*p).noVertex())
      continue;
    if ((*p).type() == -99)
      continue;
    if ((*p).vertIndex() != 0)
      continue;

    math::XYZVector mom_gen((*p).momentum().x(), (*p).momentum().y(), (*p).momentum().z());
    float phi_gen = (*p).momentum().phi();
    //    float pt_gen = (*p).momentum().Pt();
    float pt_gen = sqrt((*p).momentum().x() * (*p).momentum().x() + (*p).momentum().y() * (*p).momentum().y());
    float eta_gen = (*p).momentum().eta();
    math::XYZTLorentzVectorD vtx((*simVtcs)[p->vertIndex()].position().x(),
                                 (*simVtcs)[p->vertIndex()].position().y(),
                                 (*simVtcs)[p->vertIndex()].position().z(),
                                 (*simVtcs)[p->vertIndex()].position().e());
    float z_gen = vtx.z();

    //     cout << "\tmomentum: " <<  (*p).momentum()
    //          <<" vtx: "<<(*p).vertIndex()<<" type: "<<(*p).type()
    //          << endl;

    typedef reco::TrackCollection::const_iterator IT;
    for (IT it = tracks.begin(); it != tracks.end(); it++) {
      math::XYZVector mom_rec = (*it).momentum();
      float phi_rec = (*it).momentum().phi();
      float pt_rec = (*it).pt();
      float z_rec = (*it).vertex().z();
      float eta_rec = (*it).momentum().eta();
      //    float chi2   = (*it).chi2();
      float dphi = phi_gen - phi_rec;
      while (dphi > M_PI)
        dphi -= 2 * M_PI;
      while (dphi < -M_PI)
        dphi += 2 * M_PI;
      float deta = eta_gen - eta_rec;
      float dz = z_gen - z_rec;
      double dR = deltaR(mom_gen, mom_rec);
      //
      // matched track
      //
      if (fabs(deta) < 0.3 && fabs(dphi) < 0.3)
        static_cast<TH1 *>(hList.FindObject("h_dR"))->Fill(dR);
      if (fabs(deta) < detaMax && dR < dRMax) {
        static_cast<TH1 *>(hList.FindObject("h_Pt"))->Fill((pt_gen - pt_rec) / pt_gen);
        static_cast<TH1 *>(hList.FindObject("h_TIP"))->Fill(it->d0());
        static_cast<TH1 *>(hList.FindObject("h_VtxZ"))->Fill(dz);
        static_cast<TH1 *>(hList.FindObject("h_VtxZ_Pull"))->Fill(fabs(dz / it->dzError()));
      }
    }
  }
}

void PixelTrackVal::endJob() {
  // Make my little tree
  TFile f(file_.c_str(), "RECREATE");
  hList.Write();
  f.Close();
}

// float PixelTrackVal::deltaRR(const  math::XYZVector & m1, const
// math::XYZVector & m2) const
//{
//  float dphi = m1.phi()-m2.phi();
//  while (dphi > 2*M_PI) dphi-=2*M_PI;
//  while (dphi < -2*M_PI) dphi+=2*M_PI;
//  float deta = m1.eta() - m2.eta();
//  float dr = sqrt( sqr(dphi) + sqr(deta));
//  return dr;
//}

DEFINE_FWK_MODULE(PixelTrackVal);

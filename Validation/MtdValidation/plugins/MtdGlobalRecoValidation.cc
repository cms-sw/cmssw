#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/ValidHandle.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

class MtdGlobalRecoValidation : public DQMEDAnalyzer {
public:
  explicit MtdGlobalRecoValidation(const edm::ParameterSet&);
  ~MtdGlobalRecoValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ------------ member data ------------

  const std::string folder_;
  const float trackMinEnergy_;
  const float trackMinEta_;
  const float trackMaxEta_;

  edm::EDGetTokenT<reco::TrackCollection> RecTrackToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> RecVertexToken_;

  MonitorElement* meBTLTrackRPTime_;
  MonitorElement* meBTLTrackEffEtaTot_;
  MonitorElement* meBTLTrackEffPhiTot_;
  MonitorElement* meBTLTrackEffPtTot_;
  MonitorElement* meBTLTrackEffEtaMtd_;
  MonitorElement* meBTLTrackEffPhiMtd_;
  MonitorElement* meBTLTrackEffPtMtd_;

  MonitorElement* meETLTrackRPTime_[4];
  MonitorElement* meETLTrackNumHits_[4];
  MonitorElement* meETLTrackEffEtaTot_[2];
  MonitorElement* meETLTrackEffPhiTot_[2];
  MonitorElement* meETLTrackEffPtTot_[2];
  MonitorElement* meETLTrackEffEtaMtd_[4];
  MonitorElement* meETLTrackEffPhiMtd_[4];
  MonitorElement* meETLTrackEffPtMtd_[4];

  MonitorElement* meTrackNumHits_;

  MonitorElement* meVerNumber_;
  MonitorElement* meVerZ_;
  MonitorElement* meVerTime_;
};

// ------------ constructor and destructor --------------
MtdGlobalRecoValidation::MtdGlobalRecoValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      trackMinEnergy_(iConfig.getParameter<double>("trackMinimumEnergy")),
      trackMinEta_(iConfig.getParameter<double>("trackMinimumEta")),
      trackMaxEta_(iConfig.getParameter<double>("trackMaximumEta")) {
  RecTrackToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTagT"));
  RecVertexToken_ = consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("inputTagV"));
}

MtdGlobalRecoValidation::~MtdGlobalRecoValidation() {}

// ------------ method called for each event  ------------
void MtdGlobalRecoValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace geant_units::operators;
  using namespace std;

  edm::ESHandle<MTDTopology> topologyHandle;
  iSetup.get<MTDTopologyRcd>().get(topologyHandle);
  const MTDTopology* topology = topologyHandle.product();

  bool topo1Dis = false;
  bool topo2Dis = false;
  if (topology->getMTDTopologyMode() <= static_cast<int>(MTDTopologyMode::Mode::barphiflat)) {
    topo1Dis = true;
  }
  if (topology->getMTDTopologyMode() > static_cast<int>(MTDTopologyMode::Mode::barphiflat)) {
    topo2Dis = true;
  }

  auto RecTrackHandle = makeValid(iEvent.getHandle(RecTrackToken_));
  auto RecVertexHandle = makeValid(iEvent.getHandle(RecVertexToken_));

  // --- Loop over all RECO tracks ---
  for (const auto& track : *RecTrackHandle) {
    if (track.pt() < trackMinEnergy_)
      continue;

    if (fabs(track.eta()) < trackMinEta_) {
      // --- all BTL tracks (with and without hit in MTD) ---
      meBTLTrackEffEtaTot_->Fill(track.eta());
      meBTLTrackEffPhiTot_->Fill(track.phi());
      meBTLTrackEffPtTot_->Fill(track.pt());

      bool MTDBtl = false;
      int numMTDBtlvalidhits = 0;
      for (const auto hit : track.recHits()) {
        if (hit->isValid() == false)
          continue;
        MTDDetId Hit = hit->geographicalId();
        if ((Hit.det() == 6) && (Hit.subdetId() == 1) && (Hit.mtdSubDetector() == 1)) {
          MTDBtl = true;
          numMTDBtlvalidhits++;
        }
      }
      meTrackNumHits_->Fill(numMTDBtlvalidhits);

      // --- keeping only tracks with last hit in MTD ---
      if (MTDBtl == true) {
        meBTLTrackEffEtaMtd_->Fill(track.eta());
        meBTLTrackEffPhiMtd_->Fill(track.phi());
        meBTLTrackEffPtMtd_->Fill(track.pt());
        meBTLTrackRPTime_->Fill(track.t0());
      }
    }

    else {
      // --- all ETL tracks (with and without hit in MTD) ---
      if ((track.eta() < -trackMinEta_) && (track.eta() > -trackMaxEta_)) {
        meETLTrackEffEtaTot_[0]->Fill(track.eta());
        meETLTrackEffPhiTot_[0]->Fill(track.phi());
        meETLTrackEffPtTot_[0]->Fill(track.pt());
      }

      if ((track.eta() > trackMinEta_) && (track.eta() < trackMaxEta_)) {
        meETLTrackEffEtaTot_[1]->Fill(track.eta());
        meETLTrackEffPhiTot_[1]->Fill(track.phi());
        meETLTrackEffPtTot_[1]->Fill(track.pt());
      }

      bool MTDEtlZnegD1 = false;
      bool MTDEtlZnegD2 = false;
      bool MTDEtlZposD1 = false;
      bool MTDEtlZposD2 = false;
      int numMTDEtlvalidhits = 0;
      for (const auto hit : track.recHits()) {
        if (hit->isValid() == false)
          continue;
        MTDDetId Hit = hit->geographicalId();
        if ((Hit.det() == 6) && (Hit.subdetId() == 1) && (Hit.mtdSubDetector() == 2)) {
          ETLDetId ETLHit = hit->geographicalId();

          if (topo2Dis) {
            if ((ETLHit.zside() == -1) && (ETLHit.nDisc() == 1)) {
              MTDEtlZnegD1 = true;
              numMTDEtlvalidhits++;
            }
            if ((ETLHit.zside() == -1) && (ETLHit.nDisc() == 2)) {
              MTDEtlZnegD2 = true;
              numMTDEtlvalidhits++;
            }
            if ((ETLHit.zside() == 1) && (ETLHit.nDisc() == 1)) {
              MTDEtlZposD1 = true;
              numMTDEtlvalidhits++;
            }
            if ((ETLHit.zside() == 1) && (ETLHit.nDisc() == 2)) {
              MTDEtlZposD2 = true;
              numMTDEtlvalidhits++;
            }
          }

          if (topo1Dis) {
            if (ETLHit.zside() == -1) {
              MTDEtlZnegD1 = true;
              numMTDEtlvalidhits++;
            }
            if (ETLHit.zside() == 1) {
              MTDEtlZposD1 = true;
              numMTDEtlvalidhits++;
            }
          }
        }
      }
      meTrackNumHits_->Fill(-numMTDEtlvalidhits);

      // --- keeping only tracks with last hit in MTD ---
      if ((track.eta() < -trackMinEta_) && (track.eta() > -trackMaxEta_)) {
        if (MTDEtlZnegD1 == true) {
          meETLTrackEffEtaMtd_[0]->Fill(track.eta());
          meETLTrackEffPhiMtd_[0]->Fill(track.phi());
          meETLTrackEffPtMtd_[0]->Fill(track.pt());
          meETLTrackRPTime_[0]->Fill(track.t0());
        }
        if (MTDEtlZnegD2 == true) {
          meETLTrackEffEtaMtd_[1]->Fill(track.eta());
          meETLTrackEffPhiMtd_[1]->Fill(track.phi());
          meETLTrackEffPtMtd_[1]->Fill(track.pt());
          meETLTrackRPTime_[1]->Fill(track.t0());
        }
      }
      if ((track.eta() > trackMinEta_) && (track.eta() < trackMaxEta_)) {
        if (MTDEtlZposD1 == true) {
          meETLTrackEffEtaMtd_[2]->Fill(track.eta());
          meETLTrackEffPhiMtd_[2]->Fill(track.phi());
          meETLTrackEffPtMtd_[2]->Fill(track.pt());
          meETLTrackRPTime_[2]->Fill(track.t0());
        }
        if (MTDEtlZposD2 == true) {
          meETLTrackEffEtaMtd_[3]->Fill(track.eta());
          meETLTrackEffPhiMtd_[3]->Fill(track.phi());
          meETLTrackEffPtMtd_[3]->Fill(track.pt());
          meETLTrackRPTime_[3]->Fill(track.t0());
        }
      }
    }
  }  //RECO tracks loop

  // --- Loop over the RECO vertices ---
  int nv = 0;
  for (const auto& v : *RecVertexHandle) {
    if (v.isValid()) {
      meVerZ_->Fill(v.z());
      meVerTime_->Fill(v.t());
      nv++;
    } else
      cout << "The vertex is not valid" << endl;
  }
  meVerNumber_->Fill(nv);
}

// ------------ method for histogram booking ------------
void MtdGlobalRecoValidation::bookHistograms(DQMStore::IBooker& ibook,
                                             edm::Run const& run,
                                             edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  // histogram booking
  meBTLTrackRPTime_ = ibook.book1D("TrackBTLRPTime", "Track t0 with respect to R.P.;t0 [ns]", 100, -1, 3);
  meBTLTrackEffEtaTot_ = ibook.book1D("TrackBTLEffEtaTot", "Track efficiency vs eta (Tot);#eta_{RECO}", 100, -1.6, 1.6);
  meBTLTrackEffPhiTot_ =
      ibook.book1D("TrackBTLEffPhiTot", "Track efficiency vs phi (Tot);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meBTLTrackEffPtTot_ = ibook.book1D("TrackBTLEffPtTot", "Track efficiency vs pt (Tot);pt_{RECO} [GeV]", 50, 0, 10);
  meBTLTrackEffEtaMtd_ = ibook.book1D("TrackBTLEffEtaMtd", "Track efficiency vs eta (Mtd);#eta_{RECO}", 100, -1.6, 1.6);
  meBTLTrackEffPhiMtd_ =
      ibook.book1D("TrackBTLEffPhiMtd", "Track efficiency vs phi (Mtd);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meBTLTrackEffPtMtd_ = ibook.book1D("TrackBTLEffPtMtd", "Track efficiency vs pt (Mtd);pt_{RECO} [GeV]", 50, 0, 10);
  meETLTrackRPTime_[0] =
      ibook.book1D("TrackETLRPTimeZnegD1", "Track t0 with respect to R.P. (-Z, Firstl Disk);t0 [ns]", 100, -1, 3);
  meETLTrackRPTime_[1] =
      ibook.book1D("TrackETLRPTimeZnegD2", "Track t0 with respect to R.P. (-Z, Second Disk);t0 [ns]", 100, -1, 3);
  meETLTrackRPTime_[2] =
      ibook.book1D("TrackETLRPTimeZposD1", "Track t0 with respect to R.P. (+Z, First Disk);t0 [ns]", 100, -1, 3);
  meETLTrackRPTime_[3] =
      ibook.book1D("TrackETLRPTimeZposD2", "Track t0 with respect to R.P. (+Z, Second Disk);t0 [ns]", 100, -1, 3);
  meETLTrackEffEtaTot_[0] =
      ibook.book1D("TrackETLEffEtaTotZneg", "Track efficiency vs eta (Tot) (-Z);#eta_{RECO}", 100, -3.2, -1.4);
  meETLTrackEffEtaTot_[1] =
      ibook.book1D("TrackETLEffEtaTotZpos", "Track efficiency vs eta (Tot) (+Z);#eta_{RECO}", 100, 1.4, 3.2);
  meETLTrackEffPhiTot_[0] =
      ibook.book1D("TrackETLEffPhiTotZneg", "Track efficiency vs phi (Tot) (-Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackEffPhiTot_[1] =
      ibook.book1D("TrackETLEffPhiTotZpos", "Track efficiency vs phi (Tot) (+Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackEffPtTot_[0] =
      ibook.book1D("TrackETLEffPtTotZneg", "Track efficiency vs pt (Tot) (-Z);pt_{RECO} [GeV]", 50, 0, 10);
  meETLTrackEffPtTot_[1] =
      ibook.book1D("TrackETLEffPtTotZpos", "Track efficiency vs pt (Tot) (+Z);pt_{RECO} [GeV]", 50, 0, 10);
  meETLTrackEffEtaMtd_[0] =
      ibook.book1D("TrackETLEffEtaMtdZnegD1",
                   "Track efficiency vs eta (Mtd) (-Z, Single(topo1D)/First(topo2D) Disk);#eta_{RECO}",
                   100,
                   -3.2,
                   -1.4);
  meETLTrackEffEtaMtd_[1] = ibook.book1D(
      "TrackETLEffEtaMtdZnegD2", "Track efficiency vs eta (Mtd) (-Z, Second Disk);#eta_{RECO}", 100, -3.2, -1.4);
  meETLTrackEffEtaMtd_[2] =
      ibook.book1D("TrackETLEffEtaMtdZposD1",
                   "Track efficiency vs eta (Mtd) (+Z, Single(topo1D)/First(topo2D) Disk);#eta_{RECO}",
                   100,
                   1.4,
                   3.2);
  meETLTrackEffEtaMtd_[3] = ibook.book1D(
      "TrackETLEffEtaMtdZposD2", "Track efficiency vs eta (Mtd) (+Z, Second Disk);#eta_{RECO}", 100, 1.4, 3.2);
  meETLTrackEffPhiMtd_[0] =
      ibook.book1D("TrackETLEffPhiMtdZnegD1",
                   "Track efficiency vs phi (Mtd) (-Z, Single(topo1D)/First(topo2D) Disk);#phi_{RECO} [rad]",
                   100,
                   -3.2,
                   3.2);
  meETLTrackEffPhiMtd_[1] = ibook.book1D(
      "TrackETLEffPhiMtdZnegD2", "Track efficiency vs phi (Mtd) (-Z, Second Disk);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackEffPhiMtd_[2] =
      ibook.book1D("TrackETLEffPhiMtdZposD1",
                   "Track efficiency vs phi (Mtd) (+Z, Single(topo1D)/First(topo2D) Disk);#phi_{RECO} [rad]",
                   100,
                   -3.2,
                   3.2);
  meETLTrackEffPhiMtd_[3] = ibook.book1D(
      "TrackETLEffPhiMtdZposD2", "Track efficiency vs phi (Mtd) (+Z, Second Disk);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackEffPtMtd_[0] =
      ibook.book1D("TrackETLEffPtMtdZnegD1",
                   "Track efficiency vs pt (Mtd) (-Z, Single(topo1D)/First(topo2D) Disk);pt_{RECO} [GeV]",
                   50,
                   0,
                   10);
  meETLTrackEffPtMtd_[1] = ibook.book1D(
      "TrackETLEffPtMtdZnegD2", "Track efficiency vs pt (Mtd) (-Z, Second Disk);pt_{RECO} [GeV]", 50, 0, 10);
  meETLTrackEffPtMtd_[2] =
      ibook.book1D("TrackETLEffPtMtdZposD1",
                   "Track efficiency vs pt (Mtd) (+Z, Single(topo1D)/First(topo2D) Disk);pt_{RECO} [GeV]",
                   50,
                   0,
                   10);
  meETLTrackEffPtMtd_[3] = ibook.book1D(
      "TrackETLEffPtMtdZposD2", "Track efficiency vs pt (Mtd) (+Z, Second Disk);pt_{RECO} [GeV]", 50, 0, 10);
  meTrackNumHits_ = ibook.book1D("TrackNumHits", "Number of valid MTD hits per track ; Number of hits", 10, -5, 5);
  meVerZ_ = ibook.book1D("VerZ", "RECO Vertex Z;Z_{RECO} [cm]", 180, -18, 18);
  meVerTime_ = ibook.book1D("VerTime", "RECO Vertex Time;t0 [ns]", 100, -1, 1);
  meVerNumber_ = ibook.book1D("VerNumber", "RECO Vertex Number: Number of vertices", 100, 0, 500);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------

void MtdGlobalRecoValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/GlobalReco");
  desc.add<edm::InputTag>("inputTagT", edm::InputTag("trackExtenderWithMTD", ""));
  desc.add<edm::InputTag>("inputTagV", edm::InputTag("offlinePrimaryVertices4D", ""));
  desc.add<double>("trackMinimumEnergy", 1.0);  // [GeV]
  desc.add<double>("trackMinimumEta", 1.5);
  desc.add<double>("trackMaximumEta", 3.2);

  descriptions.add("globalReco", desc);
}

DEFINE_FWK_MODULE(MtdGlobalRecoValidation);

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
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

class EtlRecoValidation : public DQMEDAnalyzer {
public:
  explicit EtlRecoValidation(const edm::ParameterSet&);
  ~EtlRecoValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ------------ member data ------------

  const std::string folder_;
  const float hitMinEnergy_;
  const float trackMinEnergy_;
  const float trackMinEta_;
  const float trackMaxEta_;

  edm::EDGetTokenT<FTLClusterCollection> etlRecCluToken_;
  edm::EDGetTokenT<reco::TrackCollection> etlRecTrackToken_;

  MonitorElement* meCluTime_[2];
  MonitorElement* meCluEnergy_[2];
  MonitorElement* meCluPhi_[2];
  MonitorElement* meCluEta_[2];
  MonitorElement* meCluHits_[2];
  MonitorElement* meCluZvsPhi_[2];
  MonitorElement* meTrackRPTime_[2];
  MonitorElement* meTrackEffEtaTot_[2];
  MonitorElement* meTrackEffPhiTot_[2];
  MonitorElement* meTrackEffPtTot_[2];
  MonitorElement* meTrackEffEtaMtd_[2];
  MonitorElement* meTrackEffPhiMtd_[2];
  MonitorElement* meTrackEffPtMtd_[2];
};

// ------------ constructor and destructor --------------
EtlRecoValidation::EtlRecoValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      hitMinEnergy_(iConfig.getParameter<double>("hitMinimumEnergy")),
      trackMinEnergy_(iConfig.getParameter<double>("trackMinimumEnergy")),
      trackMinEta_(iConfig.getParameter<double>("trackMinimumEta")),
      trackMaxEta_(iConfig.getParameter<double>("trackMaximumEta")) {
  etlRecCluToken_ = consumes<FTLClusterCollection>(iConfig.getParameter<edm::InputTag>("inputTagC"));
  etlRecTrackToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTagT"));
}

EtlRecoValidation::~EtlRecoValidation() {}

// ------------ method called for each event  ------------
void EtlRecoValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace geant_units::operators;
  using namespace std;

  edm::ESHandle<MTDGeometry> geometryHandle;
  iSetup.get<MTDDigiGeometryRecord>().get(geometryHandle);
  const MTDGeometry* geom = geometryHandle.product();

  edm::ESHandle<MTDTopology> topologyHandle;
  iSetup.get<MTDTopologyRcd>().get(topologyHandle);

  auto etlRecCluHandle = makeValid(iEvent.getHandle(etlRecCluToken_));
  auto etlRecTrackHandle = makeValid(iEvent.getHandle(etlRecTrackToken_));

  // --- Loop over the ETL RECO tracks ---
  for (const auto& track : *etlRecTrackHandle) {
    if (track.pt() < trackMinEnergy_)
      continue;

    // --- all ETL tracks (with and without hit in MTD) ---
    if ((track.eta() < -trackMinEta_) && (track.eta() > -trackMaxEta_)) {
      meTrackEffEtaTot_[0]->Fill(track.eta());
      meTrackEffPhiTot_[0]->Fill(track.phi());
      meTrackEffPtTot_[0]->Fill(track.pt());
    }

    if ((track.eta() > trackMinEta_) && (track.eta() < trackMaxEta_)) {
      meTrackEffEtaTot_[1]->Fill(track.eta());
      meTrackEffPhiTot_[1]->Fill(track.phi());
      meTrackEffPtTot_[1]->Fill(track.pt());
    }

    bool MTDEtlZneg = false;
    bool MTDEtlZpos = false;
    for (const auto hit : track.recHits()) {
      MTDDetId Hit = hit->geographicalId();
      if ((Hit.det() == 6) && (Hit.subdetId() == 1) && (Hit.mtdSubDetector() == 2) && (Hit.zside() == -1))
        MTDEtlZneg = true;
      if ((Hit.det() == 6) && (Hit.subdetId() == 1) && (Hit.mtdSubDetector() == 2) && (Hit.zside() == 1))
        MTDEtlZpos = true;
    }

    // --- keeping only tracks with last hit in MTD ---
    if ((track.eta() < -trackMinEta_) && (track.eta() > -trackMaxEta_)) {
      if (MTDEtlZneg == true) {
        meTrackEffEtaMtd_[0]->Fill(track.eta());
        meTrackEffPhiMtd_[0]->Fill(track.phi());
        meTrackEffPtMtd_[0]->Fill(track.pt());
        meTrackRPTime_[0]->Fill(track.t0());
      }
    }
    if ((track.eta() > trackMinEta_) && (track.eta() < trackMaxEta_)) {
      if (MTDEtlZpos == true) {
        meTrackEffEtaMtd_[1]->Fill(track.eta());
        meTrackEffPhiMtd_[1]->Fill(track.phi());
        meTrackEffPtMtd_[1]->Fill(track.pt());
        meTrackRPTime_[1]->Fill(track.t0());
      }
    }
  }

  // --- Loop over the ETL RECO clusters ---
  for (const auto& DetSetClu : *etlRecCluHandle) {
    for (const auto& cluster : DetSetClu) {
      if (cluster.energy() < hitMinEnergy_)
        continue;
      ETLDetId cluId = cluster.id();
      DetId detIdObject(cluId);
      const auto& genericDet = geom->idToDetUnit(detIdObject);
      if (genericDet == nullptr) {
        throw cms::Exception("EtlRecoValidation")
            << "GeographicalID: " << std::hex << cluId << " is invalid!" << std::dec << std::endl;
      }

      const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(genericDet->topology());
      const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

      Local3DPoint local_point(topo.localX(cluster.x()), topo.localY(cluster.y()), 0.);
      const auto& global_point = genericDet->toGlobal(local_point);

      int idet = (cluId.zside() + 1) / 2;

      meCluEnergy_[idet]->Fill(cluster.energy());
      meCluTime_[idet]->Fill(cluster.time());
      meCluPhi_[idet]->Fill(global_point.phi());
      meCluEta_[idet]->Fill(global_point.eta());
      meCluZvsPhi_[idet]->Fill(global_point.x(), global_point.y());
      meCluHits_[idet]->Fill(cluster.size());
    }
  }
}

// ------------ method for histogram booking ------------
void EtlRecoValidation::bookHistograms(DQMStore::IBooker& ibook, edm::Run const& run, edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  // histogram booking
  meCluTime_[0] = ibook.book1D("EtlCluTimeZneg", "ETL cluster ToA (-Z);ToA [ns]", 250, 0, 25);
  meCluTime_[1] = ibook.book1D("EtlCluTimeZpos", "ETL cluster ToA (+Z);ToA [ns]", 250, 0, 25);
  meCluEnergy_[0] = ibook.book1D("EtlCluEnergyZneg", "ETL cluster energy (-Z);E_{RECO} [MeV]", 100, 0, 10);
  meCluEnergy_[1] = ibook.book1D("EtlCluEnergyZpos", "ETL cluster energy (+Z);E_{RECO} [MeV]", 100, 0, 10);
  meCluPhi_[0] = ibook.book1D("EtlCluPhiZneg", "ETL cluster #phi (-Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meCluPhi_[1] = ibook.book1D("EtlCluPhiZpos", "ETL cluster #phi (+Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meCluEta_[0] = ibook.book1D("EtlCluEtaZneg", "ETL cluster #eta (-Z);#eta_{RECO}", 100, -3.2, -1.4);
  meCluEta_[1] = ibook.book1D("EtlCluEtaZpos", "ETL cluster #eta (+Z);#eta_{RECO}", 100, 1.4, 3.2);
  meCluHits_[0] = ibook.book1D("EtlCluHitNumberZneg", "ETL hits per cluster (-Z)", 10, 0, 10);
  meCluHits_[1] = ibook.book1D("EtlCluHitNumberZpos", "ETL hits per cluster (+Z)", 10, 0, 10);
  meCluZvsPhi_[0] = ibook.book2D(
      "EtlOccupancyZneg", "ETL cluster X vs Y (-Z);X_{RECO} [cm]; Y_{RECO} [cm]", 100, -150., 150., 100, -150, 150);
  meCluZvsPhi_[1] = ibook.book2D(
      "EtlOccupancyZpos", "ETL cluster X vs Y (+Z);X_{RECO} [cm]; Y_{RECO} [cm]", 100, -150., 150., 100, -150, 150);
  meTrackEffEtaTot_[0] =
      ibook.book1D("TrackEffEtaTotZneg", "Track efficiency vs eta (Tot) (-Z);#eta_{RECO}", 100, -3.2, -1.4);
  meTrackEffEtaTot_[1] =
      ibook.book1D("TrackEffEtaTotZpos", "Track efficiency vs eta (Tot) (+Z);#eta_{RECO}", 100, 1.4, 3.2);
  meTrackEffPhiTot_[0] =
      ibook.book1D("TrackEffPhiTotZneg", "Track efficiency vs phi (Tot) (-Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meTrackEffPhiTot_[1] =
      ibook.book1D("TrackEffPhiTotZpos", "Track efficiency vs phi (Tot) (+Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meTrackEffPtTot_[0] =
      ibook.book1D("TrackEffPtTotZneg", "Track efficiency vs pt (Tot) (-Z);pt_{RECO} [GeV]", 50, 0, 10);
  meTrackEffPtTot_[1] =
      ibook.book1D("TrackEffPtTotZpos", "Track efficiency vs pt (Tot) (+Z);pt_{RECO} [GeV]", 50, 0, 10);
  meTrackEffEtaMtd_[0] =
      ibook.book1D("TrackEffEtaMtdZneg", "Track efficiency vs eta (Mtd) (-Z);#eta_{RECO}", 100, -3.2, -1.4);
  meTrackEffEtaMtd_[1] =
      ibook.book1D("TrackEffEtaMtdZpos", "Track efficiency vs eta (Mtd) (+Z);#eta_{RECO}", 100, 1.4, 3.2);
  meTrackEffPhiMtd_[0] =
      ibook.book1D("TrackEffPhiMtdZneg", "Track efficiency vs phi (Mtd) (-Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meTrackEffPhiMtd_[1] =
      ibook.book1D("TrackEffPhiMtdZpos", "Track efficiency vs phi (Mtd) (+Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meTrackEffPtMtd_[0] =
      ibook.book1D("TrackEffPtMtdZneg", "Track efficiency vs pt (Mtd) (-Z);pt_{RECO} [GeV]", 50, 0, 10);
  meTrackEffPtMtd_[1] =
      ibook.book1D("TrackEffPtMtdZpos", "Track efficiency vs pt (Mtd) (+Z);pt_{RECO} [GeV]", 50, 0, 10);
  meTrackRPTime_[0] = ibook.book1D("TrackRPTimeZneg", "Track t0 with respect to R.P. (-Z);t0 [ns]", 100, -10, 10);
  meTrackRPTime_[1] = ibook.book1D("TrackRPTimeZpos", "Track t0 with respect to R.P. (+Z);t0 [ns]", 100, -10, 10);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------

void EtlRecoValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/ETL/Reco");
  desc.add<edm::InputTag>("inputTagC", edm::InputTag("mtdClusters", "FTLEndcap"));
  desc.add<edm::InputTag>("inputTagT", edm::InputTag("trackExtenderWithMTD", ""));
  desc.add<double>("hitMinimumEnergy", 1.);     // [MeV]
  desc.add<double>("trackMinimumEnergy", 0.5);  // [GeV]
  desc.add<double>("trackMinimumEta", 1.4);
  desc.add<double>("trackMaximumEta", 3.2);

  descriptions.add("etlReco", desc);
}

DEFINE_FWK_MODULE(EtlRecoValidation);

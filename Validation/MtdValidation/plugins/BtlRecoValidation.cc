#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/ValidHandle.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
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
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

class BtlRecoValidation : public DQMEDAnalyzer {
public:
  explicit BtlRecoValidation(const edm::ParameterSet&);
  ~BtlRecoValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ------------ member data ------------

  const std::string folder_;
  const float hitMinEnergy_;
  const float trackMinEnergy_;
  const float trackMinEta_;

  edm::EDGetTokenT<FTLClusterCollection> btlRecCluToken_;
  edm::EDGetTokenT<reco::TrackCollection> btlRecTrackToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> RecVertexToken_;

  MonitorElement* meCluTime_;
  MonitorElement* meCluEnergy_;
  MonitorElement* meCluPhi_;
  MonitorElement* meCluEta_;
  MonitorElement* meCluHits_;
  MonitorElement* meCluZvsPhi_;
  MonitorElement* meTrackRPTime_;
  MonitorElement* meTrackEffEtaTot_;
  MonitorElement* meTrackEffPhiTot_;
  MonitorElement* meTrackEffPtTot_;
  MonitorElement* meTrackEffEtaMtd_;
  MonitorElement* meTrackEffPhiMtd_;
  MonitorElement* meTrackEffPtMtd_;
  MonitorElement* meVerNumber_;
  MonitorElement* meVerZ_;
  MonitorElement* meVerTime_;
};

// ------------ constructor and destructor --------------
BtlRecoValidation::BtlRecoValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      hitMinEnergy_(iConfig.getParameter<double>("hitMinimumEnergy")),
      trackMinEnergy_(iConfig.getParameter<double>("trackMinimumEnergy")),
      trackMinEta_(iConfig.getParameter<double>("trackMinimumEta")) {
  btlRecCluToken_ = consumes<FTLClusterCollection>(iConfig.getParameter<edm::InputTag>("inputTagC"));
  btlRecTrackToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTagT"));
  RecVertexToken_ = consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("inputTagV"));
}

BtlRecoValidation::~BtlRecoValidation() {}

// ------------ method called for each event  ------------
void BtlRecoValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace geant_units::operators;
  using namespace std;

  edm::ESHandle<MTDGeometry> geometryHandle;
  iSetup.get<MTDDigiGeometryRecord>().get(geometryHandle);
  const MTDGeometry* geom = geometryHandle.product();

  edm::ESHandle<MTDTopology> topologyHandle;
  iSetup.get<MTDTopologyRcd>().get(topologyHandle);

  auto btlRecCluHandle = makeValid(iEvent.getHandle(btlRecCluToken_));
  auto btlRecTrackHandle = makeValid(iEvent.getHandle(btlRecTrackToken_));
  auto RecVertexHandle = makeValid(iEvent.getHandle(RecVertexToken_));

  // --- Loop over the BTL RECO tracks ---
  for (const auto& track : *btlRecTrackHandle) {
    if (fabs(track.eta()) > trackMinEta_)
      continue;
    if (track.pt() < trackMinEnergy_)
      continue;

    // --- all BTL tracks (with and without hit in MTD) ---
    meTrackEffEtaTot_->Fill(track.eta());
    meTrackEffPhiTot_->Fill(track.phi());
    meTrackEffPtTot_->Fill(track.pt());

    bool MTDBtl = false;
    for (const auto hit : track.recHits()) {
      MTDDetId Hit = hit->geographicalId();
      if ((Hit.det() == 6) && (Hit.subdetId() == 1) && (Hit.mtdSubDetector() == 1))
        MTDBtl = true;
    }

    // --- keeping only tracks with last hit in MTD ---
    if (MTDBtl == true) {
      meTrackEffEtaMtd_->Fill(track.eta());
      meTrackEffPhiMtd_->Fill(track.phi());
      meTrackEffPtMtd_->Fill(track.pt());
      meTrackRPTime_->Fill(track.t0());
    }
  }

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

  // --- Loop over the BTL RECO clusters ---
  for (const auto& DetSetClu : *btlRecCluHandle) {
    for (const auto& cluster : DetSetClu) {
      if (cluster.energy() < hitMinEnergy_)
        continue;
      BTLDetId cluId = cluster.id();
      DetId detIdObject(cluId);
      const auto& genericDet = geom->idToDetUnit(detIdObject);
      if (genericDet == nullptr) {
        throw cms::Exception("BtlRecoValidation")
            << "GeographicalID: " << std::hex << cluId << " is invalid!" << std::dec << std::endl;
      }

      const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(genericDet->topology());
      const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

      Local3DPoint local_point(cluster.x() * 5.7, cluster.y() * 0.3, 0.);
      local_point = topo.pixelToModuleLocalPoint(local_point, cluId.row(topo.nrows()), cluId.column(topo.ncolumns()));
      const auto& global_point = genericDet->toGlobal(local_point);

      meCluEnergy_->Fill(cluster.energy());
      meCluTime_->Fill(cluster.time());
      meCluPhi_->Fill(global_point.phi());
      meCluEta_->Fill(global_point.eta());
      meCluZvsPhi_->Fill(global_point.z(), global_point.phi());
      meCluHits_->Fill(cluster.size());
    }
  }
}

// ------------ method for histogram booking ------------
void BtlRecoValidation::bookHistograms(DQMStore::IBooker& ibook, edm::Run const& run, edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  // histogram booking
  meCluTime_ = ibook.book1D("BtlCluTime", "BTL cluster time ToA;ToA [ns]", 250, 0, 25);
  meCluEnergy_ = ibook.book1D("BtlCluEnergy", "BTL cluster energy;E_{RECO} [MeV]", 100, 0, 20);
  meCluPhi_ = ibook.book1D("BtlCluPhi", "BTL cluster #phi;#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meCluEta_ = ibook.book1D("BtlCluEta", "BTL cluster #eta;#eta_{RECO}", 100, -1.6, 1.6);
  meCluHits_ = ibook.book1D("BtlCluHitNumber", "BTL hits per cluster", 10, 0, 10);
  meCluZvsPhi_ = ibook.book2D(
      "BtlOccupancy", "BTL cluster Z vs #phi;Z_{RECO} [cm]; #phi_{RECO} [rad]", 100, -260., 260., 100, -3.2, 3.2);
  meTrackEffEtaTot_ = ibook.book1D("TrackEffEtaTot", "Track efficiency vs eta D;#eta_{RECO}", 100, -1.6, 1.6);
  meTrackEffPhiTot_ = ibook.book1D("TrackEffPhiTot", "Track efficiency vs phi D;#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meTrackEffPtTot_ = ibook.book1D("TrackEffPtTot", "Track efficiency vs pt D;pt_{RECO} [GeV]", 50, 0, 10);
  meTrackEffEtaMtd_ = ibook.book1D("TrackEffEtaMtd", "Track efficiency vs eta N;#eta_{RECO}", 100, -1.6, 1.6);
  meTrackEffPhiMtd_ = ibook.book1D("TrackEffPhiMtd", "Track efficiency vs phi N;#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meTrackEffPtMtd_ = ibook.book1D("TrackEffPtMtd", "Track efficiency vs pt N;pt_{RECO} [GeV]", 50, 0, 10);
  meTrackRPTime_ = ibook.book1D("TrackRPTime", "Track t0 with respect to R.P.;t0 [ns]", 100, -10, 10);
  meVerZ_ = ibook.book1D("VerZ", "RECO Vertex Z;Z_{RECO} [cm]", 180, -18, 18);
  meVerTime_ = ibook.book1D("VerTime", "RECO Vertex Time;t0 [ns]", 100, -1, 1);
  meVerNumber_ = ibook.book1D("VerNumber", "RECO Vertex Number", 100, 0, 500);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------

void BtlRecoValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/BTL/Reco");
  desc.add<edm::InputTag>("inputTagC", edm::InputTag("mtdClusters", "FTLBarrel"));
  desc.add<edm::InputTag>("inputTagT", edm::InputTag("trackExtenderWithMTD", ""));
  desc.add<edm::InputTag>("inputTagV", edm::InputTag("offlinePrimaryVertices4D", ""));
  desc.add<double>("hitMinimumEnergy", 1.);    // [MeV]
  desc.add<double>("trackMinimumEnergy", 1.);  // [GeV]
  desc.add<double>("trackMinimumEta", 1.5);

  descriptions.add("btlReco", desc);
}

DEFINE_FWK_MODULE(BtlRecoValidation);

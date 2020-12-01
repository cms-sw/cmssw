// -*- C++ -*-
///bookLayer
// Package:    Phase2ITValidateCluster
// Class:      Phase2ITValidateCluster
//
/**\class Phase2ITValidateCluster Phase2ITValidateCluster.cc 

 Description: Validation plots tracker clusters. 

*/
//
// Author: Gabriel Ramirez, Suvankar Roy Chowdhury
// Date: May 23, 2020
//
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"

#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

class Phase2ITValidateCluster : public DQMEDAnalyzer {
public:
  typedef std::map<unsigned int, std::vector<PSimHit>> SimHitsMap;
  typedef std::map<unsigned int, SimTrack> SimTracksMap;

  explicit Phase2ITValidateCluster(const edm::ParameterSet&);
  ~Phase2ITValidateCluster() override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  struct ClusterMEs {
    MonitorElement* deltaX_P = nullptr;
    MonitorElement* deltaY_P = nullptr;
    MonitorElement* deltaX_P_primary = nullptr;
    MonitorElement* deltaY_P_primary = nullptr;
  };

  void fillITHistos(const edm::Event& iEvent,
                    const std::vector<edm::Handle<edm::PSimHitContainer>>& simHits,
                    const std::map<unsigned int, SimTrack>& simTracks);
  void bookLayerHistos(DQMStore::IBooker& ibooker, uint32_t det_it, const std::string& subdir);
  std::vector<unsigned int> getSimTrackId(const edm::Handle<edm::DetSetVector<PixelDigiSimLink>>& pixelSimLinks,
                                          const DetId& detId,
                                          unsigned int channel);

  std::map<std::string, ClusterMEs> layerMEs_;

  edm::ParameterSet config_;
  double simtrackminpt_;
  std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> simHitTokens_;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink>> simITLinksToken_;
  edm::EDGetTokenT<edm::SimTrackContainer> simTracksToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> clustersToken_;
  std::vector<edm::InputTag> pSimHitSrc_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const TrackerGeometry* tkGeom_ = nullptr;
  const TrackerTopology* tTopo_ = nullptr;
};
#include "Validation/SiTrackerPhase2V/interface/TrackerPhase2ValidationUtil.h"
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"

//
// constructors
//

Phase2ITValidateCluster::Phase2ITValidateCluster(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      simtrackminpt_(config_.getParameter<double>("SimTrackMinPt")),
      simITLinksToken_(consumes<edm::DetSetVector<PixelDigiSimLink>>(
          config_.getParameter<edm::InputTag>("InnerTrackerDigiSimLinkSource"))),
      simTracksToken_(consumes<edm::SimTrackContainer>(config_.getParameter<edm::InputTag>("simtracks"))),
      clustersToken_(
          consumes<edmNew::DetSetVector<SiPixelCluster>>(config_.getParameter<edm::InputTag>("ClusterSource"))),
      pSimHitSrc_(config_.getParameter<std::vector<edm::InputTag>>("PSimHitSource")),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()) {
  edm::LogInfo("Phase2ITValidateCluster") << ">>> Construct Phase2ITValidateCluster ";
  for (const auto& itag : pSimHitSrc_)
    simHitTokens_.push_back(consumes<edm::PSimHitContainer>(itag));
}

Phase2ITValidateCluster::~Phase2ITValidateCluster() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  edm::LogInfo("Phase2ITValidateCluster") << ">>> Destroy Phase2ITValidateCluster ";
}
//
// -- DQM Begin Run
//
void Phase2ITValidateCluster::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  edm::ESHandle<TrackerGeometry> geomHandle = iSetup.getHandle(geomToken_);
  tkGeom_ = &(*geomHandle);
  edm::ESHandle<TrackerTopology> tTopoHandle = iSetup.getHandle(topoToken_);
  tTopo_ = tTopoHandle.product();
}

// -- Analyze
//
void Phase2ITValidateCluster::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Getting simHits
  std::vector<edm::Handle<edm::PSimHitContainer>> simHits;
  for (const auto& itoken : simHitTokens_) {
    edm::Handle<edm::PSimHitContainer> simHitHandle;
    iEvent.getByToken(itoken, simHitHandle);
    if (!simHitHandle.isValid())
      continue;
    simHits.emplace_back(simHitHandle);
  }
  // Get the SimTracks
  edm::Handle<edm::SimTrackContainer> simTracksRaw;
  iEvent.getByToken(simTracksToken_, simTracksRaw);

  // Rearrange the simTracks for ease of use <simTrackID, simTrack>
  SimTracksMap simTracks;
  for (edm::SimTrackContainer::const_iterator simTrackIt(simTracksRaw->begin()); simTrackIt != simTracksRaw->end();
       ++simTrackIt) {
    if (simTrackIt->momentum().pt() > simtrackminpt_) {
      simTracks.emplace(simTrackIt->trackId(), *simTrackIt);
    }
  }
  fillITHistos(iEvent, simHits, simTracks);
}

void Phase2ITValidateCluster::fillITHistos(const edm::Event& iEvent,
                                           const std::vector<edm::Handle<edm::PSimHitContainer>>& simHits,
                                           const std::map<unsigned int, SimTrack>& simTracks) {
  // Getting the clusters
  edm::Handle<edmNew::DetSetVector<SiPixelCluster>> clusterHandle;
  iEvent.getByToken(clustersToken_, clusterHandle);

  // Getting PixelDigiSimLinks
  edm::Handle<edm::DetSetVector<PixelDigiSimLink>> pixelSimLinksHandle;
  iEvent.getByToken(simITLinksToken_, pixelSimLinksHandle);

  for (const auto& DSVItr : *clusterHandle) {
    // Getting the id of detector unit
    uint32_t rawid = DSVItr.detId();
    DetId detId(rawid);
    const GeomDetUnit* geomDetUnit(tkGeom_->idToDetUnit(detId));
    if (!geomDetUnit)
      continue;

    std::string folderkey = phase2tkutil::getITHistoId(detId, tTopo_);
    for (const auto& clusterItr : DSVItr) {
      MeasurementPoint mpCluster(clusterItr.x(), clusterItr.y());
      Local3DPoint localPosCluster = geomDetUnit->topology().localPosition(mpCluster);
      // Get simTracks from the cluster
      std::vector<unsigned int> clusterSimTrackIds;
      for (int irow = clusterItr.minPixelRow(); irow <= clusterItr.maxPixelRow(); ++irow) {
        for (int icol = clusterItr.minPixelCol(); icol <= clusterItr.maxPixelCol(); ++icol) {
          uint32_t channel = PixelChannelIdentifier::pixelToChannel(irow, icol);
          std::vector<unsigned int> simTrackIds(getSimTrackId(pixelSimLinksHandle, detId, channel));
          for (auto it : simTrackIds) {
            bool add = true;
            for (unsigned int j = 0; j < clusterSimTrackIds.size(); ++j) {
              // only save simtrackids that are not present yet
              if (it == clusterSimTrackIds.at(j))
                add = false;
            }
            if (add)
              clusterSimTrackIds.push_back(it);
          }
        }
      }
      std::sort(clusterSimTrackIds.begin(), clusterSimTrackIds.end());
      const PSimHit* closestSimHit = nullptr;
      float minx = 10000.;
      // Get the SimHit
      for (const auto& psimhitCont : simHits) {
        for (const auto& simhitIt : *psimhitCont) {
          if (rawid == simhitIt.detUnitId()) {
            auto it = std::lower_bound(clusterSimTrackIds.begin(), clusterSimTrackIds.end(), simhitIt.trackId());
            if (it != clusterSimTrackIds.end() && *it == simhitIt.trackId()) {
              if (!closestSimHit || fabs(simhitIt.localPosition().x() - localPosCluster.x()) < minx) {
                minx = abs(simhitIt.localPosition().x() - localPosCluster.x());
                closestSimHit = &simhitIt;
              }
            }
          }
        }  //end loop over PSimhitcontainers
      }    //end loop over simHits

      if (!closestSimHit)
        continue;
      // only look at simhits from highpT tracks
      auto simTrackIt(simTracks.find(closestSimHit->trackId()));
      if (simTrackIt == simTracks.end())
        continue;
      Local3DPoint localPosSimHit(closestSimHit->localPosition());
      const double deltaX = localPosCluster.x() - localPosSimHit.x();
      const double deltaY = localPosCluster.y() - localPosSimHit.y();

      auto layerMEIt = layerMEs_.find(folderkey);
      if (layerMEIt == layerMEs_.end())
        continue;

      ClusterMEs& local_mes = layerMEIt->second;
      local_mes.deltaX_P->Fill(deltaX);
      local_mes.deltaY_P->Fill(deltaY);
      // Primary particles only
      if (phase2tkutil::isPrimary(simTrackIt->second, closestSimHit)) {
        local_mes.deltaX_P_primary->Fill(deltaX);
        local_mes.deltaY_P_primary->Fill(deltaY);
      }
    }
  }
}

//
// -- Book Histograms
//
void Phase2ITValidateCluster::bookHistograms(DQMStore::IBooker& ibooker,
                                             edm::Run const& iRun,
                                             edm::EventSetup const& iSetup) {
  std::string top_folder = config_.getParameter<std::string>("TopFolderName");
  edm::LogInfo("Phase2ITValidateCluster") << " Booking Histograms in: " << top_folder;

  edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;
  if (theTkDigiGeomWatcher.check(iSetup)) {
    for (auto const& det_u : tkGeom_->detUnits()) {
      //Always check TrackerNumberingBuilder before changing this part
      if (!(det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXB ||
            det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXEC))
        continue;  //continue if not Pixel
      uint32_t detId_raw = det_u->geographicalId().rawId();
      bookLayerHistos(ibooker, detId_raw, top_folder);
    }
  }
}

//////////////////Layer Histo/////////////////////////////////
void Phase2ITValidateCluster::bookLayerHistos(DQMStore::IBooker& ibooker, uint32_t det_id, const std::string& subdir) {
  std::string folderName = phase2tkutil::getITHistoId(det_id, tTopo_);
  if (folderName.empty()) {
    edm::LogWarning("Phase2ITValidateCluster") << ">>>> Invalid histo_id ";
    return;
  }

  if (layerMEs_.find(folderName) == layerMEs_.end()) {
    ibooker.cd();
    ibooker.setCurrentFolder(subdir + '/' + folderName);
    edm::LogInfo("Phase2TrackerValidateDigi") << " Booking Histograms in: " << subdir + '/' + folderName;
    ClusterMEs local_mes;

    local_mes.deltaX_P =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_X_Pixel"), ibooker);

    local_mes.deltaY_P =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_Y_Pixel"), ibooker);

    // Puting primary digis in a subfolder
    ibooker.setCurrentFolder(subdir + '/' + folderName + "/PrimarySimHits");

    local_mes.deltaX_P_primary =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_X_Pixel_Primary"), ibooker);

    local_mes.deltaY_P_primary =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("Delta_Y_Pixel_Primary"), ibooker);
    layerMEs_.emplace(folderName, local_mes);
  }
}

std::vector<unsigned int> Phase2ITValidateCluster::getSimTrackId(
    const edm::Handle<edm::DetSetVector<PixelDigiSimLink>>& pixelSimLinks, const DetId& detId, unsigned int channel) {
  std::vector<unsigned int> retvec;
  edm::DetSetVector<PixelDigiSimLink>::const_iterator DSViter(pixelSimLinks->find(detId));
  if (DSViter == pixelSimLinks->end())
    return retvec;
  for (edm::DetSet<PixelDigiSimLink>::const_iterator it = DSViter->data.begin(); it != DSViter->data.end(); ++it) {
    if (channel == it->channel()) {
      retvec.push_back(it->SimTrackId());
    }
  }
  return retvec;
}

void Phase2ITValidateCluster::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  //for macro-pixel sensors
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_X_Pixel");
    psd0.add<std::string>("title", "#Delta X;Cluster resolution X dimension");
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 5.0);
    psd0.add<double>("xmin", -5.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Delta_X_Pixel", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_Y_Pixel");
    psd0.add<std::string>("title", "#Delta Y ;Cluster resolution Y dimension");
    psd0.add<double>("xmin", -5.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 5.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Delta_Y_Pixel", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_X_Pixel_Primary");
    psd0.add<std::string>("title", "#Delta X ;cluster resolution X dimension");
    psd0.add<double>("xmin", -5.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 5.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Delta_X_Pixel_Primary", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Delta_Y_Pixel_Primary");
    psd0.add<std::string>("title", "#Delta Y ;cluster resolution Y dimension");
    psd0.add<double>("xmin", -5.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 5.0);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("Delta_Y_Pixel_Primary", psd0);
  }

  desc.add<std::string>("TopFolderName", "TrackerPhase2ITClusterV");
  desc.add<edm::InputTag>("ClusterSource", edm::InputTag("siPixelClusters"));
  desc.add<edm::InputTag>("InnerTrackerDigiSimLinkSource", edm::InputTag("simSiPixelDigis", "Pixel"));
  desc.add<edm::InputTag>("simtracks", edm::InputTag("g4SimHits"));
  desc.add<double>("SimTrackMinPt", 0.0);
  desc.add<std::vector<edm::InputTag>>("PSimHitSource",
                                       {
                                           edm::InputTag("g4SimHits:TrackerHitsPixelBarrelLowTof"),
                                           edm::InputTag("g4SimHits:TrackerHitsPixelBarrelHighTof"),
                                           edm::InputTag("g4SimHits:TrackerHitsPixelEndcapLowTof"),
                                           edm::InputTag("g4SimHits:TrackerHitsPixelEndcapHighTof"),
                                       });
  descriptions.add("Phase2ITValidateCluster", desc);
}
DEFINE_FWK_MODULE(Phase2ITValidateCluster);


// -*- C++ -*-
//
// Package:   Validation/SiTrackerPhase2V
// Class:     Phase2OTValidateStub.cc
/**
 * This class is part of the Phase 2 Tracker validation framework. It validates
 * the performance of stub reconstruction by comparing them with tracking
 * particles. It generates histograms to assess stub efficiency and residual
 * performance.
 *
 * Usage:
 * To generate histograms from this code, run the test configuration files
 * provided in the DQM/SiTrackerPhase2/test directory. The generated histograms
 * can then be analyzed or visualized.
 */
//
// Original Author:
//
// Updated by: Brandi Skipworth, 2025

// system include files
#include <memory>
#include <numeric>
#include <vector>

// user include files
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "SimDataFormats/Associations/interface/TTStubAssociationMap.h"
#include "Validation/SiTrackerPhase2V/interface/TrackerPhase2ValidationUtil.h"

class Phase2OTValidateStub : public DQMEDAnalyzer {
public:
  explicit Phase2OTValidateStub(const edm::ParameterSet&);
  ~Phase2OTValidateStub() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  float phiOverBendCorrection(bool isBarrel,
                              float stub_z,
                              float stub_r,
                              const TrackerTopology* tTopo,
                              uint32_t detid,
                              const GeomDetUnit* det0,
                              const GeomDetUnit* det1);
  std::vector<double> getTPDerivedCoords(edm::Ptr<TrackingParticle> associatedTP,
                                         bool isBarrel,
                                         float stub_z,
                                         float stub_r) const;
  // TTStub stacks
  // Global position of the stubs
  MonitorElement* Stub_RZ = nullptr;  // TTStub #rho vs. z

  // delta_z hists
  MonitorElement* z_res_isPS_barrel = nullptr;
  MonitorElement* z_res_is2S_barrel = nullptr;

  // delta_r hists
  MonitorElement* r_res_isPS_fw_endcap = nullptr;
  MonitorElement* r_res_is2S_fw_endcap = nullptr;
  MonitorElement* r_res_isPS_bw_endcap = nullptr;
  MonitorElement* r_res_is2S_bw_endcap = nullptr;

  // delta_phi hists
  MonitorElement* phi_res_isPS_barrel = nullptr;
  MonitorElement* phi_res_is2S_barrel = nullptr;
  MonitorElement* phi_res_fw_endcap = nullptr;
  MonitorElement* phi_res_bw_endcap = nullptr;
  std::vector<MonitorElement*> phi_res_barrel_layers;
  std::vector<MonitorElement*> phi_res_fw_endcap_discs;
  std::vector<MonitorElement*> phi_res_bw_endcap_discs;

  // delta_bend hists
  MonitorElement* bend_res_fw_endcap = nullptr;
  MonitorElement* bend_res_bw_endcap = nullptr;
  MonitorElement* bend_res_barrel = nullptr;
  std::vector<MonitorElement*> bend_res_barrel_layers;
  std::vector<MonitorElement*> bend_res_fw_endcap_discs;
  std::vector<MonitorElement*> bend_res_bw_endcap_discs;

  std::vector<MonitorElement*>* phi_res_vec = nullptr;
  std::vector<MonitorElement*>* bend_res_vec = nullptr;

  // stub efficiency plots
  MonitorElement* gen_clusters_barrel = nullptr;                // denominator
  MonitorElement* gen_clusters_zoom_barrel = nullptr;           // denominator
  MonitorElement* gen_clusters_endcaps = nullptr;               // denominator
  MonitorElement* gen_clusters_zoom_endcaps = nullptr;          // denominator
  MonitorElement* gen_clusters_if_stub_barrel = nullptr;        // numerator
  MonitorElement* gen_clusters_if_stub_zoom_barrel = nullptr;   // numerator
  MonitorElement* gen_clusters_if_stub_endcaps = nullptr;       // numerator
  MonitorElement* gen_clusters_if_stub_zoom_endcaps = nullptr;  // numerator

private:
  edm::ParameterSet conf_;
  edm::EDGetTokenT<std::vector<TrackingParticle>> trackingParticleToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> tagTTStubsToken_;
  edm::EDGetTokenT<TTStubAssociationMap<Ref_Phase2TrackerDigi_>>
      ttStubMCTruthToken_;  // MC truth association map for stubs
  edm::EDGetTokenT<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>>
      ttClusterMCTruthToken_;  // MC truth association map for clusters
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> getTokenTrackerGeom_;
  std::string topFolderName_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const TrackerGeometry* tkGeom_ = nullptr;
  const TrackerTopology* tTopo_ = nullptr;
  int TP_minNStub;
  int TP_minNLayersStub;
  double TP_minPt;
  double TP_maxEta;
  double TP_maxVtxZ;
  double TP_maxD0;
  double TP_maxLxy;
};

// constructors and destructor
Phase2OTValidateStub::Phase2OTValidateStub(const edm::ParameterSet& iConfig)
    : conf_(iConfig),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()) {
  // now do what ever initialization is needed
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  trackingParticleToken_ =
      consumes<std::vector<TrackingParticle>>(conf_.getParameter<edm::InputTag>("trackingParticleToken"));
  tagTTStubsToken_ =
      consumes<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>(conf_.getParameter<edm::InputTag>("TTStubs"));
  ttStubMCTruthToken_ =
      consumes<TTStubAssociationMap<Ref_Phase2TrackerDigi_>>(conf_.getParameter<edm::InputTag>("MCTruthStubInputTag"));
  ttClusterMCTruthToken_ = consumes<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>>(
      conf_.getParameter<edm::InputTag>("MCTruthClusterInputTag"));
  TP_minNStub = conf_.getParameter<int>("TP_minNStub");
  TP_minNLayersStub = conf_.getParameter<int>("TP_minNLayersStub");
  TP_minPt = conf_.getParameter<double>("TP_minPt");
  TP_maxEta = conf_.getParameter<double>("TP_maxEta");
  TP_maxVtxZ = conf_.getParameter<double>("TP_maxVtxZ");
  TP_maxD0 = conf_.getParameter<double>("TP_maxD0");
  TP_maxLxy = conf_.getParameter<double>("TP_maxLxy");
}

Phase2OTValidateStub::~Phase2OTValidateStub() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void Phase2OTValidateStub::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  tkGeom_ = &(iSetup.getData(geomToken_));
  tTopo_ = &(iSetup.getData(topoToken_));

  // Clear existing histograms
  phi_res_barrel_layers.clear();
  bend_res_barrel_layers.clear();
  phi_res_fw_endcap_discs.clear();
  bend_res_fw_endcap_discs.clear();
  phi_res_bw_endcap_discs.clear();
  bend_res_bw_endcap_discs.clear();

  // Resize vectors and set elements to nullptr
  phi_res_barrel_layers.resize(trklet::N_LAYER, nullptr);
  bend_res_barrel_layers.resize(trklet::N_LAYER, nullptr);
  phi_res_fw_endcap_discs.resize(trklet::N_DISK, nullptr);
  bend_res_fw_endcap_discs.resize(trklet::N_DISK, nullptr);
  phi_res_bw_endcap_discs.resize(trklet::N_DISK, nullptr);
  bend_res_bw_endcap_discs.resize(trklet::N_DISK, nullptr);
}
// member functions

// Calculate the correction factor for tilted modules in barrel
float Phase2OTValidateStub::phiOverBendCorrection(bool isBarrel,
                                                  float stub_z,
                                                  float stub_r,
                                                  const TrackerTopology* tTopo,
                                                  uint32_t detid,
                                                  const GeomDetUnit* det0,
                                                  const GeomDetUnit* det1) {
  // Get R0, R1, Z0, Z1 values
  float R0 = det0->position().perp();
  float R1 = det1->position().perp();
  float Z0 = det0->position().z();
  float Z1 = det1->position().z();

  bool isTiltedBarrel = (isBarrel && tTopo->tobSide(detid) != 3);

  float tiltAngle = 0;  // Initialize to 0 (meaning no tilt, in the endcaps)
  if (isTiltedBarrel) {
    float deltaR = std::abs(R1 - R0);
    float deltaZ = (R1 - R0 > 0) ? (Z1 - Z0) : -(Z1 - Z0);  // if module parallel, tilt angle should
                                                            // be π/2 and deltaZ would approach zero
    // fill histograms here
    tiltAngle = atan(deltaR / std::abs(deltaZ));
  }

  float correction;
  if (isBarrel && tTopo->tobSide(detid) != 3) {  // Assuming this condition represents tiltedBarrel
    correction = cos(tiltAngle) * std::abs(stub_z) / stub_r + sin(tiltAngle);
  } else if (isBarrel) {
    correction = 1;
  } else {
    correction = std::abs(stub_z) / stub_r;  // if tiltAngle = 0, stub (not module) is parallel to the beam
                                             // line, if tiltAngle = 90, stub is perpendicular to beamline
  }

  return correction;
}

// Compute derived coordinates (z, phi, r) for tracking particle (TP)
std::vector<double> Phase2OTValidateStub::getTPDerivedCoords(edm::Ptr<TrackingParticle> associatedTP,
                                                             bool isBarrel,
                                                             float stub_z,
                                                             float stub_r) const {
  double tp_phi = -99;
  double tp_r = -99;
  double tp_z = -99;

  trklet::Settings settings;
  double bfield_ = settings.bfield();
  double c_ = settings.c();

  // Get values from the tracking particle associatedTP
  double tp_pt = associatedTP->pt();
  double tp_charge = associatedTP->charge();
  float tp_z0 = associatedTP->vertex().z();
  double tp_t = associatedTP->tanl();
  double tp_rinv = (tp_charge * bfield_) / (tp_pt);

  if (isBarrel) {
    tp_r = stub_r;
    tp_phi = associatedTP->p4().phi() - std::asin(tp_r * tp_rinv * c_ / 2.0E2);
    tp_phi = reco::reducePhiRange(tp_phi);
    tp_z = tp_z0 + (2.0E2 / c_) * tp_t * (1 / tp_rinv) * std::asin(tp_r * tp_rinv * c_ / 2.0E2);
  } else {
    tp_z = stub_z;
    tp_phi = associatedTP->p4().phi() - (tp_z - tp_z0) * tp_rinv * c_ / 2.0E2 / tp_t;
    tp_phi = reco::reducePhiRange(tp_phi);
    tp_r = 2.0E2 / tp_rinv / c_ * std::sin((tp_z - tp_z0) * tp_rinv * c_ / 2.0E2 / tp_t);
  }

  std::vector<double> tpDerived_coords{tp_z, tp_phi, tp_r};
  return tpDerived_coords;
}

// ------------ method called for each event  ------------
void Phase2OTValidateStub::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Tracking Particles
  edm::Handle<std::vector<TrackingParticle>> trackingParticleHandle;
  iEvent.getByToken(trackingParticleToken_, trackingParticleHandle);

  /// Track Trigger Stubs
  edm::Handle<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> Phase2TrackerDigiTTStubHandle;
  iEvent.getByToken(tagTTStubsToken_, Phase2TrackerDigiTTStubHandle);
  edm::Handle<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTStubHandle;
  iEvent.getByToken(ttStubMCTruthToken_, MCTruthTTStubHandle);
  edm::Handle<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTClusterHandle;
  iEvent.getByToken(ttClusterMCTruthToken_, MCTruthTTClusterHandle);

  trklet::Settings settings;
  double bfield_ = settings.bfield();
  double c_ = settings.c();

  // Geometries
  const TrackerGeometry* theTrackerGeom = tkGeom_;
  const TrackerTopology* tTopo = tTopo_;

  /// Loop over input Stubs for basic histogram filling (e.g., Stub_RZ)
  typename edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>::const_iterator inputIter;
  typename edmNew::DetSet<TTStub<Ref_Phase2TrackerDigi_>>::const_iterator contentIter;
  // Adding protection
  if (!Phase2TrackerDigiTTStubHandle.isValid() || !MCTruthTTStubHandle.isValid()) {
    edm::LogError("Phase2OTValidateStub") << "Invalid handle(s) detected.";
    return;
  }

  for (inputIter = Phase2TrackerDigiTTStubHandle->begin(); inputIter != Phase2TrackerDigiTTStubHandle->end();
       ++inputIter) {
    for (contentIter = inputIter->begin(); contentIter != inputIter->end(); ++contentIter) {
      /// Make reference stub
      edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>> tempStubRef =
          edmNew::makeRefTo(Phase2TrackerDigiTTStubHandle, contentIter);

      /// Get det ID (place of the stub)
      //  tempStubRef->getDetId() gives the stackDetId, not rawId
      DetId detIdStub = tkGeom_->idToDet((tempStubRef->clusterRef(0))->getDetId())->geographicalId();

      /// Get trigger displacement/offset
      // double rawBend = tempStubRef->rawBend();
      // double bendOffset = tempStubRef->bendOffset();

      /// Define position stub by position inner cluster
      MeasurementPoint mp = (tempStubRef->clusterRef(0))->findAverageLocalCoordinates();
      const GeomDet* theGeomDet = tkGeom_->idToDet(detIdStub);
      Global3DPoint posStub = theGeomDet->surface().toGlobal(theGeomDet->topology().localPosition(mp));

      Stub_RZ->Fill(posStub.z(), posStub.perp());
    }
  }

  // Loop over geometric detectors
  for (auto gd = tkGeom_->dets().begin(); gd != tkGeom_->dets().end(); gd++) {
    DetId detid = (*gd)->geographicalId();

    // Check if detid belongs to TOB or TID subdetectors
    if (detid.subdetId() != StripSubdetector::TOB && detid.subdetId() != StripSubdetector::TID)
      continue;

    // Process only the lower part of the stack
    if (!tTopo_->isLower(detid))
      continue;

    // Get the stack DetId
    DetId stackDetid = tTopo_->stack(detid);

    // Check if the stackDetid exists in TTStubHandle
    if (Phase2TrackerDigiTTStubHandle->find(stackDetid) == Phase2TrackerDigiTTStubHandle->end())
      continue;

    // Get the DetSets of the Clusters
    edmNew::DetSet<TTStub<Ref_Phase2TrackerDigi_>> stubs = (*Phase2TrackerDigiTTStubHandle)[stackDetid];

    // Calculate detector module positions
    const GeomDetUnit* det0 = tkGeom_->idToDetUnit(detid);
    const GeomDetUnit* det1 = tkGeom_->idToDetUnit(tTopo_->partnerDetId(detid));
    if (!det0 || !det1) {
      edm::LogError("Phase2OTValidateStub") << "Error: det0 or det1 is null";
      continue;
    }
    float modMinR = std::min(det0->position().perp(), det1->position().perp());
    float modMaxR = std::max(det0->position().perp(), det1->position().perp());
    float modMinZ = std::min(det0->position().z(), det1->position().z());
    float modMaxZ = std::max(det0->position().z(), det1->position().z());
    float sensorSpacing = sqrt((modMaxR - modMinR) * (modMaxR - modMinR) + (modMaxZ - modMinZ) * (modMaxZ - modMinZ));

    // Calculate strip pitch
    const PixelGeomDetUnit* theGeomDetUnit = dynamic_cast<const PixelGeomDetUnit*>(det0);
    if (!theGeomDetUnit) {
      edm::LogError("Phase2OTValidateStub") << "Error: theGeomDetUnit is null";
      continue;
    }
    const PixelTopology& topo = theGeomDetUnit->specificTopology();
    float stripPitch = topo.pitch().first;

    // Loop over input stubs
    for (auto stubIter = stubs.begin(); stubIter != stubs.end(); ++stubIter) {
      // Create reference to the stub
      edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>> tempStubPtr =
          edmNew::makeRefTo(Phase2TrackerDigiTTStubHandle, stubIter);

      // Check if the stub is genuine
      if (!MCTruthTTStubHandle->isGenuine(tempStubPtr))
        continue;

      // Get det ID from the stub
      DetId detIdStub = tkGeom_->idToDet((tempStubPtr->clusterRef(0))->getDetId())->geographicalId();

      // Retrieve geometrical detector
      const GeomDet* theGeomDet = tkGeom_->idToDet(detIdStub);
      if (!theGeomDet) {
        edm::LogError("Phase2OTValidateStub") << "Error: theGeomDet is null";
        continue;
      }

      // Retrieve tracking particle associated with TTStub
      edm::Ptr<TrackingParticle> associatedTP = MCTruthTTStubHandle->findTrackingParticlePtr(tempStubPtr);
      if (associatedTP.isNull())
        continue;

      // Determine layer and subdetector information
      int isBarrel = 0;
      int layer = -999999;
      if (detid.subdetId() == StripSubdetector::TOB) {
        isBarrel = 1;
        layer = static_cast<int>(tTopo_->layer(detid));
      } else if (detid.subdetId() == StripSubdetector::TID) {
        isBarrel = 0;
        layer = static_cast<int>(tTopo_->layer(detid));
      } else {
        edm::LogVerbatim("Tracklet") << "WARNING -- neither TOB nor TID stub, shouldn't happen...";
        layer = -1;
      }

      int isPSmodule = (topo.nrows() == 960) ? 1 : 0;

      // Calculate local coordinates of clusters
      MeasurementPoint innerClusterCoords = tempStubPtr->clusterRef(0)->findAverageLocalCoordinatesCentered();
      MeasurementPoint outerClusterCoords = tempStubPtr->clusterRef(1)->findAverageLocalCoordinatesCentered();

      // Convert local coordinates to global positions
      Global3DPoint innerClusterGlobalPos =
          theGeomDet->surface().toGlobal(theGeomDet->topology().localPosition(innerClusterCoords));
      Global3DPoint outerClusterGlobalPos =
          theGeomDet->surface().toGlobal(theGeomDet->topology().localPosition(outerClusterCoords));

      // Determine maximum Z positions of the stubs
      float stub_maxZ = std::max(innerClusterGlobalPos.z(), outerClusterGlobalPos.z());

      // Stub parameters
      float stub_phi = innerClusterGlobalPos.phi();
      float stub_r = innerClusterGlobalPos.perp();
      float stub_z = innerClusterGlobalPos.z();

      // Tracking particle parameters
      int tp_charge = associatedTP->charge();
      float tp_pt = associatedTP->p4().pt();
      float tp_eta = associatedTP->p4().eta();
      float tp_d0 = associatedTP->d0();
      float tp_vx = associatedTP->vx();
      float tp_vy = associatedTP->vy();
      float tp_vz = associatedTP->vz();
      float tp_Lxy = std::sqrt(tp_vx * tp_vx + tp_vy * tp_vy);

      if (tp_charge == 0)
        continue;
      if (tp_pt < TP_minPt)
        continue;
      if (std::abs(tp_eta) > TP_maxEta)
        continue;
      if (std::abs(tp_vz) > TP_maxVtxZ)
        continue;
      if (std::abs(tp_d0) > TP_maxD0)
        continue;
      if (std::abs(tp_Lxy) > TP_maxLxy)
        continue;

      // Derived coordinates
      std::vector<double> tpDerivedCoords = getTPDerivedCoords(associatedTP, isBarrel, stub_z, stub_r);
      float tp_z = tpDerivedCoords[0];
      float tp_phi = tpDerivedCoords[1];
      float tp_r = tpDerivedCoords[2];

      // Trigger information
      float trigBend = tempStubPtr->bendFE();
      if (!isBarrel && stub_maxZ < 0.0) {
        trigBend = -trigBend;
      }

      float correctionValue = phiOverBendCorrection(isBarrel, stub_z, stub_r, tTopo_, detid, det0, det1);
      float trackBend =
          -(sensorSpacing * stub_r * bfield_ * c_ * tp_charge) / (stripPitch * 2.0E2 * tp_pt * correctionValue);

      float bendRes = trackBend - trigBend;
      float zRes = tp_z - stub_z;
      float phiRes = tp_phi - stub_phi;
      float rRes = tp_r - stub_r;

      // Histograms for z_res, phi_res, and r_res based on module type and
      // location
      if (isBarrel == 1) {
        bend_res_barrel->Fill(bendRes);
        if (isPSmodule) {
          z_res_isPS_barrel->Fill(zRes);
          phi_res_isPS_barrel->Fill(phiRes);
        } else {
          z_res_is2S_barrel->Fill(zRes);
          phi_res_is2S_barrel->Fill(phiRes);
        }
      } else {
        if (stub_maxZ > 0) {
          if (isPSmodule) {
            r_res_isPS_fw_endcap->Fill(rRes);
          } else {
            r_res_is2S_fw_endcap->Fill(rRes);
          }
        } else {
          if (isPSmodule) {
            r_res_isPS_bw_endcap->Fill(rRes);
          } else {
            r_res_is2S_bw_endcap->Fill(rRes);
          }
        }
      }

      // Ensure that the vectors are correctly assigned before use
      if (isBarrel == 1) {
        phi_res_vec = &phi_res_barrel_layers;
        bend_res_vec = &bend_res_barrel_layers;
      } else {
        if (stub_maxZ > 0) {
          // Forward endcap
          bend_res_fw_endcap->Fill(bendRes);
          phi_res_fw_endcap->Fill(phiRes);
          phi_res_vec = &phi_res_fw_endcap_discs;
          bend_res_vec = &bend_res_fw_endcap_discs;
        } else {
          // Backward endcap
          bend_res_bw_endcap->Fill(bendRes);
          phi_res_bw_endcap->Fill(phiRes);
          phi_res_vec = &phi_res_bw_endcap_discs;
          bend_res_vec = &bend_res_bw_endcap_discs;
        }
      }
      // Fill the appropriate histogram based on layer/disc
      if (layer >= 1 && layer <= trklet::N_LAYER) {
        (*bend_res_vec)[layer - 1]->Fill(bendRes);
        (*phi_res_vec)[layer - 1]->Fill(phiRes);
      }
    }  // end loop over input stubs
  }  // end loop over geometric detectors

  // loop over tracking particles
  for (size_t i = 0; i < trackingParticleHandle->size(); ++i) {
    edm::Ptr<TrackingParticle> tp_ptr(trackingParticleHandle, i);

    // Calculate nLayers variable
    std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
        theStubRefs = MCTruthTTStubHandle->findTTStubRefs(tp_ptr);

    float tmp_tp_pt = tp_ptr->pt();
    float tmp_tp_eta = tp_ptr->eta();

    int hasStubInLayer[11] = {0};
    for (unsigned int is = 0; is < theStubRefs.size(); is++) {
      DetId detid(theStubRefs.at(is)->getDetId());
      int layer = -1;
      if (detid.subdetId() == StripSubdetector::TOB)
        layer = static_cast<int>(tTopo->layer(detid)) - 1;  // fill in array as entries 0-5
      else if (detid.subdetId() == StripSubdetector::TID)
        layer = static_cast<int>(tTopo->layer(detid)) + 5;  // fill in array as entries 6-10

      // treat genuine stubs separately (==2 is genuine, ==1 is not)
      if (MCTruthTTStubHandle->findTrackingParticlePtr(theStubRefs.at(is)).isNull() && hasStubInLayer[layer] < 2)
        hasStubInLayer[layer] = 1;
      else
        hasStubInLayer[layer] = 2;
    }

    int nStubLayerTP = 0;
    for (int isum = 0; isum < 11; isum++) {
      if (hasStubInLayer[isum] >= 1)
        nStubLayerTP += 1;
    }

    if (std::fabs(tmp_tp_eta) > TP_maxEta)
      continue;
    int nStubTP = -1;
    if (MCTruthTTStubHandle.isValid()) {
      std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
          theStubRefs = MCTruthTTStubHandle->findTTStubRefs(tp_ptr);
      nStubTP = (int)theStubRefs.size();
    }
    if (MCTruthTTClusterHandle.isValid() && MCTruthTTClusterHandle->findTTClusterRefs(tp_ptr).empty())
      continue;

    float tmp_tp_z0, tmp_tp_Lxy, unused_d0;
    std::tie(tmp_tp_z0, tmp_tp_Lxy, unused_d0) = phase2tkutil::computeZ0LxyD0(*tp_ptr);
    (void)unused_d0;  // suppress unused variable warning

    if (std::fabs(tmp_tp_z0) > TP_maxVtxZ)
      continue;
    if (tmp_tp_pt < TP_minPt)
      continue;
    if (tmp_tp_Lxy > TP_maxLxy)
      continue;
    if (nStubTP < TP_minNStub || nStubLayerTP < TP_minNLayersStub)
      continue;

    // Find all clusters that can be associated to a tracking particle with at
    // least one hit
    std::vector<edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>, TTCluster<Ref_Phase2TrackerDigi_>>>
        associatedClusters = MCTruthTTClusterHandle->findTTClusterRefs(tp_ptr);

    // Loop through associated clusters
    for (std::size_t k = 0; k < associatedClusters.size(); ++k) {
      const auto& clusA = associatedClusters[k];

      // Get cluster details
      DetId clusdetid = clusA->getDetId();
      if (clusdetid.subdetId() != StripSubdetector::TOB && clusdetid.subdetId() != StripSubdetector::TID)
        continue;

      bool isGenuine = MCTruthTTClusterHandle->isGenuine(clusA);
      if (!isGenuine)
        continue;

      DetId detidA = tTopo->stack(clusdetid);
      const GeomDetUnit* detA = theTrackerGeom->idToDetUnit(clusdetid);
      const PixelGeomDetUnit* theGeomDetA = dynamic_cast<const PixelGeomDetUnit*>(detA);
      const PixelTopology* topoA = dynamic_cast<const PixelTopology*>(&(theGeomDetA->specificTopology()));
      GlobalPoint coordsA =
          theGeomDetA->surface().toGlobal(topoA->localPosition(clusA->findAverageLocalCoordinatesCentered()));

      int isBarrel = 0;
      if (clusdetid.subdetId() == StripSubdetector::TOB) {
        isBarrel = 1;
      } else if (clusdetid.subdetId() == StripSubdetector::TID) {
        isBarrel = 0;
      } else {
        edm::LogVerbatim("Tracklet") << "WARNING -- neither TOB or TID stub, shouldn't happen...";
      }

      if (isBarrel == 1) {
        gen_clusters_barrel->Fill(tmp_tp_pt);
        gen_clusters_zoom_barrel->Fill(tmp_tp_pt);
      } else {
        gen_clusters_endcaps->Fill(tmp_tp_pt);
        gen_clusters_zoom_endcaps->Fill(tmp_tp_pt);
      }

      // If there are stubs on the same detid, loop on those stubs
      if (Phase2TrackerDigiTTStubHandle->find(detidA) != Phase2TrackerDigiTTStubHandle->end()) {
        edmNew::DetSet<TTStub<Ref_Phase2TrackerDigi_>> stubs = (*Phase2TrackerDigiTTStubHandle)[detidA];
        for (auto stubIter = stubs.begin(); stubIter != stubs.end(); ++stubIter) {
          auto stubRef = edmNew::makeRefTo(Phase2TrackerDigiTTStubHandle, stubIter);

          // Retrieve clusters of stubs
          auto clusterRefB = stubIter->clusterRef(0);
          auto clusterRefC = stubIter->clusterRef(1);

          // Retrieve sensor DetIds from the stub's clusters
          DetId detIdB = stubIter->clusterRef(0)->getDetId();
          DetId detIdC = stubIter->clusterRef(1)->getDetId();

          const GeomDetUnit* detB = theTrackerGeom->idToDetUnit(detIdB);
          const GeomDetUnit* detC = theTrackerGeom->idToDetUnit(detIdC);
          const PixelGeomDetUnit* theGeomDetB = dynamic_cast<const PixelGeomDetUnit*>(detB);
          const PixelGeomDetUnit* theGeomDetC = dynamic_cast<const PixelGeomDetUnit*>(detC);
          const PixelTopology* topoB = dynamic_cast<const PixelTopology*>(&(theGeomDetB->specificTopology()));
          const PixelTopology* topoC = dynamic_cast<const PixelTopology*>(&(theGeomDetC->specificTopology()));

          GlobalPoint coordsB = theGeomDetB->surface().toGlobal(
              topoB->localPosition(stubIter->clusterRef(0)->findAverageLocalCoordinatesCentered()));
          GlobalPoint coordsC = theGeomDetC->surface().toGlobal(
              topoC->localPosition(stubIter->clusterRef(1)->findAverageLocalCoordinatesCentered()));

          if (coordsA.x() == coordsB.x() || coordsA.x() == coordsC.x()) {
            edm::Ptr<TrackingParticle> stubTP = MCTruthTTStubHandle->findTrackingParticlePtr(
                edmNew::makeRefTo(Phase2TrackerDigiTTStubHandle, stubIter));
            if (stubTP.isNull())
              continue;
            float stub_tp_pt = stubTP->pt();
            if (stub_tp_pt == tmp_tp_pt) {
              if (isBarrel == 1) {
                gen_clusters_if_stub_barrel->Fill(tmp_tp_pt);
                gen_clusters_if_stub_zoom_barrel->Fill(tmp_tp_pt);
              } else {
                gen_clusters_if_stub_endcaps->Fill(tmp_tp_pt);
                gen_clusters_if_stub_zoom_endcaps->Fill(tmp_tp_pt);
              }
              break;
            }
          }  // end if stub cluster coords.x matches associated cluster coords.x
        }  // end loop over stubs on the same detid as associated clusters
      }  // end if stubs on same detid
    }  // end loop over associated clusters
  }  // end loop over tracking particles
}  // end of method

// ------------ method called when starting to processes a run  ------------
void Phase2OTValidateStub::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& run, edm::EventSetup const& es) {
  edm::ParameterSet psTTStub_RZ = conf_.getParameter<edm::ParameterSet>("TH2TTStub_RZ");
  edm::ParameterSet ps_2S_Res = conf_.getParameter<edm::ParameterSet>("TH1_2S_Res");
  edm::ParameterSet ps_PS_Res = conf_.getParameter<edm::ParameterSet>("TH1_PS_Res");
  edm::ParameterSet psPhi_Res = conf_.getParameter<edm::ParameterSet>("TH1Phi_Res");
  edm::ParameterSet psBend_Res = conf_.getParameter<edm::ParameterSet>("TH1Bend_Res");
  edm::ParameterSet psEffic_pt = conf_.getParameter<edm::ParameterSet>("TH1Effic_pt");
  edm::ParameterSet psEffic_pt_zoom = conf_.getParameter<edm::ParameterSet>("TH1Effic_pt_zoom");
  std::string HistoName;
  iBooker.setCurrentFolder(topFolderName_);
  // 2D histogram for stub_RZ
  HistoName = "Stub_RZ";
  Stub_RZ = iBooker.book2D(HistoName,
                           HistoName,
                           psTTStub_RZ.getParameter<int32_t>("Nbinsx"),
                           psTTStub_RZ.getParameter<double>("xmin"),
                           psTTStub_RZ.getParameter<double>("xmax"),
                           psTTStub_RZ.getParameter<int32_t>("Nbinsy"),
                           psTTStub_RZ.getParameter<double>("ymin"),
                           psTTStub_RZ.getParameter<double>("ymax"));

  iBooker.setCurrentFolder(topFolderName_ + "/Residual");
  // z-res for PS modules
  HistoName = "#Delta z Barrel PS modules";
  z_res_isPS_barrel = iBooker.book1D(HistoName,
                                     HistoName,
                                     ps_PS_Res.getParameter<int32_t>("Nbinsx"),
                                     ps_PS_Res.getParameter<double>("xmin"),
                                     ps_PS_Res.getParameter<double>("xmax"));
  z_res_isPS_barrel->setAxisTitle("tp_z - stub_z", 1);
  z_res_isPS_barrel->setAxisTitle("events ", 2);

  // z-res for 2S modules
  HistoName = "#Delta z Barrel 2S modules";
  z_res_is2S_barrel = iBooker.book1D(HistoName,
                                     HistoName,
                                     ps_2S_Res.getParameter<int32_t>("Nbinsx"),
                                     ps_2S_Res.getParameter<double>("xmin"),
                                     ps_2S_Res.getParameter<double>("xmax"));
  z_res_is2S_barrel->setAxisTitle("tp_z - stub_z [cm]", 1);
  z_res_is2S_barrel->setAxisTitle("events ", 2);

  // r-res for fw endcap PS modules
  HistoName = "#Delta r FW Endcap PS modules";
  r_res_isPS_fw_endcap = iBooker.book1D(HistoName,
                                        HistoName,
                                        ps_PS_Res.getParameter<int32_t>("Nbinsx"),
                                        ps_PS_Res.getParameter<double>("xmin"),
                                        ps_PS_Res.getParameter<double>("xmax"));
  r_res_isPS_fw_endcap->setAxisTitle("tp_r - stub_r [cm]", 1);
  r_res_isPS_fw_endcap->setAxisTitle("events ", 2);

  // r-res for fw endcap 2S modules
  HistoName = "#Delta r FW Endcap 2S modules";
  r_res_is2S_fw_endcap = iBooker.book1D(HistoName,
                                        HistoName,
                                        ps_2S_Res.getParameter<int32_t>("Nbinsx"),
                                        ps_2S_Res.getParameter<double>("xmin"),
                                        ps_2S_Res.getParameter<double>("xmax"));
  r_res_is2S_fw_endcap->setAxisTitle("tp_r - stub_r [cm]", 1);
  r_res_is2S_fw_endcap->setAxisTitle("events ", 2);

  // r-res for bw endcap PS modules
  HistoName = "#Delta r BW Endcap PS modules";
  r_res_isPS_bw_endcap = iBooker.book1D(HistoName,
                                        HistoName,
                                        ps_PS_Res.getParameter<int32_t>("Nbinsx"),
                                        ps_PS_Res.getParameter<double>("xmin"),
                                        ps_PS_Res.getParameter<double>("xmax"));
  r_res_isPS_bw_endcap->setAxisTitle("tp_r - stub_r [cm]", 1);
  r_res_isPS_bw_endcap->setAxisTitle("events ", 2);

  // r-res for bw endcap 2S modules
  HistoName = "#Delta r BW Endcap 2S modules";
  r_res_is2S_bw_endcap = iBooker.book1D(HistoName,
                                        HistoName,
                                        ps_2S_Res.getParameter<int32_t>("Nbinsx"),
                                        ps_2S_Res.getParameter<double>("xmin"),
                                        ps_2S_Res.getParameter<double>("xmax"));
  r_res_is2S_bw_endcap->setAxisTitle("tp_r - stub_r [cm]", 1);
  r_res_is2S_bw_endcap->setAxisTitle("events ", 2);

  // histograms for phi_res and bend_res
  HistoName = "#Delta #phi Barrel PS modules";
  phi_res_isPS_barrel = iBooker.book1D(HistoName,
                                       HistoName,
                                       psPhi_Res.getParameter<int32_t>("Nbinsx"),
                                       psPhi_Res.getParameter<double>("xmin"),
                                       psPhi_Res.getParameter<double>("xmax"));
  phi_res_isPS_barrel->setAxisTitle("tp_phi - stub_phi", 1);
  phi_res_isPS_barrel->setAxisTitle("events", 2);

  HistoName = "#Delta #phi Barrel 2S modules";
  phi_res_is2S_barrel = iBooker.book1D(HistoName,
                                       HistoName,
                                       psPhi_Res.getParameter<int32_t>("Nbinsx"),
                                       psPhi_Res.getParameter<double>("xmin"),
                                       psPhi_Res.getParameter<double>("xmax"));

  HistoName = "#Delta #phi FW Endcap";
  phi_res_fw_endcap = iBooker.book1D(HistoName,
                                     HistoName,
                                     psPhi_Res.getParameter<int32_t>("Nbinsx"),
                                     psPhi_Res.getParameter<double>("xmin"),
                                     psPhi_Res.getParameter<double>("xmax"));

  HistoName = "#Delta #phi BW Endcap";
  phi_res_bw_endcap = iBooker.book1D(HistoName,
                                     HistoName,
                                     psPhi_Res.getParameter<int32_t>("Nbinsx"),
                                     psPhi_Res.getParameter<double>("xmin"),
                                     psPhi_Res.getParameter<double>("xmax"));

  HistoName = "#Delta bend FW Endcap";
  bend_res_fw_endcap = iBooker.book1D(HistoName,
                                      HistoName,
                                      psBend_Res.getParameter<int32_t>("Nbinsx"),
                                      psBend_Res.getParameter<double>("xmin"),
                                      psBend_Res.getParameter<double>("xmax"));

  HistoName = "#Delta bend BW Endcap";
  bend_res_bw_endcap = iBooker.book1D(HistoName,
                                      HistoName,
                                      psBend_Res.getParameter<int32_t>("Nbinsx"),
                                      psBend_Res.getParameter<double>("xmin"),
                                      psBend_Res.getParameter<double>("xmax"));

  HistoName = "#Delta bend Barrel";
  bend_res_barrel = iBooker.book1D(HistoName,
                                   HistoName,
                                   psBend_Res.getParameter<int32_t>("Nbinsx"),
                                   psBend_Res.getParameter<double>("xmin"),
                                   psBend_Res.getParameter<double>("xmax"));

  // barrel layers
  for (int i = 0; i < trklet::N_LAYER; ++i) {
    std::string HistoName = "#Delta #phi Barrel L" + std::to_string(i + 1);
    phi_res_barrel_layers[i] = iBooker.book1D(HistoName,
                                              HistoName,
                                              psPhi_Res.getParameter<int32_t>("Nbinsx"),
                                              psPhi_Res.getParameter<double>("xmin"),
                                              psPhi_Res.getParameter<double>("xmax"));
    phi_res_barrel_layers[i]->setAxisTitle("tp_phi - stub_phi", 1);
    phi_res_barrel_layers[i]->setAxisTitle("events", 2);

    HistoName = "#Delta bend Barrel L" + std::to_string(i + 1);
    bend_res_barrel_layers[i] = iBooker.book1D(HistoName,
                                               HistoName,
                                               psBend_Res.getParameter<int32_t>("Nbinsx"),
                                               psBend_Res.getParameter<double>("xmin"),
                                               psBend_Res.getParameter<double>("xmax"));
    bend_res_barrel_layers[i]->setAxisTitle("tp_bend - stub_bend", 1);
    bend_res_barrel_layers[i]->setAxisTitle("events", 2);
  }

  // endcap discs
  for (int i = 0; i < trklet::N_DISK; ++i) {
    std::string HistoName = "#Delta #phi FW Endcap D" + std::to_string(i + 1);
    phi_res_fw_endcap_discs[i] = iBooker.book1D(HistoName,
                                                HistoName,
                                                psPhi_Res.getParameter<int32_t>("Nbinsx"),
                                                psPhi_Res.getParameter<double>("xmin"),
                                                psPhi_Res.getParameter<double>("xmax"));
    phi_res_fw_endcap_discs[i]->setAxisTitle("tp_phi - stub_phi", 1);
    phi_res_fw_endcap_discs[i]->setAxisTitle("events", 2);

    HistoName = "#Delta bend FW Endcap D" + std::to_string(i + 1);
    bend_res_fw_endcap_discs[i] = iBooker.book1D(HistoName,
                                                 HistoName,
                                                 psBend_Res.getParameter<int32_t>("Nbinsx"),
                                                 psBend_Res.getParameter<double>("xmin"),
                                                 psBend_Res.getParameter<double>("xmax"));
    bend_res_fw_endcap_discs[i]->setAxisTitle("tp_bend - stub_bend", 1);
    bend_res_fw_endcap_discs[i]->setAxisTitle("events", 2);

    HistoName = "#Delta #phi BW Endcap D" + std::to_string(i + 1);
    phi_res_bw_endcap_discs[i] = iBooker.book1D(HistoName,
                                                HistoName,
                                                psPhi_Res.getParameter<int32_t>("Nbinsx"),
                                                psPhi_Res.getParameter<double>("xmin"),
                                                psPhi_Res.getParameter<double>("xmax"));
    phi_res_bw_endcap_discs[i]->setAxisTitle("tp_phi - stub_phi", 1);
    phi_res_bw_endcap_discs[i]->setAxisTitle("events", 2);

    HistoName = "#Delta bend BW Endcap D" + std::to_string(i + 1);
    bend_res_bw_endcap_discs[i] = iBooker.book1D(HistoName,
                                                 HistoName,
                                                 psBend_Res.getParameter<int32_t>("Nbinsx"),
                                                 psBend_Res.getParameter<double>("xmin"),
                                                 psBend_Res.getParameter<double>("xmax"));
    bend_res_bw_endcap_discs[i]->setAxisTitle("tp_bend - stub_bend", 1);
    bend_res_bw_endcap_discs[i]->setAxisTitle("events", 2);
  }

  // 1D plots for efficiency
  iBooker.setCurrentFolder(topFolderName_ + "/EfficiencyIngredients");

  // Gen clusters barrel
  HistoName = "gen_clusters_barrel";
  gen_clusters_barrel = iBooker.book1D(HistoName,
                                       HistoName,
                                       psEffic_pt.getParameter<int32_t>("Nbinsx"),
                                       psEffic_pt.getParameter<double>("xmin"),
                                       psEffic_pt.getParameter<double>("xmax"));
  gen_clusters_barrel->setAxisTitle("p_{T} [GeV]", 1);
  gen_clusters_barrel->setAxisTitle("# tracking particles", 2);

  // Gen clusters if stub barrel
  HistoName = "gen_clusters_if_stub_barrel";
  gen_clusters_if_stub_barrel = iBooker.book1D(HistoName,
                                               HistoName,
                                               psEffic_pt.getParameter<int32_t>("Nbinsx"),
                                               psEffic_pt.getParameter<double>("xmin"),
                                               psEffic_pt.getParameter<double>("xmax"));
  gen_clusters_if_stub_barrel->setAxisTitle("p_{T} [GeV]", 1);
  gen_clusters_if_stub_barrel->setAxisTitle("# tracking particles", 2);

  // Gen clusters endcaps
  HistoName = "gen_clusters_endcaps";
  gen_clusters_endcaps = iBooker.book1D(HistoName,
                                        HistoName,
                                        psEffic_pt.getParameter<int32_t>("Nbinsx"),
                                        psEffic_pt.getParameter<double>("xmin"),
                                        psEffic_pt.getParameter<double>("xmax"));
  gen_clusters_endcaps->setAxisTitle("p_{T} [GeV]", 1);
  gen_clusters_endcaps->setAxisTitle("# tracking particles", 2);

  // Gen clusters if stub endcaps
  HistoName = "gen_clusters_if_stub_endcaps";
  gen_clusters_if_stub_endcaps = iBooker.book1D(HistoName,
                                                HistoName,
                                                psEffic_pt.getParameter<int32_t>("Nbinsx"),
                                                psEffic_pt.getParameter<double>("xmin"),
                                                psEffic_pt.getParameter<double>("xmax"));
  gen_clusters_if_stub_endcaps->setAxisTitle("p_{T} [GeV]", 1);
  gen_clusters_if_stub_endcaps->setAxisTitle("# tracking particles", 2);

  // Gen clusters pT zoom (0-10 GeV) barrel
  HistoName = "gen_clusters_zoom_barrel";
  gen_clusters_zoom_barrel = iBooker.book1D(HistoName,
                                            HistoName,
                                            psEffic_pt_zoom.getParameter<int32_t>("Nbinsx"),
                                            psEffic_pt_zoom.getParameter<double>("xmin"),
                                            psEffic_pt_zoom.getParameter<double>("xmax"));
  gen_clusters_zoom_barrel->setAxisTitle("p_{T} [GeV]", 1);
  gen_clusters_zoom_barrel->setAxisTitle("# tracking particles", 2);

  // Gen cluters if stub pT zoom (0-10 GeV) barrel
  HistoName = "gen_clusters_if_stub_zoom_barrel";
  gen_clusters_if_stub_zoom_barrel = iBooker.book1D(HistoName,
                                                    HistoName,
                                                    psEffic_pt_zoom.getParameter<int32_t>("Nbinsx"),
                                                    psEffic_pt_zoom.getParameter<double>("xmin"),
                                                    psEffic_pt_zoom.getParameter<double>("xmax"));
  gen_clusters_if_stub_zoom_barrel->setAxisTitle("p_{T} [GeV]", 1);
  gen_clusters_if_stub_zoom_barrel->setAxisTitle("# tracking particles", 2);

  // Gen clusters pT zoom (0-10 GeV) endcaps
  HistoName = "gen_clusters_zoom_endcaps";
  gen_clusters_zoom_endcaps = iBooker.book1D(HistoName,
                                             HistoName,
                                             psEffic_pt_zoom.getParameter<int32_t>("Nbinsx"),
                                             psEffic_pt_zoom.getParameter<double>("xmin"),
                                             psEffic_pt_zoom.getParameter<double>("xmax"));
  gen_clusters_zoom_endcaps->setAxisTitle("p_{T} [GeV]", 1);
  gen_clusters_zoom_endcaps->setAxisTitle("# tracking particles", 2);

  // Gen cluters if stub pT zoom (0-10 GeV) endcaps
  HistoName = "gen_clusters_if_stub_zoom_endcaps";
  gen_clusters_if_stub_zoom_endcaps = iBooker.book1D(HistoName,
                                                     HistoName,
                                                     psEffic_pt_zoom.getParameter<int32_t>("Nbinsx"),
                                                     psEffic_pt_zoom.getParameter<double>("xmin"),
                                                     psEffic_pt_zoom.getParameter<double>("xmax"));
  gen_clusters_if_stub_zoom_endcaps->setAxisTitle("p_{T} [GeV]", 1);
  gen_clusters_if_stub_zoom_endcaps->setAxisTitle("# tracking particles", 2);
}

void Phase2OTValidateStub::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // Phase2OTValidateStub
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 900);
    psd0.add<double>("xmax", 300);
    psd0.add<double>("xmin", -300);
    psd0.add<int>("Nbinsy", 900);
    psd0.add<double>("ymax", 120);
    psd0.add<double>("ymin", 0);
    desc.add<edm::ParameterSetDescription>("TH2TTStub_RZ", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 99);
    psd0.add<double>("xmax", 5.5);
    psd0.add<double>("xmin", -5.5);
    desc.add<edm::ParameterSetDescription>("TH1_2S_Res", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 99);
    psd0.add<double>("xmax", 2.0);
    psd0.add<double>("xmin", -2.0);
    desc.add<edm::ParameterSetDescription>("TH1_PS_Res", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 599);
    psd0.add<double>("xmax", 0.1);
    psd0.add<double>("xmin", -0.1);
    desc.add<edm::ParameterSetDescription>("TH1Phi_Res", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 59);
    psd0.add<double>("xmax", 5.0);
    psd0.add<double>("xmin", -5.5);
    desc.add<edm::ParameterSetDescription>("TH1Bend_Res", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 50);
    psd0.add<double>("xmax", 100);
    psd0.add<double>("xmin", 0);
    desc.add<edm::ParameterSetDescription>("TH1Effic_pt", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 50);
    psd0.add<double>("xmax", 10);
    psd0.add<double>("xmin", 0);
    desc.add<edm::ParameterSetDescription>("TH1Effic_pt_zoom", psd0);
  }

  desc.add<std::string>("TopFolderName", "TrackerPhase2OTStubV");
  desc.add<edm::InputTag>("TTStubs", edm::InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"));
  desc.add<edm::InputTag>("trackingParticleToken", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("MCTruthStubInputTag", edm::InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"));
  desc.add<edm::InputTag>("MCTruthClusterInputTag",
                          edm::InputTag("TTClusterAssociatorFromPixelDigis", "ClusterInclusive"));
  desc.add<int>("TP_minNStub", 4);
  desc.add<int>("TP_minNLayersStub", 4);
  desc.add<double>("TP_minPt", 1.5);
  desc.add<double>("TP_maxEta", 2.4);
  desc.add<double>("TP_maxVtxZ", 15.0);
  desc.add<double>("TP_maxD0", 1.0);
  desc.add<double>("TP_maxLxy", 1.0);
  descriptions.add("Phase2OTValidateStub", desc);
  // or use the following to generate the label from the module's C++ type
  // descriptions.addWithDefaultLabel(desc);
}
DEFINE_FWK_MODULE(Phase2OTValidateStub);

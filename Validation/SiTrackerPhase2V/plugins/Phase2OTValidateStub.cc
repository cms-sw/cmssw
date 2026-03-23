
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
// Updated by: Brandi Skipworth, 2026

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
#include "Validation/SiTrackerPhase2V/interface/TrackerPhase2HistUtil.h"
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

  // Number of stubs per event
  MonitorElement* number_of_stubs = nullptr;

  // delta_z hists barrel
  MonitorElement* z_res_isPS_barrel = nullptr;
  MonitorElement* z_res_is2S_barrel = nullptr;

  // delta_r hists endcaps
  MonitorElement* r_res_isPS_endcap = nullptr;
  MonitorElement* r_res_is2S_endcap = nullptr;

  // delta_r hists (Split into FW/BW)
  MonitorElement* r_res_isPS_fw_endcap = nullptr;
  MonitorElement* r_res_is2S_fw_endcap = nullptr;
  MonitorElement* r_res_isPS_bw_endcap = nullptr;
  MonitorElement* r_res_is2S_bw_endcap = nullptr;

  // delta_phi hists (PS / 2S, barrel vs endcap)
  MonitorElement* phi_res_isPS_barrel = nullptr;
  MonitorElement* phi_res_is2S_barrel = nullptr;
  MonitorElement* phi_res_isPS_endcap = nullptr;  // FW + BW combined
  MonitorElement* phi_res_is2S_endcap = nullptr;  // FW + BW combined

  // delta_phi hists (Specific FW/BW and Barrel layers)
  MonitorElement* phi_res_fw_endcap = nullptr;
  MonitorElement* phi_res_bw_endcap = nullptr;
  std::vector<MonitorElement*> phi_res_barrel_layers;
  std::vector<MonitorElement*> phi_res_fw_endcap_discs;
  std::vector<MonitorElement*> phi_res_bw_endcap_discs;

  // delta_bend hists (PS / 2S, barrel vs endcap)
  MonitorElement* bend_res_isPS_barrel = nullptr;
  MonitorElement* bend_res_is2S_barrel = nullptr;
  MonitorElement* bend_res_isPS_endcap = nullptr;  // FW + BW combined
  MonitorElement* bend_res_is2S_endcap = nullptr;  // FW + BW combined

  // delta_bend hists (General barrel/endcap and Barrel layers)
  MonitorElement* bend_res_fw_endcap = nullptr;
  MonitorElement* bend_res_bw_endcap = nullptr;
  std::vector<MonitorElement*> bend_res_barrel_layers;
  std::vector<MonitorElement*> bend_res_fw_endcap_discs;
  std::vector<MonitorElement*> bend_res_bw_endcap_discs;

  // Helper vectors
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

  int nStubs = 0;
  for (auto const& detSet : *Phase2TrackerDigiTTStubHandle) {
    nStubs += detSet.size();
  }
  number_of_stubs->Fill(nStubs);

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

      // Histograms for z_res, phi_res, bend_res, and r_res based on module type and location
      if (isBarrel == 1) {
        // --- BARREL LOGIC ---
        phi_res_vec = &phi_res_barrel_layers;
        bend_res_vec = &bend_res_barrel_layers;

        if (isPSmodule) {
          z_res_isPS_barrel->Fill(zRes);
          phi_res_isPS_barrel->Fill(phiRes);
          bend_res_isPS_barrel->Fill(bendRes);
        } else {
          z_res_is2S_barrel->Fill(zRes);
          phi_res_is2S_barrel->Fill(phiRes);
          bend_res_is2S_barrel->Fill(bendRes);
        }
      } else {
        // Fill Summary Endcap plots
        if (isPSmodule) {
          r_res_isPS_endcap->Fill(rRes);
          phi_res_isPS_endcap->Fill(phiRes);
          bend_res_isPS_endcap->Fill(bendRes);
        } else {
          r_res_is2S_endcap->Fill(rRes);
          phi_res_is2S_endcap->Fill(phiRes);
          bend_res_is2S_endcap->Fill(bendRes);
        }

        // Specific FW/BW Logic (ONLY runs if NOT Barrel)
        if (stub_maxZ > 0) {
          // Forward Endcap
          bend_res_fw_endcap->Fill(bendRes);
          phi_res_fw_endcap->Fill(phiRes);

          // Set pointers to FW vectors
          phi_res_vec = &phi_res_fw_endcap_discs;
          bend_res_vec = &bend_res_fw_endcap_discs;

          if (isPSmodule) {
            r_res_isPS_fw_endcap->Fill(rRes);
          } else {
            r_res_is2S_fw_endcap->Fill(rRes);
          }
        } else {
          // Backward Endcap
          bend_res_bw_endcap->Fill(bendRes);
          phi_res_bw_endcap->Fill(phiRes);

          // Set pointers to BW vectors
          phi_res_vec = &phi_res_bw_endcap_discs;
          bend_res_vec = &bend_res_bw_endcap_discs;

          if (isPSmodule) {
            r_res_isPS_bw_endcap->Fill(rRes);
          } else {
            r_res_is2S_bw_endcap->Fill(rRes);
          }
        }
      }

      if (isBarrel) {
        if (layer >= 1 && layer <= trklet::N_LAYER) {
          (*bend_res_vec)[layer - 1]->Fill(bendRes);
          (*phi_res_vec)[layer - 1]->Fill(phiRes);
        }
      } else {
        if (layer >= 1 && layer <= trklet::N_DISK) {
          (*bend_res_vec)[layer - 1]->Fill(bendRes);
          (*phi_res_vec)[layer - 1]->Fill(phiRes);
        }
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

    // Find all clusters that can be associated to a tracking particle with at least one hit
    std::vector<edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>, TTCluster<Ref_Phase2TrackerDigi_>>>
        associatedClusters = MCTruthTTClusterHandle->findTTClusterRefs(tp_ptr);

    int nValidClustersBarrel = 0;
    int nValidClustersEndcap = 0;

    // Count genuine clusters first
    for (const auto& clus : associatedClusters) {
      if (!MCTruthTTClusterHandle->isGenuine(clus))
        continue;
      DetId detid = clus->getDetId();
      if (detid.subdetId() == StripSubdetector::TOB)
        nValidClustersBarrel++;
      else if (detid.subdetId() == StripSubdetector::TID)
        nValidClustersEndcap++;
    }

    // Define weights (1.0 / N)
    float weightBarrel = (nValidClustersBarrel > 0) ? 1.0f / nValidClustersBarrel : 0.0f;
    float weightEndcap = (nValidClustersEndcap > 0) ? 1.0f / nValidClustersEndcap : 0.0f;

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

      // apply weight
      bool isBarrelBool = (clusdetid.subdetId() == StripSubdetector::TOB);
      float currentWeight = isBarrelBool ? weightBarrel : weightEndcap;

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

      // modified for weights
      if (isBarrel == 1) {
        gen_clusters_barrel->Fill(tmp_tp_pt, currentWeight);
        gen_clusters_zoom_barrel->Fill(tmp_tp_pt, currentWeight);
      } else {
        gen_clusters_endcaps->Fill(tmp_tp_pt, currentWeight);
        gen_clusters_zoom_endcaps->Fill(tmp_tp_pt, currentWeight);
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
              // modified for weights
              if (isBarrel == 1) {
                gen_clusters_if_stub_barrel->Fill(tmp_tp_pt, currentWeight);
                gen_clusters_if_stub_zoom_barrel->Fill(tmp_tp_pt, currentWeight);
              } else {
                gen_clusters_if_stub_endcaps->Fill(tmp_tp_pt, currentWeight);
                gen_clusters_if_stub_zoom_endcaps->Fill(tmp_tp_pt, currentWeight);
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
  edm::ParameterSet psNStubs = conf_.getParameter<edm::ParameterSet>("TH1NStubs");
  edm::ParameterSet ps_2S_Res = conf_.getParameter<edm::ParameterSet>("TH1_2S_Res");
  edm::ParameterSet ps_PS_Res = conf_.getParameter<edm::ParameterSet>("TH1_PS_Res");
  edm::ParameterSet psPhi_Res = conf_.getParameter<edm::ParameterSet>("TH1Phi_Res");
  edm::ParameterSet psBend_Res = conf_.getParameter<edm::ParameterSet>("TH1Bend_Res");
  edm::ParameterSet psEffic_pt = conf_.getParameter<edm::ParameterSet>("TH1Effic_pt");
  edm::ParameterSet psEffic_pt_zoom = conf_.getParameter<edm::ParameterSet>("TH1Effic_pt_zoom");
  using phase2tkutil::book1DFromPS;

  phi_res_barrel_layers.assign(trklet::N_LAYER, nullptr);
  bend_res_barrel_layers.assign(trklet::N_LAYER, nullptr);
  phi_res_fw_endcap_discs.assign(trklet::N_DISK, nullptr);
  bend_res_fw_endcap_discs.assign(trklet::N_DISK, nullptr);
  phi_res_bw_endcap_discs.assign(trklet::N_DISK, nullptr);
  bend_res_bw_endcap_discs.assign(trklet::N_DISK, nullptr);

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

  // number of stubs histogram
  number_of_stubs = book1DFromPS(iBooker, "number_of_stubs", psNStubs, "# stubs", "events");

  iBooker.setCurrentFolder(topFolderName_ + "/Residual");
  // z residuals barrel (Summary)
  z_res_isPS_barrel = book1DFromPS(iBooker, "#Delta z Barrel PS modules", ps_PS_Res, "tp_z - stub_z", "events");
  z_res_is2S_barrel = book1DFromPS(iBooker, "#Delta z Barrel 2S modules", ps_2S_Res, "tp_z - stub_z [cm]", "events");

  // r residuals endcaps (Summary)
  r_res_isPS_endcap = book1DFromPS(iBooker, "#Delta r Endcaps PS modules", ps_PS_Res, "tp_r - stub_r [cm]", "events");
  r_res_is2S_endcap = book1DFromPS(iBooker, "#Delta r Endcaps 2S modules", ps_2S_Res, "tp_r - stub_r [cm]", "events");

  // phi residuals (PS / 2S, barrel vs endcaps) (Summary)
  phi_res_isPS_barrel =
      book1DFromPS(iBooker, "#Delta #phi Barrel PS modules", psPhi_Res, "tp_phi - stub_phi", "events");
  phi_res_is2S_barrel =
      book1DFromPS(iBooker, "#Delta #phi Barrel 2S modules", psPhi_Res, "tp_phi - stub_phi", "events");
  phi_res_isPS_endcap =
      book1DFromPS(iBooker, "#Delta #phi Endcaps PS modules", psPhi_Res, "tp_phi - stub_phi", "events");
  phi_res_is2S_endcap =
      book1DFromPS(iBooker, "#Delta #phi Endcaps 2S modules", psPhi_Res, "tp_phi - stub_phi", "events");

  // bend residuals (PS / 2S, barrel vs endcaps) (Summary)
  bend_res_isPS_barrel =
      book1DFromPS(iBooker, "#Delta bend Barrel PS modules", psBend_Res, "tp_bend - stub_bend", "events");
  bend_res_is2S_barrel =
      book1DFromPS(iBooker, "#Delta bend Barrel 2S modules", psBend_Res, "tp_bend - stub_bend", "events");
  bend_res_isPS_endcap =
      book1DFromPS(iBooker, "#Delta bend Endcaps PS modules", psBend_Res, "tp_bend - stub_bend", "events");
  bend_res_is2S_endcap =
      book1DFromPS(iBooker, "#Delta bend Endcaps 2S modules", psBend_Res, "tp_bend - stub_bend", "events");

  iBooker.setCurrentFolder(topFolderName_ + "/Residual/Detailed");

  // Detailed Endcap R-Residuals (Split by FW/BW)
  r_res_isPS_fw_endcap =
      book1DFromPS(iBooker, "#Delta r FW Endcap PS modules", ps_PS_Res, "tp_r - stub_r [cm]", "events");
  r_res_is2S_fw_endcap =
      book1DFromPS(iBooker, "#Delta r FW Endcap 2S modules", ps_2S_Res, "tp_r - stub_r [cm]", "events");
  r_res_isPS_bw_endcap =
      book1DFromPS(iBooker, "#Delta r BW Endcap PS modules", ps_PS_Res, "tp_r - stub_r [cm]", "events");
  r_res_is2S_bw_endcap =
      book1DFromPS(iBooker, "#Delta r BW Endcap 2S modules", ps_2S_Res, "tp_r - stub_r [cm]", "events");

  // Detailed Endcap Phi/Bend (Split by FW/BW)
  phi_res_fw_endcap = book1DFromPS(iBooker, "#Delta #phi FW Endcap", psPhi_Res, "tp_phi - stub_phi", "events");
  phi_res_bw_endcap = book1DFromPS(iBooker, "#Delta #phi BW Endcap", psPhi_Res, "tp_phi - stub_phi", "events");
  bend_res_fw_endcap = book1DFromPS(iBooker, "#Delta bend FW Endcap", psBend_Res, "tp_bend - stub_bend", "events");
  bend_res_bw_endcap = book1DFromPS(iBooker, "#Delta bend BW Endcap", psBend_Res, "tp_bend - stub_bend", "events");

  // Barrel Layers (Per Layer)
  for (int i = 0; i < trklet::N_LAYER; ++i) {
    std::string layerParams = "L" + std::to_string(i + 1);
    phi_res_barrel_layers[i] =
        book1DFromPS(iBooker, "#Delta #phi Barrel " + layerParams, psPhi_Res, "tp_phi - stub_phi", "events");
    bend_res_barrel_layers[i] =
        book1DFromPS(iBooker, "#Delta bend Barrel " + layerParams, psBend_Res, "tp_bend - stub_bend", "events");
  }

  // Endcap Disks (Per Disk, Split FW/BW)
  for (int i = 0; i < trklet::N_DISK; ++i) {
    std::string diskParams = "D" + std::to_string(i + 1);
    phi_res_fw_endcap_discs[i] =
        book1DFromPS(iBooker, "#Delta #phi FW Endcap " + diskParams, psPhi_Res, "tp_phi - stub_phi", "events");
    bend_res_fw_endcap_discs[i] =
        book1DFromPS(iBooker, "#Delta bend FW Endcap " + diskParams, psBend_Res, "tp_bend - stub_bend", "events");
    phi_res_bw_endcap_discs[i] =
        book1DFromPS(iBooker, "#Delta #phi BW Endcap " + diskParams, psPhi_Res, "tp_phi - stub_phi", "events");
    bend_res_bw_endcap_discs[i] =
        book1DFromPS(iBooker, "#Delta bend BW Endcap " + diskParams, psBend_Res, "tp_bend - stub_bend", "events");
  }

  // 1D plots for stub efficiency vs pT
  iBooker.setCurrentFolder(topFolderName_ + "/EfficiencyIngredients");

  gen_clusters_barrel = book1DFromPS(iBooker, "gen_clusters_barrel", psEffic_pt, "p_{T} [GeV]", "# tracking particles");
  gen_clusters_if_stub_barrel =
      book1DFromPS(iBooker, "gen_clusters_if_stub_barrel", psEffic_pt, "p_{T} [GeV]", "# tracking particles");
  gen_clusters_endcaps =
      book1DFromPS(iBooker, "gen_clusters_endcaps", psEffic_pt, "p_{T} [GeV]", "# tracking particles");
  gen_clusters_if_stub_endcaps =
      book1DFromPS(iBooker, "gen_clusters_if_stub_endcaps", psEffic_pt, "p_{T} [GeV]", "# tracking particles");

  gen_clusters_zoom_barrel =
      book1DFromPS(iBooker, "gen_clusters_zoom_barrel", psEffic_pt_zoom, "p_{T} [GeV]", "# tracking particles");
  gen_clusters_if_stub_zoom_barrel =
      book1DFromPS(iBooker, "gen_clusters_if_stub_zoom_barrel", psEffic_pt_zoom, "p_{T} [GeV]", "# tracking particles");
  gen_clusters_zoom_endcaps =
      book1DFromPS(iBooker, "gen_clusters_zoom_endcaps", psEffic_pt_zoom, "p_{T} [GeV]", "# tracking particles");
  gen_clusters_if_stub_zoom_endcaps = book1DFromPS(
      iBooker, "gen_clusters_if_stub_zoom_endcaps", psEffic_pt_zoom, "p_{T} [GeV]", "# tracking particles");
}

void Phase2OTValidateStub::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // Phase2OTValidateStub
  edm::ParameterSetDescription desc;
  // Helper for 1D histograms
  auto addHist = [&desc](const std::string& name, int bins, double xmin, double xmax) {
    edm::ParameterSetDescription psd;
    psd.add<int>("Nbinsx", bins);
    psd.add<double>("xmin", xmin);
    psd.add<double>("xmax", xmax);
    desc.add<edm::ParameterSetDescription>(name, psd);
  };

  // Helper for 2D histograms
  auto addHist2D =
      [&desc](const std::string& name, int binsX, double xmin, double xmax, int binsY, double ymin, double ymax) {
        edm::ParameterSetDescription psd;
        psd.add<int>("Nbinsx", binsX);
        psd.add<double>("xmin", xmin);
        psd.add<double>("xmax", xmax);
        psd.add<int>("Nbinsy", binsY);
        psd.add<double>("ymin", ymin);
        psd.add<double>("ymax", ymax);
        desc.add<edm::ParameterSetDescription>(name, psd);
      };

  // 2D Histograms
  addHist2D("TH2TTStub_RZ", 900, -300.0, 300.0, 900, 0.0, 120.0);

  // 1D Histograms
  addHist("TH1NStubs", 100, 0.0, 25000.0);
  addHist("TH1_2S_Res", 99, -5.5, 5.5);
  addHist("TH1_PS_Res", 99, -2.0, 2.0);
  addHist("TH1Phi_Res", 100, -0.025, 0.025);
  addHist("TH1Bend_Res", 59, -5.5, 5.0);
  addHist("TH1Effic_pt", 50, 0.0, 100.0);
  addHist("TH1Effic_pt_zoom", 50, 0.0, 10.0);

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

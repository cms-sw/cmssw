// -*- C++ -*-
//
// Package:    Validation/SiTrackerPhase2V
// Class:      Phase2OTValidateTTStub

/**
 * This class is part of the Phase 2 Tracker validation framework. It validates the
 * association of tracking particles to stubs and evaluates stub reconstruction performance
 * by generating detailed histograms, including residuals for key parameters.
 * 
 * Usage:
 * To generate histograms from this code, run the test configuration files provided
 * in the DQM/SiTrackerPhase2/test directory. The generated histograms can then be
 * analyzed or visualized to study stub performance.
 */

// Original Author:

// Updated by: Brandi Skipworth, 2025

// system include files
#include <memory>
#include <numeric>
#include <vector>

// user include files
#include "SimDataFormats/Associations/interface/TTStubAssociationMap.h"

#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

class Phase2OTValidateTTStub : public DQMEDAnalyzer {
public:
  explicit Phase2OTValidateTTStub(const edm::ParameterSet&);
  ~Phase2OTValidateTTStub() override;
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
  std::vector<MonitorElement*> bend_res_barrel_layers;
  std::vector<MonitorElement*> bend_res_fw_endcap_discs;
  std::vector<MonitorElement*> bend_res_bw_endcap_discs;

  std::vector<MonitorElement*>* phi_res_vec = nullptr;
  std::vector<MonitorElement*>* bend_res_vec = nullptr;

private:
  edm::ParameterSet conf_;
  edm::EDGetTokenT<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> tagTTStubsToken_;
  edm::EDGetTokenT<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> ttStubMCTruthToken_;
  std::string topFolderName_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const TrackerGeometry* tkGeom_ = nullptr;
  const TrackerTopology* tTopo_ = nullptr;
  double TP_minPt;
  double TP_maxEta;
  double TP_maxVtxZ;
  double TP_maxD0;
  double TP_maxDxy;
};

// constructors and destructor
Phase2OTValidateTTStub::Phase2OTValidateTTStub(const edm::ParameterSet& iConfig)
    : conf_(iConfig),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()) {
  // now do what ever initialization is needed
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  tagTTStubsToken_ =
      consumes<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>(conf_.getParameter<edm::InputTag>("TTStubs"));
  ttStubMCTruthToken_ =
      consumes<TTStubAssociationMap<Ref_Phase2TrackerDigi_>>(conf_.getParameter<edm::InputTag>("MCTruthStubInputTag"));
  TP_minPt = conf_.getParameter<double>("TP_minPt");
  TP_maxEta = conf_.getParameter<double>("TP_maxEta");
  TP_maxVtxZ = conf_.getParameter<double>("TP_maxVtxZ");
  TP_maxD0 = conf_.getParameter<double>("TP_maxD0");
  TP_maxDxy = conf_.getParameter<double>("TP_maxDxy");
}

Phase2OTValidateTTStub::~Phase2OTValidateTTStub() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void Phase2OTValidateTTStub::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
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
  phi_res_barrel_layers.resize(6, nullptr);
  bend_res_barrel_layers.resize(6, nullptr);
  phi_res_fw_endcap_discs.resize(5, nullptr);
  bend_res_fw_endcap_discs.resize(5, nullptr);
  phi_res_bw_endcap_discs.resize(5, nullptr);
  bend_res_bw_endcap_discs.resize(5, nullptr);
}
// member functions

// Calculate the correction factor for tilted modules in barrel
float Phase2OTValidateTTStub::phiOverBendCorrection(bool isBarrel,
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
    float deltaZ = (R1 - R0 > 0)
                       ? (Z1 - Z0)
                       : -(Z1 - Z0);  // if module parallel, tilt angle should be π/2 and deltaZ would approach zero
    // fill histograms here
    tiltAngle = atan(deltaR / std::abs(deltaZ));
  }

  float correction;
  if (isBarrel && tTopo->tobSide(detid) != 3) {  // Assuming this condition represents tiltedBarrel
    correction = cos(tiltAngle) * std::abs(stub_z) / stub_r + sin(tiltAngle);
  } else if (isBarrel) {
    correction = 1;
  } else {
    correction =
        std::abs(stub_z) /
        stub_r;  // if tiltAngle = 0, stub (not module) is parallel to the beam line, if tiltAngle = 90, stub is perpendicular to beamline
  }

  return correction;
}

// Compute derived coordinates (z, phi, r) for tracking particle (TP)
std::vector<double> Phase2OTValidateTTStub::getTPDerivedCoords(edm::Ptr<TrackingParticle> associatedTP,
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
    tp_phi = reco::reduceRange(tp_phi);
    tp_z = tp_z0 + (2.0E2 / c_) * tp_t * (1 / tp_rinv) * std::asin(tp_r * tp_rinv * c_ / 2.0E2);
  } else {
    tp_z = stub_z;
    tp_phi = associatedTP->p4().phi() - (tp_z - tp_z0) * tp_rinv * c_ / 2.0E2 / tp_t;
    tp_phi = reco::reduceRange(tp_phi);
    tp_r = 2.0E2 / tp_rinv / c_ * std::sin((tp_z - tp_z0) * tp_rinv * c_ / 2.0E2 / tp_t);
  }

  std::vector<double> tpDerived_coords{tp_z, tp_phi, tp_r};
  return tpDerived_coords;
}

// ------------ method called for each event  ------------
void Phase2OTValidateTTStub::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  /// Track Trigger Stubs
  edm::Handle<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> Phase2TrackerDigiTTStubHandle;
  iEvent.getByToken(tagTTStubsToken_, Phase2TrackerDigiTTStubHandle);
  edm::Handle<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTStubHandle;
  iEvent.getByToken(ttStubMCTruthToken_, MCTruthTTStubHandle);

  trklet::Settings settings;
  double bfield_ = settings.bfield();
  double c_ = settings.c();

  /// Loop over input Stubs for basic histogram filling (e.g., Stub_RZ)
  typename edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>::const_iterator inputIter;
  typename edmNew::DetSet<TTStub<Ref_Phase2TrackerDigi_>>::const_iterator contentIter;
  // Adding protection
  if (!Phase2TrackerDigiTTStubHandle.isValid() || !MCTruthTTStubHandle.isValid()) {
    edm::LogError("Phase2OTValidateTTStub") << "Invalid handle(s) detected.";
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
      //double rawBend = tempStubRef->rawBend();
      //double bendOffset = tempStubRef->bendOffset();

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
      edm::LogError("Phase2OTValidateTTStub") << "Error: det0 or det1 is null";
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
      edm::LogError("Phase2OTValidateTTStub") << "Error: theGeomDetUnit is null";
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
        edm::LogError("Phase2OTValidateTTStub") << "Error: theGeomDet is null";
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

      bool isPSmodule = tkGeom_->getDetectorType(detid) == TrackerGeometry::ModuleType::Ph2PSP;

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
      float tp_dxy = std::sqrt(tp_vx * tp_vx + tp_vy * tp_vy);

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
      if (std::abs(tp_dxy) > TP_maxDxy)
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

      // Histograms for z_res, phi_res, and r_res based on module type and location
      if (isBarrel == 1) {
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
      if (layer >= 1 && layer <= 6) {
        (*bend_res_vec)[layer - 1]->Fill(bendRes);
        (*phi_res_vec)[layer - 1]->Fill(phiRes);
      }
    }
  }
}  // end of method

// ------------ method called when starting to processes a run  ------------
void Phase2OTValidateTTStub::bookHistograms(DQMStore::IBooker& iBooker,
                                            edm::Run const& run,
                                            edm::EventSetup const& es) {
  edm::ParameterSet psTTStub_RZ = conf_.getParameter<edm::ParameterSet>("TH2TTStub_RZ");
  edm::ParameterSet ps_2S_Res = conf_.getParameter<edm::ParameterSet>("TH1_2S_Res");
  edm::ParameterSet ps_PS_Res = conf_.getParameter<edm::ParameterSet>("TH1_PS_Res");
  edm::ParameterSet psPhi_Res = conf_.getParameter<edm::ParameterSet>("TH1Phi_Res");
  edm::ParameterSet psBend_Res = conf_.getParameter<edm::ParameterSet>("TH1Bend_Res");
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

  // barrel layers
  for (int i = 0; i < 6; ++i) {
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
  for (int i = 0; i < 5; ++i) {
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
}

void Phase2OTValidateTTStub::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // Phase2OTValidateTTStub
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

  desc.add<std::string>("TopFolderName", "TrackerPhase2OTStubV");
  desc.add<edm::InputTag>("TTStubs", edm::InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"));
  desc.add<edm::InputTag>("trackingParticleToken", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("MCTruthStubInputTag", edm::InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"));
  desc.add<edm::InputTag>("MCTruthClusterInputTag",
                          edm::InputTag("TTClusterAssociatorFromPixelDigis", "ClusterInclusive"));
  desc.add<int>("TP_minNStub", 4);
  desc.add<int>("TP_minNLayersStub", 4);
  desc.add<double>("TP_minPt", 2.0);
  desc.add<double>("TP_maxEta", 2.4);
  desc.add<double>("TP_maxVtxZ", 15.0);
  desc.add<double>("TP_maxD0", 1.0);
  desc.add<double>("TP_maxDxy", 1.0);
  descriptions.add("Phase2OTValidateTTStub", desc);
  // or use the following to generate the label from the module's C++ type
  //descriptions.addWithDefaultLabel(desc);
}
DEFINE_FWK_MODULE(Phase2OTValidateTTStub);

// -*- C++ -*-
//
/**\class SiOuterTracker Phase2OTValidateTTStub.cc
 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:
//

// system include files
#include <memory>
#include <numeric>
#include <vector>
#include <iostream> // Include for printing debug statements

// user include files
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"

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
  explicit Phase2OTValidateTTStub(const edm::ParameterSet &);
  ~Phase2OTValidateTTStub() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  float phiOverBendCorrection(bool isBarrel, float stub_z, float stub_r, const TrackerTopology* tTopo, uint32_t detid, const GeomDetUnit* det0, const GeomDetUnit* det1);
  std::vector<double> getTPDerivedCoords(edm::Ptr<TrackingParticle> my_tp, bool isBarrel, double modMaxZ, double modMinZ, float stub_r) const;

  // TTStub stacks
  // Global position of the stubs
  MonitorElement *Stub_RZ = nullptr;  // TTStub #rho vs. z

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

  std::vector<MonitorElement*>* phi_res_discs = nullptr;
  std::vector<MonitorElement*>* bend_res_discs = nullptr;

private:
  edm::ParameterSet conf_;
  edm::EDGetTokenT<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> tagTTStubsToken_;
  edm::EDGetTokenT<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> ttStubMCTruthToken_;  // MC truth association map for stubs
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  std::string topFolderName_;
  const TrackerGeometry *theTrackerGeom_ = nullptr;
  const TrackerTopology *tTopo_ = nullptr;
  double TP_minPt;
  double TP_maxEta;
};

// constructors and destructor
Phase2OTValidateTTStub::Phase2OTValidateTTStub(const edm::ParameterSet &iConfig)
    : conf_(iConfig),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      tTopoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()) {
  // now do whatever initialization is needed
  ttStubMCTruthToken_ =
      consumes<TTStubAssociationMap<Ref_Phase2TrackerDigi_>>(conf_.getParameter<edm::InputTag>("MCTruthStubInputTag"));
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  tagTTStubsToken_ =
      consumes<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>(conf_.getParameter<edm::InputTag>("TTStubs"));
  TP_minPt = conf_.getParameter<double>("TP_minPt");      // min pT to consider matching
  TP_maxEta = conf_.getParameter<double>("TP_maxEta");    // max eta to consider matching
}

Phase2OTValidateTTStub::~Phase2OTValidateTTStub() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void Phase2OTValidateTTStub::dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) {
  theTrackerGeom_ = &(iSetup.getData(geomToken_));
  tTopo_ = &(iSetup.getData(tTopoToken_));

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

float Phase2OTValidateTTStub::phiOverBendCorrection(bool isBarrel, float stub_z, float stub_r, const TrackerTopology* tTopo, uint32_t detid, const GeomDetUnit* det0, const GeomDetUnit* det1) {
    // Get R0, R1, Z0, Z1 values
    float R0 = det0->position().perp();
    float R1 = det1->position().perp();
    float Z0 = det0->position().z();
    float Z1 = det1->position().z();

    bool isTiltedBarrel = (isBarrel && tTopo->tobSide(detid) != 3);
    
    float tiltAngle = 0; // Initialize to 0 (meaning no tilt, in the endcaps)
    if (isTiltedBarrel) {
        float deltaR = std::abs(R1 - R0);
        float deltaZ = (R1 - R0 > 0) ? (Z1 - Z0) : -(Z1 - Z0); // if module parallel, tilt angle should be π/2 and deltaZ would approach zero
        // fill histograms here
        tiltAngle = atan(deltaR / std::abs(deltaZ));
    }

    float correction;
    if (isBarrel && tTopo->tobSide(detid) != 3) {  // Assuming this condition represents tiltedBarrel
        correction = cos(tiltAngle) * std::abs(stub_z) / stub_r + sin(tiltAngle);
    } else if (isBarrel) {
        correction = 1;
    } else {
        correction = std::abs(stub_z) / stub_r; // if tiltAngle = 0, stub (not module) is parallel to the beam line, if tiltAngle = 90, stub is perpendicular to beamline
    }

    return correction;
}

std::vector<double> Phase2OTValidateTTStub::getTPDerivedCoords(edm::Ptr<TrackingParticle> my_tp, bool isBarrel, double modMaxZ, double modMinZ, float stub_r) const {
  double tp_phi = -99;
  double tp_r = -99;
  double tp_z = -99;
  
  trklet::Settings settings;
  double bfield_ = settings.bfield();
  double c_ = settings.c();
  
  // Get values from the tracking particle my_tp
  double myTP_pt = my_tp->pt();
  double myTP_charge = my_tp->charge();
  float myTP_z0 = my_tp->vertex().z();
  double myTP_t = my_tp->tanl();
  double myTP_rinv = (myTP_charge * bfield_) / (myTP_pt);

  if (isBarrel) { 
      tp_r = stub_r;
      tp_phi = my_tp->p4().phi() - std::asin(tp_r * myTP_rinv * c_ / 2.0E2);
      tp_phi = reco::reduceRange(tp_phi);  
      tp_z = myTP_z0 + (2.0E2 / c_) * myTP_t * (1 / myTP_rinv) * std::asin(tp_r * myTP_rinv * c_ / 2.0E2);
  } else {
      tp_z = (modMaxZ + modMinZ) / 2;
      tp_phi = my_tp->p4().phi() - (tp_z - myTP_z0) * myTP_rinv * c_ / 2.0E2 / myTP_t; 
      tp_phi = reco::reduceRange(tp_phi);
  }

  std::vector<double> tpDerived_coords{tp_z, tp_phi};
  return tpDerived_coords;
}

// ------------ method called for each event  ------------
void Phase2OTValidateTTStub::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // Handle to Track Trigger Stubs
  edm::Handle<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> Phase2TrackerDigiTTStubHandle;
  iEvent.getByToken(tagTTStubsToken_, Phase2TrackerDigiTTStubHandle);
  edm::Handle<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTStubHandle;
  iEvent.getByToken(ttStubMCTruthToken_, MCTruthTTStubHandle);

  trklet::Settings settings;
  double bfield_ = settings.bfield();
  double c_ = settings.c();

  // Ensure valid handles
  if (!Phase2TrackerDigiTTStubHandle.isValid() || !MCTruthTTStubHandle.isValid()) {
    edm::LogError("Phase2OTValidateTTStub") << "Invalid handle(s) detected.";
    return;
  }

  // Loop over geometric detectors
  for (auto gd = theTrackerGeom_->dets().begin(); gd != theTrackerGeom_->dets().end(); gd++) {
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
    const GeomDetUnit* det0 = theTrackerGeom_->idToDetUnit(detid);
    const GeomDetUnit* det1 = theTrackerGeom_->idToDetUnit(tTopo_->partnerDetId(detid));
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
      edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>> tempStubPtr = edmNew::makeRefTo(Phase2TrackerDigiTTStubHandle, stubIter);

      // Check if the stub is genuine
      if (!MCTruthTTStubHandle->isGenuine(tempStubPtr))
        continue;

      // Get det ID from the stub
      DetId detIdStub = theTrackerGeom_->idToDet((tempStubPtr->clusterRef(0))->getDetId())->geographicalId();

      // Retrieve geometrical detector
      const GeomDet* theGeomDet = theTrackerGeom_->idToDet(detIdStub);
      if (!theGeomDet) {
        edm::LogError("Phase2OTValidateTTStub") << "Error: theGeomDet is null";
        continue;
      }

      // Process tracking particle information
      edm::Ptr<TrackingParticle> my_tp = MCTruthTTStubHandle->findTrackingParticlePtr(tempStubPtr);
      if (my_tp.isNull())
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
      Global3DPoint innerClusterGlobalPos = theGeomDet->surface().toGlobal(theGeomDet->topology().localPosition(innerClusterCoords));
      Global3DPoint outerClusterGlobalPos = theGeomDet->surface().toGlobal(theGeomDet->topology().localPosition(outerClusterCoords));

      // Determine maximum Z positions of the stubs
      float stub_maxZ = std::max(innerClusterGlobalPos.z(), outerClusterGlobalPos.z());

      // Stub parameters
      float stub_phi = innerClusterGlobalPos.phi();
      float stub_r = innerClusterGlobalPos.perp();
      float stub_z = innerClusterGlobalPos.z();

      // Tracking particle parameters
      int myTP_charge = my_tp->charge();
      float myTP_pt = my_tp->p4().pt();
      float myTP_eta = my_tp->p4().eta();

      if (myTP_charge == 0) continue;
      if (myTP_pt < TP_minPt) continue;
      if (std::abs(myTP_eta) > TP_maxEta) continue;

      // Derived coordinates
      std::vector<double> tpDerivedCoords = getTPDerivedCoords(my_tp, isBarrel, modMaxZ, modMinZ, stub_r);
      float tp_z = tpDerivedCoords[0];
      float tp_phi = tpDerivedCoords[1];
      
      // Trigger information
      float trigBend = tempStubPtr->bendFE();
      if (!isBarrel && stub_maxZ < 0.0) {
        trigBend = -trigBend;
      }

      float correctionValue = phiOverBendCorrection(isBarrel, stub_z, stub_r, tTopo_, detid, det0, det1);
      float trackBend = -(sensorSpacing * stub_r * bfield_ * c_ * myTP_charge) /
                        (stripPitch * 2.0E2 * myTP_pt * correctionValue);

      float bendRes = trackBend - trigBend;
      float zRes = tp_z - stub_z;
      float phiRes = tp_phi - stub_phi;
      
      // Fill histograms
      if (Stub_RZ) {
        Stub_RZ->Fill(stub_z, stub_r);
      } else {
        edm::LogError("Phase2OTValidateTTStub") << "Error: Stub_RZ histogram is null";
      }
      // Histograms for z_res and r_res based on module type and location
      if (isBarrel == 1) {
          if (isPSmodule) {
              z_res_isPS_barrel->Fill(zRes);
              phi_res_isPS_barrel->Fill(phiRes);
          } else {
              z_res_is2S_barrel->Fill(zRes);
              phi_res_is2S_barrel->Fill(phiRes);
          }
      }

      // Ensure that the vectors are correctly assigned before use
      if (isBarrel == 1) {
          phi_res_discs = &phi_res_barrel_layers; // Pointing to barrel layers vector
          bend_res_discs = &bend_res_barrel_layers; // Pointing to barrel layers vector
      } else {
          if (stub_maxZ > 0) {
              // Forward endcap
              bend_res_fw_endcap->Fill(bendRes);
              phi_res_fw_endcap->Fill(phiRes);
              phi_res_discs = &phi_res_fw_endcap_discs; // Pointing to forward endcap vector
              bend_res_discs = &bend_res_fw_endcap_discs; // Pointing to forward endcap vector
          } else {
              // Backward endcap
              bend_res_bw_endcap->Fill(bendRes);
              phi_res_bw_endcap->Fill(phiRes);
              phi_res_discs = &phi_res_bw_endcap_discs; // Pointing to backward endcap vector
              bend_res_discs = &bend_res_bw_endcap_discs; // Pointing to backward endcap vector
          }
      }

      // Fill the appropriate histogram based on layer/disc
      if (layer >= 1 && layer <= 6) {
          (*bend_res_discs)[layer - 1]->Fill(bendRes);
          (*phi_res_discs)[layer - 1]->Fill(phiRes);
      }
    }
  }
} // end of method

// ------------ method called when starting to processes a run  ------------
void Phase2OTValidateTTStub::bookHistograms(DQMStore::IBooker &iBooker,
                                            edm::Run const &run,
                                            edm::EventSetup const &es) {
  std::string HistoName;
  iBooker.setCurrentFolder(topFolderName_);
  edm::ParameterSet psTTStub_RZ = conf_.getParameter<edm::ParameterSet>("TH2TTStub_RZ");
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
  // stub vs tp z-coord diff
  edm::ParameterSet psZ_Res = conf_.getParameter<edm::ParameterSet>("TH1z_Res");

  // z-res for PS modules
  HistoName = "#Delta z Barrel PS modules";
  z_res_isPS_barrel = iBooker.book1D(HistoName,
                            HistoName,
                            psZ_Res.getParameter<int32_t>("Nbinsx"),
                            psZ_Res.getParameter<double>("xmin"),
                            psZ_Res.getParameter<double>("xmax"));
  z_res_isPS_barrel->setAxisTitle("tp_z - stub_z", 1);
  z_res_isPS_barrel->setAxisTitle("events ", 2);

  // z-res for 2S modules
  HistoName = "#Delta z Barrel 2S modules";
  z_res_is2S_barrel = iBooker.book1D(HistoName,
                            HistoName,
                            psZ_Res.getParameter<int32_t>("Nbinsx"),
                            psZ_Res.getParameter<double>("xmin"),
                            psZ_Res.getParameter<double>("xmax"));
  z_res_is2S_barrel->setAxisTitle("tp_z - stub_z [cm]", 1);
  z_res_is2S_barrel->setAxisTitle("events ", 2);

  edm::ParameterSet psPhi_Res = conf_.getParameter<edm::ParameterSet>("TH1Phi_Res");
  edm::ParameterSet psBend_Res = conf_.getParameter<edm::ParameterSet>("TH1Bend_Res");

  // Booking for barrel layers
  for (int i = 0; i < 6; ++i) {
    std::string HistoName = "#Delta #phi barrel L" + std::to_string(i+1);
    phi_res_barrel_layers[i] = iBooker.book1D(HistoName, HistoName, 
                                              psPhi_Res.getParameter<int32_t>("Nbinsx"),
                                              psPhi_Res.getParameter<double>("xmin"),
                                              psPhi_Res.getParameter<double>("xmax"));
    HistoName = "#Delta bend barrel L" + std::to_string(i+1);
    bend_res_barrel_layers[i] = iBooker.book1D(HistoName, HistoName,
                                               psBend_Res.getParameter<int32_t>("Nbinsx"),
                                               psBend_Res.getParameter<double>("xmin"),
                                               psBend_Res.getParameter<double>("xmax"));
  }

  // Booking for endcap discs
  for (int i = 0; i < 5; ++i) {
    std::string HistoName = "#Delta #phi fw endcap D" + std::to_string(i+1);
    phi_res_fw_endcap_discs[i] = iBooker.book1D(HistoName, HistoName,
                                                psPhi_Res.getParameter<int32_t>("Nbinsx"),
                                                psPhi_Res.getParameter<double>("xmin"),
                                                psPhi_Res.getParameter<double>("xmax"));
    HistoName = "#Delta bend fw endcap D" + std::to_string(i+1);
    bend_res_fw_endcap_discs[i] = iBooker.book1D(HistoName, HistoName,
                                                 psBend_Res.getParameter<int32_t>("Nbinsx"),
                                                 psBend_Res.getParameter<double>("xmin"),
                                                 psBend_Res.getParameter<double>("xmax"));

    HistoName = "#Delta #phi bw endcap D" + std::to_string(i+1);
    phi_res_bw_endcap_discs[i] = iBooker.book1D(HistoName, HistoName,
                                                psPhi_Res.getParameter<int32_t>("Nbinsx"),
                                                psPhi_Res.getParameter<double>("xmin"),
                                                psPhi_Res.getParameter<double>("xmax"));
    HistoName = "#Delta bend bw endcap D" + std::to_string(i+1);
    bend_res_bw_endcap_discs[i] = iBooker.book1D(HistoName, HistoName,
                                                 psBend_Res.getParameter<int32_t>("Nbinsx"),
                                                 psBend_Res.getParameter<double>("xmin"),
                                                 psBend_Res.getParameter<double>("xmax"));
  }

  // Book the histograms for phi_res and bend_res
  phi_res_isPS_barrel = iBooker.book1D("phi_res_isPS_barrel", "Phi Resolution for PS Modules in Barrel",
                                       psPhi_Res.getParameter<int32_t>("Nbinsx"),
                                       psPhi_Res.getParameter<double>("xmin"),
                                       psPhi_Res.getParameter<double>("xmax"));
  
  phi_res_is2S_barrel = iBooker.book1D("phi_res_is2S_barrel", "Phi Resolution for 2S Modules in Barrel",
                                       psPhi_Res.getParameter<int32_t>("Nbinsx"),
                                       psPhi_Res.getParameter<double>("xmin"),
                                       psPhi_Res.getParameter<double>("xmax"));

  phi_res_fw_endcap = iBooker.book1D("phi_res_fw_endcap", "Phi Resolution for Forward Endcap",
                                     psPhi_Res.getParameter<int32_t>("Nbinsx"),
                                     psPhi_Res.getParameter<double>("xmin"),
                                     psPhi_Res.getParameter<double>("xmax"));

  phi_res_bw_endcap = iBooker.book1D("phi_res_bw_endcap", "Phi Resolution for Backward Endcap",
                                     psPhi_Res.getParameter<int32_t>("Nbinsx"),
                                     psPhi_Res.getParameter<double>("xmin"),
                                     psPhi_Res.getParameter<double>("xmax"));

  bend_res_fw_endcap = iBooker.book1D("bend_res_fw_endcap", "Bend Resolution for Forward Endcap",
                                      psBend_Res.getParameter<int32_t>("Nbinsx"),
                                      psBend_Res.getParameter<double>("xmin"),
                                      psBend_Res.getParameter<double>("xmax"));

  bend_res_bw_endcap = iBooker.book1D("bend_res_bw_endcap", "Bend Resolution for Backward Endcap",
                                      psBend_Res.getParameter<int32_t>("Nbinsx"),
                                      psBend_Res.getParameter<double>("xmin"),
                                      psBend_Res.getParameter<double>("xmax"));
}

void Phase2OTValidateTTStub::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
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
    desc.add<edm::ParameterSetDescription>("TH1z_Res", psd0);
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
  desc.add<edm::InputTag>("MCTruthClusterInputTag", edm::InputTag("TTClusterAssociatorFromPixelDigis", "ClusterInclusive"));
  desc.add<int>("TP_minNStub", 4);
  desc.add<int>("TP_minNLayersStub", 4);
  desc.add<double>("TP_minPt", 2.0);
  desc.add<double>("TP_maxEta", 2.4);
  desc.add<double>("TP_maxVtxZ", 15.0);
  descriptions.add("Phase2OTValidateTTStub", desc);
  // or use the following to generate the label from the module's C++ type
  //descriptions.addWithDefaultLabel(desc);
}
DEFINE_FWK_MODULE(Phase2OTValidateTTStub);
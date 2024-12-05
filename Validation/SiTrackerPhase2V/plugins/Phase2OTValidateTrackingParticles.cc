// Package:    SiOuterTrackerV
// Class:      SiOuterTrackerV

// Original Author:  Emily MacDonald

// system include files
#include <memory>
#include <numeric>
#include <vector>
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Associations/interface/TTClusterAssociationMap.h"
#include "SimDataFormats/Associations/interface/TTStubAssociationMap.h"
#include "SimDataFormats/Associations/interface/TTTrackAssociationMap.h"

class Phase2OTValidateTrackingParticles : public DQMEDAnalyzer {
public:
  explicit Phase2OTValidateTrackingParticles(const edm::ParameterSet &);
  ~Phase2OTValidateTrackingParticles() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  // Tracking particle distributions
  MonitorElement *trackParts_Eta = nullptr;
  MonitorElement *trackParts_Phi = nullptr;
  MonitorElement *trackParts_Pt = nullptr;

  // pT and eta for efficiency plots
  MonitorElement *tp_pt = nullptr;             // denominator
  MonitorElement *tp_pt_zoom = nullptr;        // denominator
  MonitorElement *tp_eta = nullptr;            // denominator
  MonitorElement *tp_d0 = nullptr;             // denominator
  MonitorElement *tp_VtxR = nullptr;           // denominator (also known as vxy)
  MonitorElement *tp_VtxZ = nullptr;           // denominator
  MonitorElement *match_tp_pt = nullptr;       // numerator
  MonitorElement *match_tp_pt_zoom = nullptr;  // numerator
  MonitorElement *match_tp_eta = nullptr;      // numerator
  MonitorElement *match_tp_d0 = nullptr;       // numerator
  MonitorElement *match_tp_VtxR = nullptr;     // numerator (also known as vxy)
  MonitorElement *match_tp_VtxZ = nullptr;     // numerator

  // stub efficiency plots
  MonitorElement *gen_clusters_barrel = nullptr;               // denominator
  MonitorElement *gen_clusters_zoom_barrel = nullptr;          // denominator
  MonitorElement *gen_clusters_if_stub_barrel = nullptr;       // numerator
  MonitorElement *gen_clusters_if_stub_zoom_barrel = nullptr;  // numerator

  // 1D intermediate resolution plots (pT and eta)
  MonitorElement *res_eta = nullptr;    // for all eta and pT
  MonitorElement *res_pt = nullptr;     // for all eta and pT
  MonitorElement *res_ptRel = nullptr;  // for all eta and pT (delta(pT)/pT)
  MonitorElement *respt_eta0to0p7_pt2to3 = nullptr;
  MonitorElement *respt_eta0p7to1_pt2to3 = nullptr;
  MonitorElement *respt_eta1to1p2_pt2to3 = nullptr;
  MonitorElement *respt_eta1p2to1p6_pt2to3 = nullptr;
  MonitorElement *respt_eta1p6to2_pt2to3 = nullptr;
  MonitorElement *respt_eta2to2p4_pt2to3 = nullptr;
  MonitorElement *respt_eta0to0p7_pt3to8 = nullptr;
  MonitorElement *respt_eta0p7to1_pt3to8 = nullptr;
  MonitorElement *respt_eta1to1p2_pt3to8 = nullptr;
  MonitorElement *respt_eta1p2to1p6_pt3to8 = nullptr;
  MonitorElement *respt_eta1p6to2_pt3to8 = nullptr;
  MonitorElement *respt_eta2to2p4_pt3to8 = nullptr;
  MonitorElement *respt_eta0to0p7_pt8toInf = nullptr;
  MonitorElement *respt_eta0p7to1_pt8toInf = nullptr;
  MonitorElement *respt_eta1to1p2_pt8toInf = nullptr;
  MonitorElement *respt_eta1p2to1p6_pt8toInf = nullptr;
  MonitorElement *respt_eta1p6to2_pt8toInf = nullptr;
  MonitorElement *respt_eta2to2p4_pt8toInf = nullptr;
  MonitorElement *reseta_eta0to0p7 = nullptr;
  MonitorElement *reseta_eta0p7to1 = nullptr;
  MonitorElement *reseta_eta1to1p2 = nullptr;
  MonitorElement *reseta_eta1p2to1p6 = nullptr;
  MonitorElement *reseta_eta1p6to2 = nullptr;
  MonitorElement *reseta_eta2to2p4 = nullptr;
  MonitorElement *resphi_eta0to0p7 = nullptr;
  MonitorElement *resphi_eta0p7to1 = nullptr;
  MonitorElement *resphi_eta1to1p2 = nullptr;
  MonitorElement *resphi_eta1p2to1p6 = nullptr;
  MonitorElement *resphi_eta1p6to2 = nullptr;
  MonitorElement *resphi_eta2to2p4 = nullptr;
  MonitorElement *resVtxZ_eta0to0p7 = nullptr;
  MonitorElement *resVtxZ_eta0p7to1 = nullptr;
  MonitorElement *resVtxZ_eta1to1p2 = nullptr;
  MonitorElement *resVtxZ_eta1p2to1p6 = nullptr;
  MonitorElement *resVtxZ_eta1p6to2 = nullptr;
  MonitorElement *resVtxZ_eta2to2p4 = nullptr;

  // For d0
  MonitorElement *resd0_eta0to0p7 = nullptr;
  MonitorElement *resd0_eta0p7to1 = nullptr;
  MonitorElement *resd0_eta1to1p2 = nullptr;
  MonitorElement *resd0_eta1p2to1p6 = nullptr;
  MonitorElement *resd0_eta1p6to2 = nullptr;
  MonitorElement *resd0_eta2to2p4 = nullptr;

private:
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> m_topoToken;
  edm::ParameterSet conf_;
  edm::EDGetTokenT<std::vector<TrackingParticle>> trackingParticleToken_;
  edm::EDGetTokenT<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>>
      ttClusterMCTruthToken_;  // MC truth association map for clusters
  edm::EDGetTokenT<TTStubAssociationMap<Ref_Phase2TrackerDigi_>>
      ttStubMCTruthToken_;  // MC truth association map for stubs
  edm::EDGetTokenT<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>>
      ttTrackMCTruthToken_;  // MC truth association map for tracks
  edm::EDGetTokenT<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> ttStubToken_;  // L1 Stub token
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> getTokenTrackerGeom_;     // Tracker geometry token
  int L1Tk_minNStub;
  double L1Tk_maxChi2dof;
  int TP_minNStub;
  int TP_minNLayersStub;
  double TP_minPt;
  double TP_maxEta;
  double TP_maxVtxZ;
  std::string topFolderName_;
};

//
// constructors and destructor
//
Phase2OTValidateTrackingParticles::Phase2OTValidateTrackingParticles(const edm::ParameterSet &iConfig)
    : m_topoToken(esConsumes()), conf_(iConfig) {
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  trackingParticleToken_ =
      consumes<std::vector<TrackingParticle>>(conf_.getParameter<edm::InputTag>("trackingParticleToken"));
  ttStubMCTruthToken_ =
      consumes<TTStubAssociationMap<Ref_Phase2TrackerDigi_>>(conf_.getParameter<edm::InputTag>("MCTruthStubInputTag"));
  ttClusterMCTruthToken_ = consumes<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>>(
      conf_.getParameter<edm::InputTag>("MCTruthClusterInputTag"));
  ttTrackMCTruthToken_ = consumes<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>>(
      conf_.getParameter<edm::InputTag>("MCTruthTrackInputTag"));
  ttStubToken_ = consumes<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>(
      edm::InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"));
  getTokenTrackerGeom_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
  L1Tk_minNStub = conf_.getParameter<int>("L1Tk_minNStub");         // min number of stubs in the track
  L1Tk_maxChi2dof = conf_.getParameter<double>("L1Tk_maxChi2dof");  // maximum chi2/dof of the track
  TP_minNStub = conf_.getParameter<int>("TP_minNStub");             // min number of stubs in the tracking particle to
  //min number of layers with stubs in the tracking particle to consider matching
  TP_minNLayersStub = conf_.getParameter<int>("TP_minNLayersStub");
  TP_minPt = conf_.getParameter<double>("TP_minPt");      // min pT to consider matching
  TP_maxEta = conf_.getParameter<double>("TP_maxEta");    // max eta to consider matching
  TP_maxVtxZ = conf_.getParameter<double>("TP_maxVtxZ");  // max vertZ (or z0) to consider matching
}

Phase2OTValidateTrackingParticles::~Phase2OTValidateTrackingParticles() = default;

// member functions

// ------------ method called for each event  ------------
void Phase2OTValidateTrackingParticles::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // Tracking Particles
  edm::Handle<std::vector<TrackingParticle>> trackingParticleHandle;
  iEvent.getByToken(trackingParticleToken_, trackingParticleHandle);

  // Truth Association Maps
  edm::Handle<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTTrackHandle;
  iEvent.getByToken(ttTrackMCTruthToken_, MCTruthTTTrackHandle);
  edm::Handle<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTClusterHandle;
  iEvent.getByToken(ttClusterMCTruthToken_, MCTruthTTClusterHandle);
  edm::Handle<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTStubHandle;
  iEvent.getByToken(ttStubMCTruthToken_, MCTruthTTStubHandle);
  edm::Handle<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> TTStubHandle;
  iEvent.getByToken(ttStubToken_, TTStubHandle);

  // Geometries
  const TrackerTopology *const tTopo = &iSetup.getData(m_topoToken);
  const TrackerGeometry *theTrackerGeom = &iSetup.getData(getTokenTrackerGeom_);

  // Loop over tracking particles
  int this_tp = 0;
  for (const auto &iterTP : *trackingParticleHandle) {
    edm::Ptr<TrackingParticle> tp_ptr(trackingParticleHandle, this_tp);
    this_tp++;

    // int tmp_eventid = iterTP.eventId().event();
    float tmp_tp_pt = iterTP.pt();
    float tmp_tp_phi = iterTP.phi();
    float tmp_tp_eta = iterTP.eta();

    //Calculate nLayers variable
    std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
        theStubRefs = MCTruthTTStubHandle->findTTStubRefs(tp_ptr);

    int hasStubInLayer[11] = {0};
    for (unsigned int is = 0; is < theStubRefs.size(); is++) {
      DetId detid(theStubRefs.at(is)->getDetId());
      int layer = -1;
      if (detid.subdetId() == StripSubdetector::TOB)
        layer = static_cast<int>(tTopo->layer(detid)) - 1;  //fill in array as entries 0-5
      else if (detid.subdetId() == StripSubdetector::TID)
        layer = static_cast<int>(tTopo->layer(detid)) + 5;  //fill in array as entries 6-10

      //treat genuine stubs separately (==2 is genuine, ==1 is not)
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
    // Fill the 1D distribution plots for tracking particles, to monitor change in stub definition
    if (tmp_tp_pt > TP_minPt && nStubLayerTP >= TP_minNLayersStub) {
      trackParts_Pt->Fill(tmp_tp_pt);
      trackParts_Eta->Fill(tmp_tp_eta);
      trackParts_Phi->Fill(tmp_tp_phi);
    }

    // if (TP_select_eventid == 0 && tmp_eventid != 0)
    //   continue;  //only care about tracking particles from the primary interaction for efficiency/resolution
    int nStubTP = -1;
    if (MCTruthTTStubHandle.isValid()) {
      std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
          theStubRefs = MCTruthTTStubHandle->findTTStubRefs(tp_ptr);
      nStubTP = (int)theStubRefs.size();
    }
    if (MCTruthTTClusterHandle.isValid() && MCTruthTTClusterHandle->findTTClusterRefs(tp_ptr).empty())
      continue;

    float tmp_tp_vz = iterTP.vz();
    float tmp_tp_vx = iterTP.vx();
    float tmp_tp_vy = iterTP.vy();
    float tmp_tp_charge = tp_ptr->charge();
    int tmp_tp_pdgid = iterTP.pdgId();

    // ----------------------------------------------------------------------------------------------
    // calculate d0 and VtxZ propagated back to the IP, pass if greater than max
    // VtxZ
    float tmp_tp_t = tan(2.0 * atan(1.0) - 2.0 * atan(exp(-tmp_tp_eta)));
    float delx = -tmp_tp_vx;
    float dely = -tmp_tp_vy;
    float K = 0.01 * 0.5696 / tmp_tp_pt * tmp_tp_charge;  // curvature correction
    float A = 1. / (2. * K);
    float tmp_tp_x0p = delx - A * sin(tmp_tp_phi);
    float tmp_tp_y0p = dely + A * cos(tmp_tp_phi);
    float tmp_tp_rp = sqrt(tmp_tp_x0p * tmp_tp_x0p + tmp_tp_y0p * tmp_tp_y0p);
    static double pi = 4.0 * atan(1.0);
    float delphi = tmp_tp_phi - atan2(-K * tmp_tp_x0p, K * tmp_tp_y0p);
    if (delphi < -pi)
      delphi += 2.0 * pi;
    if (delphi > pi)
      delphi -= 2.0 * pi;

    float tmp_tp_VtxZ = tmp_tp_vz + tmp_tp_t * delphi / (2.0 * K);
    float tmp_tp_VtxR = sqrt(tmp_tp_vx * tmp_tp_vx + tmp_tp_vy * tmp_tp_vy);
    float tmp_tp_d0 = tmp_tp_charge * tmp_tp_rp - (1. / (2. * K));

    // simpler formula for d0, in cases where the charge is zero:
    // https://github.com/cms-sw/cmssw/blob/master/DataFormats/TrackReco/interface/TrackBase.h
    float other_d0 = -tmp_tp_vx * sin(tmp_tp_phi) + tmp_tp_vy * cos(tmp_tp_phi);
    tmp_tp_d0 = tmp_tp_d0 * (-1);  // fix d0 sign
    if (K == 0) {
      tmp_tp_d0 = other_d0;
      tmp_tp_VtxZ = tmp_tp_vz;
    }
    if (std::fabs(tmp_tp_VtxZ) > TP_maxVtxZ)
      continue;

    // To make efficiency plots where the denominator has NO stub cuts
    if (tmp_tp_VtxR < 1.0) {
      tp_pt->Fill(tmp_tp_pt);  //pT effic, no cut on pT, but VtxR cut
      if (tmp_tp_pt <= 10)
        tp_pt_zoom->Fill(tmp_tp_pt);  //pT effic, no cut on pT, but VtxR cut
    }
    if (tmp_tp_pt < TP_minPt)
      continue;
    tp_VtxR->Fill(tmp_tp_VtxR);  // VtxR efficiency has no cut on VtxR
    if (tmp_tp_VtxR > 1.0)
      continue;
    tp_eta->Fill(tmp_tp_eta);
    tp_d0->Fill(tmp_tp_d0);
    tp_VtxZ->Fill(tmp_tp_VtxZ);

    if (nStubTP < TP_minNStub || nStubLayerTP < TP_minNLayersStub)
      continue;  //nStub cut not included in denominator of efficiency plots

    // Find all clusters that can be associated to a tracking particle with at least one hit
    std::vector<edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>, TTCluster<Ref_Phase2TrackerDigi_>>>
        associatedClusters = MCTruthTTClusterHandle->findTTClusterRefs(tp_ptr);

    // Address of tracking particle
    auto tp_ptr_address = tp_ptr.get();

    // Loop through associated clusters
    for (std::size_t k = 0; k < associatedClusters.size(); ++k) {
        auto clusA = associatedClusters[k];

        // Get cluster details
        DetId clusdetid = clusA->getDetId();
        if (clusdetid.subdetId() != StripSubdetector::TOB && clusdetid.subdetId() != StripSubdetector::TID)
            continue;

        bool isGenuine = MCTruthTTClusterHandle->isGenuine(clusA);
        if (!isGenuine)
            continue;

        // Suppress unused variable warnings
        DetId detidA = tTopo->stack(clusdetid);
        (void)detidA;

        const GeomDetUnit *detA = theTrackerGeom->idToDetUnit(clusdetid);
        const PixelGeomDetUnit *theGeomDetA = dynamic_cast<const PixelGeomDetUnit *>(detA);
        const PixelTopology *topoA = dynamic_cast<const PixelTopology *>(&(theGeomDetA->specificTopology()));
        GlobalPoint coordsA = theGeomDetA->surface().toGlobal(topoA->localPosition(clusA->findAverageLocalCoordinatesCentered()));
        (void)coordsA;

        // Retrieve TrackingParticles associated with the cluster
        const std::vector<TrackingParticlePtr> &theseTrackingParticles =
            MCTruthTTClusterHandle->findTrackingParticlePtrs(clusA);

        if (theseTrackingParticles.empty())
            continue;

        for (const auto &associatedTP : theseTrackingParticles) {
          if (!associatedTP)
            continue;
            auto associatedTP_address = associatedTP.get();

            // Compare addresses
            if (tp_ptr_address != associatedTP_address) {
                std::cout << "Addresses not the same: "
                          << "tp_ptr_address = " << tp_ptr_address 
                          << ", associatedTP_address = " << associatedTP_address << std::endl;
            }
        }

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
      }

      // If there are stubs on the same detid, loop on those stubs
      if (TTStubHandle->find(detidA) != TTStubHandle->end()) {
        edmNew::DetSet<TTStub<Ref_Phase2TrackerDigi_>> stubs = (*TTStubHandle)[detidA];
        for (auto stubIter = stubs.begin(); stubIter != stubs.end(); ++stubIter) {
          auto stubRef = edmNew::makeRefTo(TTStubHandle, stubIter);

          // Retrieve clusters of stubs
          auto clusterRefB = stubIter->clusterRef(0);
          auto clusterRefC = stubIter->clusterRef(1);

          // Retrieve sensor DetIds from the stub's clusters
          DetId detIdB = stubIter->clusterRef(0)->getDetId();
          DetId detIdC = stubIter->clusterRef(1)->getDetId();

          const GeomDetUnit *detB = theTrackerGeom->idToDetUnit(detIdB);
          const GeomDetUnit *detC = theTrackerGeom->idToDetUnit(detIdC);
          const PixelGeomDetUnit *theGeomDetB = dynamic_cast<const PixelGeomDetUnit *>(detB);
          const PixelGeomDetUnit *theGeomDetC = dynamic_cast<const PixelGeomDetUnit *>(detC);
          const PixelTopology *topoB = dynamic_cast<const PixelTopology *>(&(theGeomDetB->specificTopology()));
          const PixelTopology *topoC = dynamic_cast<const PixelTopology *>(&(theGeomDetC->specificTopology()));

          GlobalPoint coordsB = theGeomDetB->surface().toGlobal(
              topoB->localPosition(stubIter->clusterRef(0)->findAverageLocalCoordinatesCentered()));
          GlobalPoint coordsC = theGeomDetC->surface().toGlobal(
              topoC->localPosition(stubIter->clusterRef(1)->findAverageLocalCoordinatesCentered()));

          if (coordsA.x() == coordsB.x() || coordsA.x() == coordsC.x()) {
            edm::Ptr<TrackingParticle> stubTP =
                MCTruthTTStubHandle->findTrackingParticlePtr(edmNew::makeRefTo(TTStubHandle, stubIter));
            if (stubTP.isNull())
              continue;
            float stub_tp_pt = stubTP->pt();
            if (stub_tp_pt == tmp_tp_pt) {
              if (isBarrel == 1) {
                gen_clusters_if_stub_barrel->Fill(tmp_tp_pt);
                gen_clusters_if_stub_zoom_barrel->Fill(tmp_tp_pt);
              }
            }
          }
        }
      }
    }

    // ----------------------------------------------------------------------------------------------
    // look for L1 tracks matched to the tracking particle
    int tp_nMatch = 0;
    int i_track = -1;
    float i_chi2dof = 99999;
    if (MCTruthTTTrackHandle.isValid()) {
      std::vector<edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>> matchedTracks =
          MCTruthTTTrackHandle->findTTTrackPtrs(tp_ptr);

      // ----------------------------------------------------------------------------------------------
      // loop over matched L1 tracks
      // here, "match" means tracks that can be associated to a TrackingParticle
      // with at least one hit of at least one of its clusters
      // https://twiki.cern.ch/twiki/bin/viewauth/CMS/SLHCTrackerTriggerSWTools#MC_truth_for_TTTrack
      int trkCounter = 0;
      for (const auto &thisTrack : matchedTracks) {
        if (!MCTruthTTTrackHandle->isGenuine(thisTrack))
          continue;
        // ----------------------------------------------------------------------------------------------
        // further require L1 track to be (loosely) genuine, that there is only
        // one TP matched to the track
        // + have >= L1Tk_minNStub stubs for it to be a valid match
        int tmp_trk_nstub = thisTrack->getStubRefs().size();
        if (tmp_trk_nstub < L1Tk_minNStub)
          continue;
        float dmatch_pt = 999;
        float dmatch_eta = 999;
        float dmatch_phi = 999;
        int match_id = 999;

        edm::Ptr<TrackingParticle> my_tp = MCTruthTTTrackHandle->findTrackingParticlePtr(thisTrack);
        dmatch_pt = std::fabs(my_tp->p4().pt() - tmp_tp_pt);
        dmatch_eta = std::fabs(my_tp->p4().eta() - tmp_tp_eta);
        dmatch_phi = std::fabs(my_tp->p4().phi() - tmp_tp_phi);
        match_id = my_tp->pdgId();
        float tmp_trk_chi2dof = thisTrack->chi2Red();

        // ensure that track is uniquely matched to the TP we are looking at!
        if (dmatch_pt < 0.1 && dmatch_eta < 0.1 && dmatch_phi < 0.1 && tmp_tp_pdgid == match_id) {
          tp_nMatch++;
          if (i_track < 0 || tmp_trk_chi2dof < i_chi2dof) {
            i_track = trkCounter;
            i_chi2dof = tmp_trk_chi2dof;
          }
        }
        trkCounter++;
      }  // end loop over matched L1 tracks

      if (tp_nMatch < 1)
        continue;
      // Get information on the matched tracks
      float tmp_matchtrk_pt = -999;
      float tmp_matchtrk_eta = -999;
      float tmp_matchtrk_phi = -999;
      float tmp_matchtrk_VtxZ = -999;
      float tmp_matchtrk_chi2dof = -999;
      int tmp_matchTrk_nStub = -999;
      float tmp_matchtrk_d0 = -999;

      tmp_matchtrk_pt = matchedTracks[i_track]->momentum().perp();
      tmp_matchtrk_eta = matchedTracks[i_track]->momentum().eta();
      tmp_matchtrk_phi = matchedTracks[i_track]->momentum().phi();
      tmp_matchtrk_VtxZ = matchedTracks[i_track]->z0();
      tmp_matchtrk_chi2dof = matchedTracks[i_track]->chi2Red();
      tmp_matchTrk_nStub = (int)matchedTracks[i_track]->getStubRefs().size();

      //for d0
      float tmp_matchtrk_x0 = matchedTracks[i_track]->POCA().x();
      float tmp_matchtrk_y0 = matchedTracks[i_track]->POCA().y();
      tmp_matchtrk_d0 = -tmp_matchtrk_x0 * sin(tmp_matchtrk_phi) + tmp_matchtrk_y0 * cos(tmp_matchtrk_phi);

      //Add cuts for the matched tracks, numerator
      if (tmp_matchTrk_nStub < L1Tk_minNStub || tmp_matchtrk_chi2dof > L1Tk_maxChi2dof)
        continue;

      // fill matched track histograms (if passes all criteria)
      match_tp_pt->Fill(tmp_tp_pt);
      if (tmp_tp_pt > 0 && tmp_tp_pt <= 10)
        match_tp_pt_zoom->Fill(tmp_tp_pt);
      match_tp_eta->Fill(tmp_tp_eta);
      match_tp_d0->Fill(tmp_tp_d0);
      match_tp_VtxR->Fill(tmp_tp_VtxR);
      match_tp_VtxZ->Fill(tmp_tp_VtxZ);

      // Eta and pT histograms for resolution
      float pt_diff = tmp_matchtrk_pt - tmp_tp_pt;
      float pt_res = pt_diff / tmp_tp_pt;
      float eta_res = tmp_matchtrk_eta - tmp_tp_eta;
      float phi_res = tmp_matchtrk_phi - tmp_tp_phi;
      float VtxZ_res = tmp_matchtrk_VtxZ - tmp_tp_VtxZ;
      float d0_res = tmp_matchtrk_d0 - tmp_tp_d0;

      // fill total resolution histograms
      res_pt->Fill(pt_diff);
      res_ptRel->Fill(pt_res);
      res_eta->Fill(eta_res);

      // Fill resolution plots for different abs(eta) bins:
      // (0, 0.7), (0.7, 1.0), (1.0, 1.2), (1.2, 1.6), (1.6, 2.0), (2.0, 2.4)
      if (std::fabs(tmp_tp_eta) >= 0 && std::fabs(tmp_tp_eta) < 0.7) {
        reseta_eta0to0p7->Fill(eta_res);
        resphi_eta0to0p7->Fill(phi_res);
        resVtxZ_eta0to0p7->Fill(VtxZ_res);
        resd0_eta0to0p7->Fill(d0_res);
        if (tmp_tp_pt >= 2 && tmp_tp_pt < 3)
          respt_eta0to0p7_pt2to3->Fill(pt_res);
        else if (tmp_tp_pt >= 3 && tmp_tp_pt < 8)
          respt_eta0to0p7_pt3to8->Fill(pt_res);
        else if (tmp_tp_pt >= 8)
          respt_eta0to0p7_pt8toInf->Fill(pt_res);
      } else if (std::fabs(tmp_tp_eta) >= 0.7 && std::fabs(tmp_tp_eta) < 1.0) {
        reseta_eta0p7to1->Fill(eta_res);
        resphi_eta0p7to1->Fill(phi_res);
        resVtxZ_eta0p7to1->Fill(VtxZ_res);
        resd0_eta0p7to1->Fill(d0_res);
        if (tmp_tp_pt >= 2 && tmp_tp_pt < 3)
          respt_eta0p7to1_pt2to3->Fill(pt_res);
        else if (tmp_tp_pt >= 3 && tmp_tp_pt < 8)
          respt_eta0p7to1_pt3to8->Fill(pt_res);
        else if (tmp_tp_pt >= 8)
          respt_eta0p7to1_pt8toInf->Fill(pt_res);
      } else if (std::fabs(tmp_tp_eta) >= 1.0 && std::fabs(tmp_tp_eta) < 1.2) {
        reseta_eta1to1p2->Fill(eta_res);
        resphi_eta1to1p2->Fill(phi_res);
        resVtxZ_eta1to1p2->Fill(VtxZ_res);
        resd0_eta1to1p2->Fill(d0_res);
        if (tmp_tp_pt >= 2 && tmp_tp_pt < 3)
          respt_eta1to1p2_pt2to3->Fill(pt_res);
        else if (tmp_tp_pt >= 3 && tmp_tp_pt < 8)
          respt_eta1to1p2_pt3to8->Fill(pt_res);
        else if (tmp_tp_pt >= 8)
          respt_eta1to1p2_pt8toInf->Fill(pt_res);
      } else if (std::fabs(tmp_tp_eta) >= 1.2 && std::fabs(tmp_tp_eta) < 1.6) {
        reseta_eta1p2to1p6->Fill(eta_res);
        resphi_eta1p2to1p6->Fill(phi_res);
        resVtxZ_eta1p2to1p6->Fill(VtxZ_res);
        resd0_eta1p2to1p6->Fill(d0_res);
        if (tmp_tp_pt >= 2 && tmp_tp_pt < 3)
          respt_eta1p2to1p6_pt2to3->Fill(pt_res);
        else if (tmp_tp_pt >= 3 && tmp_tp_pt < 8)
          respt_eta1p2to1p6_pt3to8->Fill(pt_res);
        else if (tmp_tp_pt >= 8)
          respt_eta1p2to1p6_pt8toInf->Fill(pt_res);
      } else if (std::fabs(tmp_tp_eta) >= 1.6 && std::fabs(tmp_tp_eta) < 2.0) {
        reseta_eta1p6to2->Fill(eta_res);
        resphi_eta1p6to2->Fill(phi_res);
        resVtxZ_eta1p6to2->Fill(VtxZ_res);
        resd0_eta1p6to2->Fill(d0_res);
        if (tmp_tp_pt >= 2 && tmp_tp_pt < 3)
          respt_eta1p6to2_pt2to3->Fill(pt_res);
        else if (tmp_tp_pt >= 3 && tmp_tp_pt < 8)
          respt_eta1p6to2_pt3to8->Fill(pt_res);
        else if (tmp_tp_pt >= 8)
          respt_eta1p6to2_pt8toInf->Fill(pt_res);
      } else if (std::fabs(tmp_tp_eta) >= 2.0 && std::fabs(tmp_tp_eta) <= 2.4) {
        reseta_eta2to2p4->Fill(eta_res);
        resphi_eta2to2p4->Fill(phi_res);
        resVtxZ_eta2to2p4->Fill(VtxZ_res);
        resd0_eta2to2p4->Fill(d0_res);
        if (tmp_tp_pt >= 2 && tmp_tp_pt < 3)
          respt_eta2to2p4_pt2to3->Fill(pt_res);
        else if (tmp_tp_pt >= 3 && tmp_tp_pt < 8)
          respt_eta2to2p4_pt3to8->Fill(pt_res);
        else if (tmp_tp_pt >= 8)
          respt_eta2to2p4_pt8toInf->Fill(pt_res);
      }
    }  //if MC TTTrack handle is valid
  }    //end loop over tracking particles
}  // end of method

// ------------ method called once each job just before starting event loop
// ------------
void Phase2OTValidateTrackingParticles::bookHistograms(DQMStore::IBooker &iBooker,
                                                       edm::Run const &run,
                                                       edm::EventSetup const &es) {
  // Histogram setup and definitions
  std::string HistoName;
  iBooker.setCurrentFolder(topFolderName_ + "/trackParticles");

  // 1D: pT
  edm::ParameterSet psTrackParts_Pt = conf_.getParameter<edm::ParameterSet>("TH1TrackParts_Pt");
  HistoName = "trackParts_Pt";
  trackParts_Pt = iBooker.book1D(HistoName,
                                 HistoName,
                                 psTrackParts_Pt.getParameter<int32_t>("Nbinsx"),
                                 psTrackParts_Pt.getParameter<double>("xmin"),
                                 psTrackParts_Pt.getParameter<double>("xmax"));
  trackParts_Pt->setAxisTitle("p_{T} [GeV]", 1);
  trackParts_Pt->setAxisTitle("# tracking particles", 2);

  // 1D: eta
  edm::ParameterSet psTrackParts_Eta = conf_.getParameter<edm::ParameterSet>("TH1TrackParts_Eta");
  HistoName = "trackParts_Eta";
  trackParts_Eta = iBooker.book1D(HistoName,
                                  HistoName,
                                  psTrackParts_Eta.getParameter<int32_t>("Nbinsx"),
                                  psTrackParts_Eta.getParameter<double>("xmin"),
                                  psTrackParts_Eta.getParameter<double>("xmax"));
  trackParts_Eta->setAxisTitle("#eta", 1);
  trackParts_Eta->setAxisTitle("# tracking particles", 2);

  // 1D: phi
  edm::ParameterSet psTrackParts_Phi = conf_.getParameter<edm::ParameterSet>("TH1TrackParts_Phi");
  HistoName = "trackParts_Phi";
  trackParts_Phi = iBooker.book1D(HistoName,
                                  HistoName,
                                  psTrackParts_Phi.getParameter<int32_t>("Nbinsx"),
                                  psTrackParts_Phi.getParameter<double>("xmin"),
                                  psTrackParts_Phi.getParameter<double>("xmax"));
  trackParts_Phi->setAxisTitle("#phi", 1);
  trackParts_Phi->setAxisTitle("# tracking particles", 2);

  // 1D plots for efficiency
  iBooker.setCurrentFolder(topFolderName_ + "/EfficiencyIngredients");
  // pT
  edm::ParameterSet psEffic_pt = conf_.getParameter<edm::ParameterSet>("TH1Effic_pt");
  HistoName = "tp_pt";
  tp_pt = iBooker.book1D(HistoName,
                         HistoName,
                         psEffic_pt.getParameter<int32_t>("Nbinsx"),
                         psEffic_pt.getParameter<double>("xmin"),
                         psEffic_pt.getParameter<double>("xmax"));
  tp_pt->setAxisTitle("p_{T} [GeV]", 1);
  tp_pt->setAxisTitle("# tracking particles", 2);

  // Matched TP's pT
  HistoName = "match_tp_pt";
  match_tp_pt = iBooker.book1D(HistoName,
                               HistoName,
                               psEffic_pt.getParameter<int32_t>("Nbinsx"),
                               psEffic_pt.getParameter<double>("xmin"),
                               psEffic_pt.getParameter<double>("xmax"));
  match_tp_pt->setAxisTitle("p_{T} [GeV]", 1);
  match_tp_pt->setAxisTitle("# matched tracking particles", 2);

  // Gen clusters
  HistoName = "gen_clusters_barrel";
  gen_clusters_barrel = iBooker.book1D(HistoName,
                                HistoName,
                                psEffic_pt.getParameter<int32_t>("Nbinsx"),
                                psEffic_pt.getParameter<double>("xmin"),
                                psEffic_pt.getParameter<double>("xmax"));
  gen_clusters_barrel->setAxisTitle("p_{T} [GeV]", 1);
  gen_clusters_barrel->setAxisTitle("# tracking particles", 2);

  // Gen clusters if stub
  HistoName = "gen_clusters_if_stub_barrel";
  gen_clusters_if_stub_barrel = iBooker.book1D(HistoName,
                                        HistoName,
                                        psEffic_pt.getParameter<int32_t>("Nbinsx"),
                                        psEffic_pt.getParameter<double>("xmin"),
                                        psEffic_pt.getParameter<double>("xmax"));
  gen_clusters_if_stub_barrel->setAxisTitle("p_{T} [GeV]", 1);
  gen_clusters_if_stub_barrel->setAxisTitle("# tracking particles", 2);

  // pT zoom (0-10 GeV)
  edm::ParameterSet psEffic_pt_zoom = conf_.getParameter<edm::ParameterSet>("TH1Effic_pt_zoom");
  HistoName = "tp_pt_zoom";
  tp_pt_zoom = iBooker.book1D(HistoName,
                              HistoName,
                              psEffic_pt_zoom.getParameter<int32_t>("Nbinsx"),
                              psEffic_pt_zoom.getParameter<double>("xmin"),
                              psEffic_pt_zoom.getParameter<double>("xmax"));
  tp_pt_zoom->setAxisTitle("p_{T} [GeV]", 1);
  tp_pt_zoom->setAxisTitle("# tracking particles", 2);

  // Matched pT zoom (0-10 GeV)
  HistoName = "match_tp_pt_zoom";
  match_tp_pt_zoom = iBooker.book1D(HistoName,
                                    HistoName,
                                    psEffic_pt_zoom.getParameter<int32_t>("Nbinsx"),
                                    psEffic_pt_zoom.getParameter<double>("xmin"),
                                    psEffic_pt_zoom.getParameter<double>("xmax"));
  match_tp_pt_zoom->setAxisTitle("p_{T} [GeV]", 1);
  match_tp_pt_zoom->setAxisTitle("# matched tracking particles", 2);

  // pT zoom (0-10 GeV)
  HistoName = "gen_clusters_zoom_barrel";
  gen_clusters_zoom_barrel = iBooker.book1D(HistoName,
                                     HistoName,
                                     psEffic_pt_zoom.getParameter<int32_t>("Nbinsx"),
                                     psEffic_pt_zoom.getParameter<double>("xmin"),
                                     psEffic_pt_zoom.getParameter<double>("xmax"));
  gen_clusters_zoom_barrel->setAxisTitle("p_{T} [GeV]", 1);
  gen_clusters_zoom_barrel->setAxisTitle("# tracking particles", 2);

  // pT zoom (0-10 GeV)
  HistoName = "gen_clusters_if_stub_zoom_barrel";
  gen_clusters_if_stub_zoom_barrel = iBooker.book1D(HistoName,
                                             HistoName,
                                             psEffic_pt_zoom.getParameter<int32_t>("Nbinsx"),
                                             psEffic_pt_zoom.getParameter<double>("xmin"),
                                             psEffic_pt_zoom.getParameter<double>("xmax"));
  gen_clusters_if_stub_zoom_barrel->setAxisTitle("p_{T} [GeV]", 1);
  gen_clusters_if_stub_zoom_barrel->setAxisTitle("# tracking particles", 2);

  // eta
  edm::ParameterSet psEffic_eta = conf_.getParameter<edm::ParameterSet>("TH1Effic_eta");
  HistoName = "tp_eta";
  tp_eta = iBooker.book1D(HistoName,
                          HistoName,
                          psEffic_eta.getParameter<int32_t>("Nbinsx"),
                          psEffic_eta.getParameter<double>("xmin"),
                          psEffic_eta.getParameter<double>("xmax"));
  tp_eta->setAxisTitle("#eta", 1);
  tp_eta->setAxisTitle("# tracking particles", 2);

  // Matched eta
  HistoName = "match_tp_eta";
  match_tp_eta = iBooker.book1D(HistoName,
                                HistoName,
                                psEffic_eta.getParameter<int32_t>("Nbinsx"),
                                psEffic_eta.getParameter<double>("xmin"),
                                psEffic_eta.getParameter<double>("xmax"));
  match_tp_eta->setAxisTitle("#eta", 1);
  match_tp_eta->setAxisTitle("# matched tracking particles", 2);

  // d0
  edm::ParameterSet psEffic_d0 = conf_.getParameter<edm::ParameterSet>("TH1Effic_d0");
  HistoName = "tp_d0";
  tp_d0 = iBooker.book1D(HistoName,
                         HistoName,
                         psEffic_d0.getParameter<int32_t>("Nbinsx"),
                         psEffic_d0.getParameter<double>("xmin"),
                         psEffic_d0.getParameter<double>("xmax"));
  tp_d0->setAxisTitle("d_{0} [cm]", 1);
  tp_d0->setAxisTitle("# tracking particles", 2);

  // Matched d0
  HistoName = "match_tp_d0";
  match_tp_d0 = iBooker.book1D(HistoName,
                               HistoName,
                               psEffic_d0.getParameter<int32_t>("Nbinsx"),
                               psEffic_d0.getParameter<double>("xmin"),
                               psEffic_d0.getParameter<double>("xmax"));
  match_tp_d0->setAxisTitle("d_{0} [cm]", 1);
  match_tp_d0->setAxisTitle("# matched tracking particles", 2);

  // VtxR (also known as vxy)
  edm::ParameterSet psEffic_VtxR = conf_.getParameter<edm::ParameterSet>("TH1Effic_VtxR");
  HistoName = "tp_VtxR";
  tp_VtxR = iBooker.book1D(HistoName,
                           HistoName,
                           psEffic_VtxR.getParameter<int32_t>("Nbinsx"),
                           psEffic_VtxR.getParameter<double>("xmin"),
                           psEffic_VtxR.getParameter<double>("xmax"));
  tp_VtxR->setAxisTitle("d_{xy} [cm]", 1);
  tp_VtxR->setAxisTitle("# tracking particles", 2);

  // Matched VtxR
  HistoName = "match_tp_VtxR";
  match_tp_VtxR = iBooker.book1D(HistoName,
                                 HistoName,
                                 psEffic_VtxR.getParameter<int32_t>("Nbinsx"),
                                 psEffic_VtxR.getParameter<double>("xmin"),
                                 psEffic_VtxR.getParameter<double>("xmax"));
  match_tp_VtxR->setAxisTitle("d_{xy} [cm]", 1);
  match_tp_VtxR->setAxisTitle("# matched tracking particles", 2);

  // VtxZ
  edm::ParameterSet psEffic_VtxZ = conf_.getParameter<edm::ParameterSet>("TH1Effic_VtxZ");
  HistoName = "tp_VtxZ";
  tp_VtxZ = iBooker.book1D(HistoName,
                           HistoName,
                           psEffic_VtxZ.getParameter<int32_t>("Nbinsx"),
                           psEffic_VtxZ.getParameter<double>("xmin"),
                           psEffic_VtxZ.getParameter<double>("xmax"));
  tp_VtxZ->setAxisTitle("z_{0} [cm]", 1);
  tp_VtxZ->setAxisTitle("# tracking particles", 2);

  // Matched d0
  HistoName = "match_tp_VtxZ";
  match_tp_VtxZ = iBooker.book1D(HistoName,
                                 HistoName,
                                 psEffic_VtxZ.getParameter<int32_t>("Nbinsx"),
                                 psEffic_VtxZ.getParameter<double>("xmin"),
                                 psEffic_VtxZ.getParameter<double>("xmax"));
  match_tp_VtxZ->setAxisTitle("z_{0} [cm]", 1);
  match_tp_VtxZ->setAxisTitle("# matched tracking particles", 2);

  // 1D plots for resolution
  iBooker.setCurrentFolder(topFolderName_ + "/ResolutionIngredients");
  // full pT
  edm::ParameterSet psRes_pt = conf_.getParameter<edm::ParameterSet>("TH1Res_pt");
  HistoName = "res_pt";
  res_pt = iBooker.book1D(HistoName,
                          HistoName,
                          psRes_pt.getParameter<int32_t>("Nbinsx"),
                          psRes_pt.getParameter<double>("xmin"),
                          psRes_pt.getParameter<double>("xmax"));
  res_pt->setAxisTitle("p_{T} [GeV]", 1);
  res_pt->setAxisTitle("# tracking particles", 2);

  // Full eta
  edm::ParameterSet psRes_eta = conf_.getParameter<edm::ParameterSet>("TH1Res_eta");
  HistoName = "res_eta";
  res_eta = iBooker.book1D(HistoName,
                           HistoName,
                           psRes_eta.getParameter<int32_t>("Nbinsx"),
                           psRes_eta.getParameter<double>("xmin"),
                           psRes_eta.getParameter<double>("xmax"));
  res_eta->setAxisTitle("#eta", 1);
  res_eta->setAxisTitle("# tracking particles", 2);

  // Relative pT
  edm::ParameterSet psRes_ptRel = conf_.getParameter<edm::ParameterSet>("TH1Res_ptRel");
  HistoName = "res_ptRel";
  res_ptRel = iBooker.book1D(HistoName,
                             HistoName,
                             psRes_ptRel.getParameter<int32_t>("Nbinsx"),
                             psRes_ptRel.getParameter<double>("xmin"),
                             psRes_ptRel.getParameter<double>("xmax"));
  res_ptRel->setAxisTitle("Relative p_{T} [GeV]", 1);
  res_ptRel->setAxisTitle("# tracking particles", 2);

  // Eta parts (for resolution)
  // Eta 1 (0 to 0.7)
  HistoName = "reseta_eta0to0p7";
  reseta_eta0to0p7 = iBooker.book1D(HistoName,
                                    HistoName,
                                    psRes_eta.getParameter<int32_t>("Nbinsx"),
                                    psRes_eta.getParameter<double>("xmin"),
                                    psRes_eta.getParameter<double>("xmax"));
  reseta_eta0to0p7->setAxisTitle("#eta_{trk} - #eta_{tp}", 1);
  reseta_eta0to0p7->setAxisTitle("# tracking particles", 2);

  // Eta 2 (0.7 to 1.0)
  HistoName = "reseta_eta0p7to1";
  reseta_eta0p7to1 = iBooker.book1D(HistoName,
                                    HistoName,
                                    psRes_eta.getParameter<int32_t>("Nbinsx"),
                                    psRes_eta.getParameter<double>("xmin"),
                                    psRes_eta.getParameter<double>("xmax"));
  reseta_eta0p7to1->setAxisTitle("#eta_{trk} - #eta_{tp}", 1);
  reseta_eta0p7to1->setAxisTitle("# tracking particles", 2);

  // Eta 3 (1.0 to 1.2)
  HistoName = "reseta_eta1to1p2";
  reseta_eta1to1p2 = iBooker.book1D(HistoName,
                                    HistoName,
                                    psRes_eta.getParameter<int32_t>("Nbinsx"),
                                    psRes_eta.getParameter<double>("xmin"),
                                    psRes_eta.getParameter<double>("xmax"));
  reseta_eta1to1p2->setAxisTitle("#eta_{trk} - #eta_{tp}", 1);
  reseta_eta1to1p2->setAxisTitle("# tracking particles", 2);

  // Eta 4 (1.2 to 1.6)
  HistoName = "reseta_eta1p2to1p6";
  reseta_eta1p2to1p6 = iBooker.book1D(HistoName,
                                      HistoName,
                                      psRes_eta.getParameter<int32_t>("Nbinsx"),
                                      psRes_eta.getParameter<double>("xmin"),
                                      psRes_eta.getParameter<double>("xmax"));
  reseta_eta1p2to1p6->setAxisTitle("#eta_{trk} - #eta_{tp}", 1);
  reseta_eta1p2to1p6->setAxisTitle("# tracking particles", 2);

  // Eta 5 (1.6 to 2.0)
  HistoName = "reseta_eta1p6to2";
  reseta_eta1p6to2 = iBooker.book1D(HistoName,
                                    HistoName,
                                    psRes_eta.getParameter<int32_t>("Nbinsx"),
                                    psRes_eta.getParameter<double>("xmin"),
                                    psRes_eta.getParameter<double>("xmax"));
  reseta_eta1p6to2->setAxisTitle("#eta_{trk} - #eta_{tp}", 1);
  reseta_eta1p6to2->setAxisTitle("# tracking particles", 2);

  // Eta 6 (2.0 to 2.4)
  HistoName = "reseta_eta2to2p4";
  reseta_eta2to2p4 = iBooker.book1D(HistoName,
                                    HistoName,
                                    psRes_eta.getParameter<int32_t>("Nbinsx"),
                                    psRes_eta.getParameter<double>("xmin"),
                                    psRes_eta.getParameter<double>("xmax"));
  reseta_eta2to2p4->setAxisTitle("#eta_{trk} - #eta_{tp}", 1);
  reseta_eta2to2p4->setAxisTitle("# tracking particles", 2);

  // pT parts for resolution (pT res vs eta)
  // pT a (2 to 3 GeV)
  // Eta 1 (0 to 0.7)
  HistoName = "respt_eta0to0p7_pt2to3";
  respt_eta0to0p7_pt2to3 = iBooker.book1D(HistoName,
                                          HistoName,
                                          psRes_pt.getParameter<int32_t>("Nbinsx"),
                                          psRes_pt.getParameter<double>("xmin"),
                                          psRes_pt.getParameter<double>("xmax"));
  respt_eta0to0p7_pt2to3->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
  respt_eta0to0p7_pt2to3->setAxisTitle("# tracking particles", 2);

  // Eta 2 (0.7 to 1.0)
  HistoName = "respt_eta0p7to1_pt2to3";
  respt_eta0p7to1_pt2to3 = iBooker.book1D(HistoName,
                                          HistoName,
                                          psRes_pt.getParameter<int32_t>("Nbinsx"),
                                          psRes_pt.getParameter<double>("xmin"),
                                          psRes_pt.getParameter<double>("xmax"));
  respt_eta0p7to1_pt2to3->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
  respt_eta0p7to1_pt2to3->setAxisTitle("# tracking particles", 2);

  // Eta 3 (1.0 to 1.2)
  HistoName = "respt_eta1to1p2_pt2to3";
  respt_eta1to1p2_pt2to3 = iBooker.book1D(HistoName,
                                          HistoName,
                                          psRes_pt.getParameter<int32_t>("Nbinsx"),
                                          psRes_pt.getParameter<double>("xmin"),
                                          psRes_pt.getParameter<double>("xmax"));
  respt_eta1to1p2_pt2to3->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
  respt_eta1to1p2_pt2to3->setAxisTitle("# tracking particles", 2);

  // Eta 4 (1.2 to 1.6)
  HistoName = "respt_eta1p2to1p6_pt2to3";
  respt_eta1p2to1p6_pt2to3 = iBooker.book1D(HistoName,
                                            HistoName,
                                            psRes_pt.getParameter<int32_t>("Nbinsx"),
                                            psRes_pt.getParameter<double>("xmin"),
                                            psRes_pt.getParameter<double>("xmax"));
  respt_eta1p2to1p6_pt2to3->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
  respt_eta1p2to1p6_pt2to3->setAxisTitle("# tracking particles", 2);

  // Eta 5 (1.6 to 2.0)
  HistoName = "respt_eta1p6to2_pt2to3";
  respt_eta1p6to2_pt2to3 = iBooker.book1D(HistoName,
                                          HistoName,
                                          psRes_pt.getParameter<int32_t>("Nbinsx"),
                                          psRes_pt.getParameter<double>("xmin"),
                                          psRes_pt.getParameter<double>("xmax"));
  respt_eta1p6to2_pt2to3->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
  respt_eta1p6to2_pt2to3->setAxisTitle("# tracking particles", 2);

  // Eta 6 (2.0 to 2.4)
  HistoName = "respt_eta2to2p4_pt2to3";
  respt_eta2to2p4_pt2to3 = iBooker.book1D(HistoName,
                                          HistoName,
                                          psRes_pt.getParameter<int32_t>("Nbinsx"),
                                          psRes_pt.getParameter<double>("xmin"),
                                          psRes_pt.getParameter<double>("xmax"));
  respt_eta2to2p4_pt2to3->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
  respt_eta2to2p4_pt2to3->setAxisTitle("# tracking particles", 2);

  // pT b (3 to 8 GeV)
  // Eta 1 (0 to 0.7)
  HistoName = "respt_eta0to0p7_pt3to8";
  respt_eta0to0p7_pt3to8 = iBooker.book1D(HistoName,
                                          HistoName,
                                          psRes_pt.getParameter<int32_t>("Nbinsx"),
                                          psRes_pt.getParameter<double>("xmin"),
                                          psRes_pt.getParameter<double>("xmax"));
  respt_eta0to0p7_pt3to8->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
  respt_eta0to0p7_pt3to8->setAxisTitle("# tracking particles", 2);

  // Eta 2 (0.7 to 1.0)
  HistoName = "respt_eta0p7to1_pt3to8";
  respt_eta0p7to1_pt3to8 = iBooker.book1D(HistoName,
                                          HistoName,
                                          psRes_pt.getParameter<int32_t>("Nbinsx"),
                                          psRes_pt.getParameter<double>("xmin"),
                                          psRes_pt.getParameter<double>("xmax"));
  respt_eta0p7to1_pt3to8->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
  respt_eta0p7to1_pt3to8->setAxisTitle("# tracking particles", 2);

  // Eta 3 (1.0 to 1.2)
  HistoName = "respt_eta1to1p2_pt3to8";
  respt_eta1to1p2_pt3to8 = iBooker.book1D(HistoName,
                                          HistoName,
                                          psRes_pt.getParameter<int32_t>("Nbinsx"),
                                          psRes_pt.getParameter<double>("xmin"),
                                          psRes_pt.getParameter<double>("xmax"));
  respt_eta1to1p2_pt3to8->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
  respt_eta1to1p2_pt3to8->setAxisTitle("# tracking particles", 2);

  // Eta 4 (1.2 to 1.6)
  HistoName = "respt_eta1p2to1p6_pt3to8";
  respt_eta1p2to1p6_pt3to8 = iBooker.book1D(HistoName,
                                            HistoName,
                                            psRes_pt.getParameter<int32_t>("Nbinsx"),
                                            psRes_pt.getParameter<double>("xmin"),
                                            psRes_pt.getParameter<double>("xmax"));
  respt_eta1p2to1p6_pt3to8->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
  respt_eta1p2to1p6_pt3to8->setAxisTitle("# tracking particles", 2);

  // Eta 5 (1.6 to 2.0)
  HistoName = "respt_eta1p6to2_pt3to8";
  respt_eta1p6to2_pt3to8 = iBooker.book1D(HistoName,
                                          HistoName,
                                          psRes_pt.getParameter<int32_t>("Nbinsx"),
                                          psRes_pt.getParameter<double>("xmin"),
                                          psRes_pt.getParameter<double>("xmax"));
  respt_eta1p6to2_pt3to8->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
  respt_eta1p6to2_pt3to8->setAxisTitle("# tracking particles", 2);

  // Eta 6 (2.0 to 2.4)
  HistoName = "respt_eta2to2p4_pt3to8";
  respt_eta2to2p4_pt3to8 = iBooker.book1D(HistoName,
                                          HistoName,
                                          psRes_pt.getParameter<int32_t>("Nbinsx"),
                                          psRes_pt.getParameter<double>("xmin"),
                                          psRes_pt.getParameter<double>("xmax"));
  respt_eta2to2p4_pt3to8->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
  respt_eta2to2p4_pt3to8->setAxisTitle("# tracking particles", 2);

  // pT c (>8 GeV)
  // Eta 1 (0 to 0.7)
  HistoName = "respt_eta0to0p7_pt8toInf";
  respt_eta0to0p7_pt8toInf = iBooker.book1D(HistoName,
                                            HistoName,
                                            psRes_pt.getParameter<int32_t>("Nbinsx"),
                                            psRes_pt.getParameter<double>("xmin"),
                                            psRes_pt.getParameter<double>("xmax"));
  respt_eta0to0p7_pt8toInf->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
  respt_eta0to0p7_pt8toInf->setAxisTitle("# tracking particles", 2);

  // Eta 2 (0.7 to 1.0)
  HistoName = "respt_eta0p7to1_pt8toInf";
  respt_eta0p7to1_pt8toInf = iBooker.book1D(HistoName,
                                            HistoName,
                                            psRes_pt.getParameter<int32_t>("Nbinsx"),
                                            psRes_pt.getParameter<double>("xmin"),
                                            psRes_pt.getParameter<double>("xmax"));
  respt_eta0p7to1_pt8toInf->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
  respt_eta0p7to1_pt8toInf->setAxisTitle("# tracking particles", 2);

  // Eta 3 (1.0 to 1.2)
  HistoName = "respt_eta1to1p2_pt8toInf";
  respt_eta1to1p2_pt8toInf = iBooker.book1D(HistoName,
                                            HistoName,
                                            psRes_pt.getParameter<int32_t>("Nbinsx"),
                                            psRes_pt.getParameter<double>("xmin"),
                                            psRes_pt.getParameter<double>("xmax"));
  respt_eta1to1p2_pt8toInf->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
  respt_eta1to1p2_pt8toInf->setAxisTitle("# tracking particles", 2);

  // Eta 4 (1.2 to 1.6)
  HistoName = "respt_eta1p2to1p6_pt8toInf";
  respt_eta1p2to1p6_pt8toInf = iBooker.book1D(HistoName,
                                              HistoName,
                                              psRes_pt.getParameter<int32_t>("Nbinsx"),
                                              psRes_pt.getParameter<double>("xmin"),
                                              psRes_pt.getParameter<double>("xmax"));
  respt_eta1p2to1p6_pt8toInf->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
  respt_eta1p2to1p6_pt8toInf->setAxisTitle("# tracking particles", 2);

  // Eta 5 (1.6 to 2.0)
  HistoName = "respt_eta1p6to2_pt8toInf";
  respt_eta1p6to2_pt8toInf = iBooker.book1D(HistoName,
                                            HistoName,
                                            psRes_pt.getParameter<int32_t>("Nbinsx"),
                                            psRes_pt.getParameter<double>("xmin"),
                                            psRes_pt.getParameter<double>("xmax"));
  respt_eta1p6to2_pt8toInf->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
  respt_eta1p6to2_pt8toInf->setAxisTitle("# tracking particles", 2);

  // Eta 6 (2.0 to 2.4)
  HistoName = "respt_eta2to2p4_pt8toInf";
  respt_eta2to2p4_pt8toInf = iBooker.book1D(HistoName,
                                            HistoName,
                                            psRes_pt.getParameter<int32_t>("Nbinsx"),
                                            psRes_pt.getParameter<double>("xmin"),
                                            psRes_pt.getParameter<double>("xmax"));
  respt_eta2to2p4_pt8toInf->setAxisTitle("(p_{T}(trk) - p_{T}(tp))/p_{T}(tp)", 1);
  respt_eta2to2p4_pt8toInf->setAxisTitle("# tracking particles", 2);

  // Phi parts (for resolution)
  // Eta 1 (0 to 0.7)
  edm::ParameterSet psRes_phi = conf_.getParameter<edm::ParameterSet>("TH1Res_phi");
  HistoName = "resphi_eta0to0p7";
  resphi_eta0to0p7 = iBooker.book1D(HistoName,
                                    HistoName,
                                    psRes_phi.getParameter<int32_t>("Nbinsx"),
                                    psRes_phi.getParameter<double>("xmin"),
                                    psRes_phi.getParameter<double>("xmax"));
  resphi_eta0to0p7->setAxisTitle("#phi_{trk} - #phi_{tp}", 1);
  resphi_eta0to0p7->setAxisTitle("# tracking particles", 2);

  // Eta 2 (0.7 to 1.0)
  HistoName = "resphi_eta0p7to1";
  resphi_eta0p7to1 = iBooker.book1D(HistoName,
                                    HistoName,
                                    psRes_phi.getParameter<int32_t>("Nbinsx"),
                                    psRes_phi.getParameter<double>("xmin"),
                                    psRes_phi.getParameter<double>("xmax"));
  resphi_eta0p7to1->setAxisTitle("#phi_{trk} - #phi_{tp}", 1);
  resphi_eta0p7to1->setAxisTitle("# tracking particles", 2);

  // Eta 3 (1.0 to 1.2)
  HistoName = "resphi_eta1to1p2";
  resphi_eta1to1p2 = iBooker.book1D(HistoName,
                                    HistoName,
                                    psRes_phi.getParameter<int32_t>("Nbinsx"),
                                    psRes_phi.getParameter<double>("xmin"),
                                    psRes_phi.getParameter<double>("xmax"));
  resphi_eta1to1p2->setAxisTitle("#phi_{trk} - #phi_{tp}", 1);
  resphi_eta1to1p2->setAxisTitle("# tracking particles", 2);

  // Eta 4 (1.2 to 1.6)
  HistoName = "resphi_eta1p2to1p6";
  resphi_eta1p2to1p6 = iBooker.book1D(HistoName,
                                      HistoName,
                                      psRes_phi.getParameter<int32_t>("Nbinsx"),
                                      psRes_phi.getParameter<double>("xmin"),
                                      psRes_phi.getParameter<double>("xmax"));
  resphi_eta1p2to1p6->setAxisTitle("#phi_{trk} - #phi_{tp}", 1);
  resphi_eta1p2to1p6->setAxisTitle("# tracking particles", 2);

  // Eta 5 (1.6 to 2.0)
  HistoName = "resphi_eta1p6to2";
  resphi_eta1p6to2 = iBooker.book1D(HistoName,
                                    HistoName,
                                    psRes_phi.getParameter<int32_t>("Nbinsx"),
                                    psRes_phi.getParameter<double>("xmin"),
                                    psRes_phi.getParameter<double>("xmax"));
  resphi_eta1p6to2->setAxisTitle("#phi_{trk} - #phi_{tp}", 1);
  resphi_eta1p6to2->setAxisTitle("# tracking particles", 2);

  // Eta 6 (2.0 to 2.4)
  HistoName = "resphi_eta2to2p4";
  resphi_eta2to2p4 = iBooker.book1D(HistoName,
                                    HistoName,
                                    psRes_phi.getParameter<int32_t>("Nbinsx"),
                                    psRes_phi.getParameter<double>("xmin"),
                                    psRes_phi.getParameter<double>("xmax"));
  resphi_eta2to2p4->setAxisTitle("#phi_{trk} - #phi_{tp}", 1);
  resphi_eta2to2p4->setAxisTitle("# tracking particles", 2);

  // VtxZ parts (for resolution)
  // Eta 1 (0 to 0.7)
  edm::ParameterSet psRes_VtxZ = conf_.getParameter<edm::ParameterSet>("TH1Res_VtxZ");
  HistoName = "resVtxZ_eta0to0p7";
  resVtxZ_eta0to0p7 = iBooker.book1D(HistoName,
                                     HistoName,
                                     psRes_VtxZ.getParameter<int32_t>("Nbinsx"),
                                     psRes_VtxZ.getParameter<double>("xmin"),
                                     psRes_VtxZ.getParameter<double>("xmax"));
  resVtxZ_eta0to0p7->setAxisTitle("VtxZ_{trk} - VtxZ_{tp} [cm]", 1);
  resVtxZ_eta0to0p7->setAxisTitle("# tracking particles", 2);

  // Eta 2 (0.7 to 1.0)
  HistoName = "resVtxZ_eta0p7to1";
  resVtxZ_eta0p7to1 = iBooker.book1D(HistoName,
                                     HistoName,
                                     psRes_VtxZ.getParameter<int32_t>("Nbinsx"),
                                     psRes_VtxZ.getParameter<double>("xmin"),
                                     psRes_VtxZ.getParameter<double>("xmax"));
  resVtxZ_eta0p7to1->setAxisTitle("VtxZ_{trk} - VtxZ_{tp} [cm]", 1);
  resVtxZ_eta0p7to1->setAxisTitle("# tracking particles", 2);

  // Eta 3 (1.0 to 1.2)
  HistoName = "resVtxZ_eta1to1p2";
  resVtxZ_eta1to1p2 = iBooker.book1D(HistoName,
                                     HistoName,
                                     psRes_VtxZ.getParameter<int32_t>("Nbinsx"),
                                     psRes_VtxZ.getParameter<double>("xmin"),
                                     psRes_VtxZ.getParameter<double>("xmax"));
  resVtxZ_eta1to1p2->setAxisTitle("VtxZ_{trk} - VtxZ_{tp} [cm]", 1);
  resVtxZ_eta1to1p2->setAxisTitle("# tracking particles", 2);

  // Eta 4 (1.2 to 1.6)
  HistoName = "resVtxZ_eta1p2to1p6";
  resVtxZ_eta1p2to1p6 = iBooker.book1D(HistoName,
                                       HistoName,
                                       psRes_VtxZ.getParameter<int32_t>("Nbinsx"),
                                       psRes_VtxZ.getParameter<double>("xmin"),
                                       psRes_VtxZ.getParameter<double>("xmax"));
  resVtxZ_eta1p2to1p6->setAxisTitle("VtxZ_{trk} - VtxZ_{tp} [cm]", 1);
  resVtxZ_eta1p2to1p6->setAxisTitle("# tracking particles", 2);

  // Eta 5 (1.6 to 2.0)
  HistoName = "resVtxZ_eta1p6to2";
  resVtxZ_eta1p6to2 = iBooker.book1D(HistoName,
                                     HistoName,
                                     psRes_VtxZ.getParameter<int32_t>("Nbinsx"),
                                     psRes_VtxZ.getParameter<double>("xmin"),
                                     psRes_VtxZ.getParameter<double>("xmax"));
  resVtxZ_eta1p6to2->setAxisTitle("VtxZ_{trk} - VtxZ_{tp} [cm]", 1);
  resVtxZ_eta1p6to2->setAxisTitle("# tracking particles", 2);

  // Eta 6 (2.0 to 2.4)
  HistoName = "resVtxZ_eta2to2p4";
  resVtxZ_eta2to2p4 = iBooker.book1D(HistoName,
                                     HistoName,
                                     psRes_VtxZ.getParameter<int32_t>("Nbinsx"),
                                     psRes_VtxZ.getParameter<double>("xmin"),
                                     psRes_VtxZ.getParameter<double>("xmax"));
  resVtxZ_eta2to2p4->setAxisTitle("VtxZ_{trk} - VtxZ_{tp} [cm]", 1);
  resVtxZ_eta2to2p4->setAxisTitle("# tracking particles", 2);

  // d0 parts (for resolution)
  // Eta 1 (0 to 0.7)
  edm::ParameterSet psRes_d0 = conf_.getParameter<edm::ParameterSet>("TH1Res_d0");
  HistoName = "resd0_eta0to0p7";
  resd0_eta0to0p7 = iBooker.book1D(HistoName,
                                   HistoName,
                                   psRes_d0.getParameter<int32_t>("Nbinsx"),
                                   psRes_d0.getParameter<double>("xmin"),
                                   psRes_d0.getParameter<double>("xmax"));
  resd0_eta0to0p7->setAxisTitle("d0_{trk} - d0_{tp} [cm]", 1);
  resd0_eta0to0p7->setAxisTitle("# tracking particles", 2);

  // Eta 2 (0.7 to 1.0)
  HistoName = "resd0_eta0p7to1";
  resd0_eta0p7to1 = iBooker.book1D(HistoName,
                                   HistoName,
                                   psRes_d0.getParameter<int32_t>("Nbinsx"),
                                   psRes_d0.getParameter<double>("xmin"),
                                   psRes_d0.getParameter<double>("xmax"));
  resd0_eta0p7to1->setAxisTitle("d0_{trk} - d0_{tp} [cm]", 1);
  resd0_eta0p7to1->setAxisTitle("# tracking particles", 2);

  // Eta 3 (1.0 to 1.2)
  HistoName = "resd0_eta1to1p2";
  resd0_eta1to1p2 = iBooker.book1D(HistoName,
                                   HistoName,
                                   psRes_d0.getParameter<int32_t>("Nbinsx"),
                                   psRes_d0.getParameter<double>("xmin"),
                                   psRes_d0.getParameter<double>("xmax"));
  resd0_eta1to1p2->setAxisTitle("d0_{trk} - d0_{tp} [cm]", 1);
  resd0_eta1to1p2->setAxisTitle("# tracking particles", 2);

  // Eta 4 (1.2 to 1.6)
  HistoName = "resd0_eta1p2to1p6";
  resd0_eta1p2to1p6 = iBooker.book1D(HistoName,
                                     HistoName,
                                     psRes_d0.getParameter<int32_t>("Nbinsx"),
                                     psRes_d0.getParameter<double>("xmin"),
                                     psRes_d0.getParameter<double>("xmax"));
  resd0_eta1p2to1p6->setAxisTitle("d0_{trk} - d0_{tp} [cm]", 1);
  resd0_eta1p2to1p6->setAxisTitle("# tracking particles", 2);

  // Eta 5 (1.6 to 2.0)
  HistoName = "resd0_eta1p6to2";
  resd0_eta1p6to2 = iBooker.book1D(HistoName,
                                   HistoName,
                                   psRes_d0.getParameter<int32_t>("Nbinsx"),
                                   psRes_d0.getParameter<double>("xmin"),
                                   psRes_d0.getParameter<double>("xmax"));
  resd0_eta1p6to2->setAxisTitle("d0_{trk} - d0_{tp} [cm]", 1);
  resd0_eta1p6to2->setAxisTitle("# tracking particles", 2);

  // Eta 6 (2.0 to 2.4)
  HistoName = "resd0_eta2to2p4";
  resd0_eta2to2p4 = iBooker.book1D(HistoName,
                                   HistoName,
                                   psRes_d0.getParameter<int32_t>("Nbinsx"),
                                   psRes_d0.getParameter<double>("xmin"),
                                   psRes_d0.getParameter<double>("xmax"));
  resd0_eta2to2p4->setAxisTitle("d0_{trk} - d0_{tp} [cm]", 1);
  resd0_eta2to2p4->setAxisTitle("# tracking particles", 2);

}  // end of method

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
void Phase2OTValidateTrackingParticles::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // OuterTrackerMonitorTrackingParticles
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 45);
    psd0.add<double>("xmax", 3);
    psd0.add<double>("xmin", -3);
    desc.add<edm::ParameterSetDescription>("TH1TrackParts_Eta", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 60);
    psd0.add<double>("xmax", 3.141592653589793);
    psd0.add<double>("xmin", -3.141592653589793);
    desc.add<edm::ParameterSetDescription>("TH1TrackParts_Phi", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 45);
    psd0.add<double>("xmax", 100);
    psd0.add<double>("xmin", 0);
    desc.add<edm::ParameterSetDescription>("TH1TrackParts_Pt", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 200);
    psd0.add<double>("xmax", 0.5);
    psd0.add<double>("xmin", -0.5);
    desc.add<edm::ParameterSetDescription>("TH1Res_ptRel", psd0);
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
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 50);
    psd0.add<double>("xmax", 2.5);
    psd0.add<double>("xmin", -2.5);
    desc.add<edm::ParameterSetDescription>("TH1Effic_eta", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 50);
    psd0.add<double>("xmax", 2);
    psd0.add<double>("xmin", -2);
    desc.add<edm::ParameterSetDescription>("TH1Effic_d0", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 50);
    psd0.add<double>("xmax", 5);
    psd0.add<double>("xmin", -5);
    desc.add<edm::ParameterSetDescription>("TH1Effic_VtxR", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 50);
    psd0.add<double>("xmax", 30);
    psd0.add<double>("xmin", -30);
    desc.add<edm::ParameterSetDescription>("TH1Effic_VtxZ", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 100);
    psd0.add<double>("xmax", 0.2);
    psd0.add<double>("xmin", -0.2);
    desc.add<edm::ParameterSetDescription>("TH1Res_pt", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 100);
    psd0.add<double>("xmax", 0.01);
    psd0.add<double>("xmin", -0.01);
    desc.add<edm::ParameterSetDescription>("TH1Res_eta", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 100);
    psd0.add<double>("xmax", 0.01);
    psd0.add<double>("xmin", -0.01);
    desc.add<edm::ParameterSetDescription>("TH1Res_phi", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 100);
    psd0.add<double>("xmax", 1.0);
    psd0.add<double>("xmin", -1.0);
    desc.add<edm::ParameterSetDescription>("TH1Res_VtxZ", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("Nbinsx", 100);
    psd0.add<double>("xmax", 0.05);
    psd0.add<double>("xmin", -0.05);
    desc.add<edm::ParameterSetDescription>("TH1Res_d0", psd0);
  }
  desc.add<std::string>("TopFolderName", "TrackerPhase2OTL1TrackV");
  desc.add<edm::InputTag>("trackingParticleToken", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("MCTruthStubInputTag", edm::InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"));
  desc.add<edm::InputTag>("MCTruthTrackInputTag", edm::InputTag("TTTrackAssociatorFromPixelDigis", "Level1TTTracks"));
  desc.add<edm::InputTag>("MCTruthClusterInputTag",
                          edm::InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"));
  desc.add<int>("L1Tk_minNStub", 4);
  desc.add<double>("L1Tk_maxChi2dof", 25.0);
  desc.add<int>("TP_minNStub", 4);
  desc.add<int>("TP_minNLayersStub", 4);
  desc.add<double>("TP_minPt", 2.0);
  desc.add<double>("TP_maxEta", 2.4);
  desc.add<double>("TP_maxVtxZ", 15.0);
  descriptions.add("Phase2OTValidateTrackingParticles", desc);
  // or use the following to generate the label from the module's C++ type
  //descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(Phase2OTValidateTrackingParticles);

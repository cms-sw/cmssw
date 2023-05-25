///*
#include <string>
#include <tuple>
#include <numeric>
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>  // unique_ptr

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/ValidHandle.h"
#include "DataFormats/Math/interface/deltaR.h"
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
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimDataFormats/Associations/interface/TrackToGenParticleAssociator.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "HepMC/GenRanges.h"
#include "CLHEP/Units/PhysicalConstants.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"  // Adding header files for electrons
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"         // Adding header files for electrons
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

// eff vs PU test libraries

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimTracker/VertexAssociation/interface/calculateVertexSharedTracks.h"
#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"
#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"

// end of test libraries

class MtdEleIsoValidation : public DQMEDAnalyzer {
public:
  explicit MtdEleIsoValidation(const edm::ParameterSet&);
  ~MtdEleIsoValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  const bool mvaGenSel(const HepMC::GenParticle&, const float&);
  const bool mvaRecSel(const reco::TrackBase&, const reco::Vertex&, const double&, const double&);
  const bool mvaGenRecMatch(const HepMC::GenParticle&, const double&, const reco::TrackBase&);
  bool pdgCheck(int pdg);

  // ------------ member data ------------

  const std::string folder_;
  const float trackMinPt_;
  const float trackMinEta_;
  const float trackMaxEta_;
  const double rel_iso_cut_;

  bool track_match_PV_;
  bool dt_sig_vtx_;
  bool dt_sig_track_;
  bool dt_distributions_;

  static constexpr float min_dR_cut = 0.01;
  static constexpr float max_dR_cut = 0.3;
  static constexpr float min_pt_cut_EB = 0.7;
  static constexpr float min_pt_cut_EE = 0.4;
  static constexpr float max_dz_cut_EB = 0.5;
  static constexpr float max_dz_cut_EE = 0.5;
  static constexpr float max_dz_vtx_cut = 0.5;
  static constexpr float max_dxy_vtx_cut = 0.2;
  // timing cuts - has to be 7 values here!!! Code created for 7 dt values!! Values for resolution 100;90;80;70;60;50;40 ps
  //const std::vector<double> max_dt_vtx_cut{0.30, 0.27, 0.24, 0.21, 0.18, 0.15, 0.12};  // default cuts
  //const std::vector<double> max_dt_track_cut{0.30, 0.27, 0.24, 0.21, 0.18, 0.15, 0.12}; // default cuts
  const std::vector<double> max_dt_vtx_cut{0.30, 0.24, 0.18, 0.15, 0.12, 0.08, 0.04};    // test cuts
  const std::vector<double> max_dt_track_cut{0.30, 0.24, 0.18, 0.15, 0.12, 0.08, 0.04};  // test cuts
  const std::vector<double> max_dt_significance_cut{4.0, 3.0, 2.0};                      // test cuts
  static constexpr float min_strip_cut = 0.01;
  static constexpr float min_track_mtd_mva_cut = 0.5;
  const std::vector<double> pT_bins_dt_distrb{10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  const std::vector<double> eta_bins_dt_distrib{0.0, 0.5, 1.0, 1.5, 2.0, 2.4};
  static constexpr double avg_sim_sigTrk_t_err = 0.03239;  // avg error/resolution for SIM tracks in nanoseconds
  static constexpr double avg_sim_PUtrack_t_err = 0.03465;
  static constexpr double avg_sim_vertex_t_err = 0.1;

  edm::EDGetTokenT<reco::TrackCollection> GenRecTrackToken_;
  edm::EDGetTokenT<reco::TrackCollection> RecTrackToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> RecVertexToken_;

  edm::EDGetTokenT<reco::GsfElectronCollection> GsfElectronToken_EB_;  // Adding token for electron collection
  edm::EDGetTokenT<reco::GsfElectronCollection> GsfElectronToken_EE_;
  edm::EDGetTokenT<reco::GenParticleCollection> GenParticleToken_;

  edm::EDGetTokenT<edm::HepMCProduct> HepMCProductToken_;

  edm::EDGetTokenT<edm::ValueMap<int>> trackAssocToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> pathLengthToken_;

  edm::EDGetTokenT<edm::ValueMap<float>> tmtdToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> SigmatmtdToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> t0SrcToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> Sigmat0SrcToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> t0PidToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> Sigmat0PidToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> t0SafePidToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> Sigmat0SafePidToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> trackMVAQualToken_;

  edm::ESGetToken<HepPDT::ParticleDataTable, edm::DefaultRecord> particleTableToken_;

  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleCollectionToken_;  // From Aurora
  edm::EDGetTokenT<reco::RecoToSimCollection> recoToSimAssociationToken_;         // From Aurora

  // Signal histograms

  MonitorElement* meEle_no_dt_check_;
  MonitorElement* meTrk_genMatch_check_;
  MonitorElement* meEle_test_check_;

  MonitorElement* meEle_avg_error_SigTrk_check_;
  MonitorElement* meEle_avg_error_PUTrk_check_;
  MonitorElement* meEle_avg_error_vtx_check_;

  MonitorElement* meEleISO_Ntracks_Sig_EB_;  // Adding histograms for barrel electrons (isolation stuff)
  MonitorElement* meEleISO_chIso_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_1_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_1_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_1_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_2_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_2_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_2_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_3_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_3_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_3_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_4_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_4_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_4_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_5_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_5_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_5_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_6_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_6_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_6_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_7_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_7_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_7_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_2sigma_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_2sigma_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_2sigma_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_3sigma_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_3sigma_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_3sigma_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_4sigma_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_4sigma_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_4sigma_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_1_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_1_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_1_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_2_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_2_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_2_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_3_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_3_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_3_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_4_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_4_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_4_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_5_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_5_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_5_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_6_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_6_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_6_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_7_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_7_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_7_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_2sigma_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_2sigma_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_2sigma_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_3sigma_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_3sigma_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_3sigma_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_4sigma_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_4sigma_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_4sigma_Sig_EB_;

  MonitorElement* meEle_pt_tot_Sig_EB_;
  MonitorElement* meEle_pt_sim_tot_Sig_EB_;
  MonitorElement* meEle_eta_tot_Sig_EB_;
  MonitorElement* meEle_phi_tot_Sig_EB_;

  MonitorElement* meEle_pt_MTD_1_Sig_EB_;
  MonitorElement* meEle_pt_sim_MTD_1_Sig_EB_;
  MonitorElement* meEle_eta_MTD_1_Sig_EB_;
  MonitorElement* meEle_phi_MTD_1_Sig_EB_;

  MonitorElement* meEle_pt_MTD_2_Sig_EB_;
  MonitorElement* meEle_pt_sim_MTD_2_Sig_EB_;
  MonitorElement* meEle_eta_MTD_2_Sig_EB_;
  MonitorElement* meEle_phi_MTD_2_Sig_EB_;

  MonitorElement* meEle_pt_MTD_3_Sig_EB_;
  MonitorElement* meEle_pt_sim_MTD_3_Sig_EB_;
  MonitorElement* meEle_eta_MTD_3_Sig_EB_;
  MonitorElement* meEle_phi_MTD_3_Sig_EB_;

  MonitorElement* meEle_pt_MTD_4_Sig_EB_;
  MonitorElement* meEle_pt_sim_MTD_4_Sig_EB_;
  MonitorElement* meEle_eta_MTD_4_Sig_EB_;
  MonitorElement* meEle_phi_MTD_4_Sig_EB_;

  MonitorElement* meEle_pt_MTD_5_Sig_EB_;
  MonitorElement* meEle_pt_sim_MTD_5_Sig_EB_;
  MonitorElement* meEle_eta_MTD_5_Sig_EB_;
  MonitorElement* meEle_phi_MTD_5_Sig_EB_;

  MonitorElement* meEle_pt_MTD_6_Sig_EB_;
  MonitorElement* meEle_pt_sim_MTD_6_Sig_EB_;
  MonitorElement* meEle_eta_MTD_6_Sig_EB_;
  MonitorElement* meEle_phi_MTD_6_Sig_EB_;

  MonitorElement* meEle_pt_MTD_7_Sig_EB_;
  MonitorElement* meEle_pt_sim_MTD_7_Sig_EB_;
  MonitorElement* meEle_eta_MTD_7_Sig_EB_;
  MonitorElement* meEle_phi_MTD_7_Sig_EB_;

  MonitorElement* meEle_pt_noMTD_Sig_EB_;
  MonitorElement* meEle_eta_noMTD_Sig_EB_;
  MonitorElement* meEle_phi_noMTD_Sig_EB_;

  MonitorElement* meEle_pt_MTD_2sigma_Sig_EB_;
  MonitorElement* meEle_pt_sim_MTD_2sigma_Sig_EB_;
  MonitorElement* meEle_eta_MTD_2sigma_Sig_EB_;
  MonitorElement* meEle_phi_MTD_2sigma_Sig_EB_;

  MonitorElement* meEle_pt_MTD_3sigma_Sig_EB_;
  MonitorElement* meEle_pt_sim_MTD_3sigma_Sig_EB_;
  MonitorElement* meEle_eta_MTD_3sigma_Sig_EB_;
  MonitorElement* meEle_phi_MTD_3sigma_Sig_EB_;

  MonitorElement* meEle_pt_MTD_4sigma_Sig_EB_;
  MonitorElement* meEle_pt_sim_MTD_4sigma_Sig_EB_;
  MonitorElement* meEle_eta_MTD_4sigma_Sig_EB_;
  MonitorElement* meEle_phi_MTD_4sigma_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_Sig_EE_;  // Adding histograms for endcap electrons (isolation stuff)
  MonitorElement* meEleISO_chIso_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_1_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_1_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_1_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_2_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_2_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_2_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_3_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_3_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_3_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_4_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_4_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_4_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_5_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_5_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_5_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_6_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_6_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_6_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_7_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_7_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_7_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_2sigma_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_2sigma_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_2sigma_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_3sigma_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_3sigma_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_3sigma_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_4sigma_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_4sigma_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_4sigma_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_1_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_1_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_1_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_2_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_2_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_2_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_3_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_3_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_3_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_4_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_4_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_4_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_5_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_5_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_5_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_6_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_6_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_6_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_7_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_7_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_7_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_2sigma_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_2sigma_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_2sigma_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_3sigma_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_3sigma_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_3sigma_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_4sigma_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_4sigma_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_4sigma_Sig_EE_;

  MonitorElement* meEle_pt_tot_Sig_EE_;
  MonitorElement* meEle_pt_sim_tot_Sig_EE_;
  MonitorElement* meEle_eta_tot_Sig_EE_;
  MonitorElement* meEle_phi_tot_Sig_EE_;

  MonitorElement* meEle_pt_MTD_1_Sig_EE_;
  MonitorElement* meEle_pt_sim_MTD_1_Sig_EE_;
  MonitorElement* meEle_eta_MTD_1_Sig_EE_;
  MonitorElement* meEle_phi_MTD_1_Sig_EE_;

  MonitorElement* meEle_pt_MTD_2_Sig_EE_;
  MonitorElement* meEle_pt_sim_MTD_2_Sig_EE_;
  MonitorElement* meEle_eta_MTD_2_Sig_EE_;
  MonitorElement* meEle_phi_MTD_2_Sig_EE_;

  MonitorElement* meEle_pt_MTD_3_Sig_EE_;
  MonitorElement* meEle_pt_sim_MTD_3_Sig_EE_;
  MonitorElement* meEle_eta_MTD_3_Sig_EE_;
  MonitorElement* meEle_phi_MTD_3_Sig_EE_;

  MonitorElement* meEle_pt_MTD_4_Sig_EE_;
  MonitorElement* meEle_pt_sim_MTD_4_Sig_EE_;
  MonitorElement* meEle_eta_MTD_4_Sig_EE_;
  MonitorElement* meEle_phi_MTD_4_Sig_EE_;

  MonitorElement* meEle_pt_MTD_5_Sig_EE_;
  MonitorElement* meEle_pt_sim_MTD_5_Sig_EE_;
  MonitorElement* meEle_eta_MTD_5_Sig_EE_;
  MonitorElement* meEle_phi_MTD_5_Sig_EE_;

  MonitorElement* meEle_pt_MTD_6_Sig_EE_;
  MonitorElement* meEle_pt_sim_MTD_6_Sig_EE_;
  MonitorElement* meEle_eta_MTD_6_Sig_EE_;
  MonitorElement* meEle_phi_MTD_6_Sig_EE_;

  MonitorElement* meEle_pt_MTD_7_Sig_EE_;
  MonitorElement* meEle_pt_sim_MTD_7_Sig_EE_;
  MonitorElement* meEle_eta_MTD_7_Sig_EE_;
  MonitorElement* meEle_phi_MTD_7_Sig_EE_;

  MonitorElement* meEle_pt_noMTD_Sig_EE_;
  MonitorElement* meEle_eta_noMTD_Sig_EE_;
  MonitorElement* meEle_phi_noMTD_Sig_EE_;

  MonitorElement* meEle_pt_MTD_2sigma_Sig_EE_;
  MonitorElement* meEle_pt_sim_MTD_2sigma_Sig_EE_;
  MonitorElement* meEle_eta_MTD_2sigma_Sig_EE_;
  MonitorElement* meEle_phi_MTD_2sigma_Sig_EE_;

  MonitorElement* meEle_pt_MTD_3sigma_Sig_EE_;
  MonitorElement* meEle_pt_sim_MTD_3sigma_Sig_EE_;
  MonitorElement* meEle_eta_MTD_3sigma_Sig_EE_;
  MonitorElement* meEle_phi_MTD_3sigma_Sig_EE_;

  MonitorElement* meEle_pt_MTD_4sigma_Sig_EE_;
  MonitorElement* meEle_pt_sim_MTD_4sigma_Sig_EE_;
  MonitorElement* meEle_eta_MTD_4sigma_Sig_EE_;
  MonitorElement* meEle_phi_MTD_4sigma_Sig_EE_;

  // Signal histograms end

  // Background histograms
  MonitorElement* meEleISO_Ntracks_Bkg_EB_;  // Adding histograms for barrel electrons (isolation stuff)
  MonitorElement* meEleISO_chIso_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_1_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_1_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_1_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_2_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_2_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_2_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_3_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_3_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_3_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_4_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_4_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_4_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_5_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_5_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_5_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_6_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_6_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_6_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_7_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_7_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_7_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_2sigma_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_2sigma_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_2sigma_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_3sigma_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_3sigma_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_3sigma_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_4sigma_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_4sigma_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_4sigma_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_1_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_1_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_1_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_2_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_2_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_2_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_3_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_3_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_3_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_4_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_4_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_4_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_5_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_5_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_5_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_6_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_6_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_6_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_7_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_7_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_7_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_2sigma_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_2sigma_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_2sigma_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_3sigma_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_3sigma_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_3sigma_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_4sigma_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_sim_4sigma_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_4sigma_Bkg_EB_;

  MonitorElement* meEle_pt_tot_Bkg_EB_;
  MonitorElement* meEle_pt_sim_tot_Bkg_EB_;
  MonitorElement* meEle_eta_tot_Bkg_EB_;
  MonitorElement* meEle_phi_tot_Bkg_EB_;

  MonitorElement* meEle_pt_MTD_1_Bkg_EB_;
  MonitorElement* meEle_pt_sim_MTD_1_Bkg_EB_;
  MonitorElement* meEle_eta_MTD_1_Bkg_EB_;
  MonitorElement* meEle_phi_MTD_1_Bkg_EB_;

  MonitorElement* meEle_pt_MTD_2_Bkg_EB_;
  MonitorElement* meEle_pt_sim_MTD_2_Bkg_EB_;
  MonitorElement* meEle_eta_MTD_2_Bkg_EB_;
  MonitorElement* meEle_phi_MTD_2_Bkg_EB_;

  MonitorElement* meEle_pt_MTD_3_Bkg_EB_;
  MonitorElement* meEle_pt_sim_MTD_3_Bkg_EB_;
  MonitorElement* meEle_eta_MTD_3_Bkg_EB_;
  MonitorElement* meEle_phi_MTD_3_Bkg_EB_;

  MonitorElement* meEle_pt_MTD_4_Bkg_EB_;
  MonitorElement* meEle_pt_sim_MTD_4_Bkg_EB_;
  MonitorElement* meEle_eta_MTD_4_Bkg_EB_;
  MonitorElement* meEle_phi_MTD_4_Bkg_EB_;

  MonitorElement* meEle_pt_MTD_5_Bkg_EB_;
  MonitorElement* meEle_pt_sim_MTD_5_Bkg_EB_;
  MonitorElement* meEle_eta_MTD_5_Bkg_EB_;
  MonitorElement* meEle_phi_MTD_5_Bkg_EB_;

  MonitorElement* meEle_pt_MTD_6_Bkg_EB_;
  MonitorElement* meEle_pt_sim_MTD_6_Bkg_EB_;
  MonitorElement* meEle_eta_MTD_6_Bkg_EB_;
  MonitorElement* meEle_phi_MTD_6_Bkg_EB_;

  MonitorElement* meEle_pt_MTD_7_Bkg_EB_;
  MonitorElement* meEle_pt_sim_MTD_7_Bkg_EB_;
  MonitorElement* meEle_eta_MTD_7_Bkg_EB_;
  MonitorElement* meEle_phi_MTD_7_Bkg_EB_;

  MonitorElement* meEle_pt_noMTD_Bkg_EB_;
  MonitorElement* meEle_eta_noMTD_Bkg_EB_;
  MonitorElement* meEle_phi_noMTD_Bkg_EB_;

  MonitorElement* meEle_pt_MTD_2sigma_Bkg_EB_;
  MonitorElement* meEle_pt_sim_MTD_2sigma_Bkg_EB_;
  MonitorElement* meEle_eta_MTD_2sigma_Bkg_EB_;
  MonitorElement* meEle_phi_MTD_2sigma_Bkg_EB_;

  MonitorElement* meEle_pt_MTD_3sigma_Bkg_EB_;
  MonitorElement* meEle_pt_sim_MTD_3sigma_Bkg_EB_;
  MonitorElement* meEle_eta_MTD_3sigma_Bkg_EB_;
  MonitorElement* meEle_phi_MTD_3sigma_Bkg_EB_;

  MonitorElement* meEle_pt_MTD_4sigma_Bkg_EB_;
  MonitorElement* meEle_pt_sim_MTD_4sigma_Bkg_EB_;
  MonitorElement* meEle_eta_MTD_4sigma_Bkg_EB_;
  MonitorElement* meEle_phi_MTD_4sigma_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_Bkg_EE_;  // Adding histograms for endcap electrons (isolation stuff)
  MonitorElement* meEleISO_chIso_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_1_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_1_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_1_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_2_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_2_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_2_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_3_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_3_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_3_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_4_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_4_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_4_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_5_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_5_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_5_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_6_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_6_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_6_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_7_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_7_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_7_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_2sigma_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_2sigma_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_2sigma_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_3sigma_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_3sigma_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_3sigma_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_4sigma_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_4sigma_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_4sigma_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_1_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_1_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_1_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_2_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_2_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_2_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_3_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_3_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_3_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_4_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_4_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_4_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_5_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_5_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_5_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_6_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_6_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_6_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_7_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_7_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_7_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_2sigma_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_2sigma_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_2sigma_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_3sigma_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_3sigma_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_3sigma_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_sim_4sigma_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_sim_4sigma_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_sim_4sigma_Bkg_EE_;

  MonitorElement* meEle_pt_tot_Bkg_EE_;
  MonitorElement* meEle_pt_sim_tot_Bkg_EE_;
  MonitorElement* meEle_eta_tot_Bkg_EE_;
  MonitorElement* meEle_phi_tot_Bkg_EE_;

  MonitorElement* meEle_pt_MTD_1_Bkg_EE_;
  MonitorElement* meEle_pt_sim_MTD_1_Bkg_EE_;
  MonitorElement* meEle_eta_MTD_1_Bkg_EE_;
  MonitorElement* meEle_phi_MTD_1_Bkg_EE_;

  MonitorElement* meEle_pt_MTD_2_Bkg_EE_;
  MonitorElement* meEle_pt_sim_MTD_2_Bkg_EE_;
  MonitorElement* meEle_eta_MTD_2_Bkg_EE_;
  MonitorElement* meEle_phi_MTD_2_Bkg_EE_;

  MonitorElement* meEle_pt_MTD_3_Bkg_EE_;
  MonitorElement* meEle_pt_sim_MTD_3_Bkg_EE_;
  MonitorElement* meEle_eta_MTD_3_Bkg_EE_;
  MonitorElement* meEle_phi_MTD_3_Bkg_EE_;

  MonitorElement* meEle_pt_MTD_4_Bkg_EE_;
  MonitorElement* meEle_pt_sim_MTD_4_Bkg_EE_;
  MonitorElement* meEle_eta_MTD_4_Bkg_EE_;
  MonitorElement* meEle_phi_MTD_4_Bkg_EE_;

  MonitorElement* meEle_pt_MTD_5_Bkg_EE_;
  MonitorElement* meEle_pt_sim_MTD_5_Bkg_EE_;
  MonitorElement* meEle_eta_MTD_5_Bkg_EE_;
  MonitorElement* meEle_phi_MTD_5_Bkg_EE_;

  MonitorElement* meEle_pt_MTD_6_Bkg_EE_;
  MonitorElement* meEle_pt_sim_MTD_6_Bkg_EE_;
  MonitorElement* meEle_eta_MTD_6_Bkg_EE_;
  MonitorElement* meEle_phi_MTD_6_Bkg_EE_;

  MonitorElement* meEle_pt_MTD_7_Bkg_EE_;
  MonitorElement* meEle_pt_sim_MTD_7_Bkg_EE_;
  MonitorElement* meEle_eta_MTD_7_Bkg_EE_;
  MonitorElement* meEle_phi_MTD_7_Bkg_EE_;

  MonitorElement* meEle_pt_noMTD_Bkg_EE_;
  MonitorElement* meEle_eta_noMTD_Bkg_EE_;
  MonitorElement* meEle_phi_noMTD_Bkg_EE_;

  MonitorElement* meEle_pt_MTD_2sigma_Bkg_EE_;
  MonitorElement* meEle_pt_sim_MTD_2sigma_Bkg_EE_;
  MonitorElement* meEle_eta_MTD_2sigma_Bkg_EE_;
  MonitorElement* meEle_phi_MTD_2sigma_Bkg_EE_;

  MonitorElement* meEle_pt_MTD_3sigma_Bkg_EE_;
  MonitorElement* meEle_pt_sim_MTD_3sigma_Bkg_EE_;
  MonitorElement* meEle_eta_MTD_3sigma_Bkg_EE_;
  MonitorElement* meEle_phi_MTD_3sigma_Bkg_EE_;

  MonitorElement* meEle_pt_MTD_4sigma_Bkg_EE_;
  MonitorElement* meEle_pt_sim_MTD_4sigma_Bkg_EE_;
  MonitorElement* meEle_eta_MTD_4sigma_Bkg_EE_;
  MonitorElement* meEle_phi_MTD_4sigma_Bkg_EE_;
  // Background histograms end

  // histograms for dt distributions in pT/eta bins

  MonitorElement* meEle_dt_general_pT_1;
  MonitorElement* meEle_dt_general_pT_2;
  MonitorElement* meEle_dt_general_pT_3;
  MonitorElement* meEle_dt_general_pT_4;
  MonitorElement* meEle_dt_general_pT_5;
  MonitorElement* meEle_dt_general_pT_6;
  MonitorElement* meEle_dt_general_pT_7;
  MonitorElement* meEle_dt_general_pT_8;
  MonitorElement* meEle_dt_general_pT_9;

  MonitorElement* meEle_dtSignif_general_pT_1;
  MonitorElement* meEle_dtSignif_general_pT_2;
  MonitorElement* meEle_dtSignif_general_pT_3;
  MonitorElement* meEle_dtSignif_general_pT_4;
  MonitorElement* meEle_dtSignif_general_pT_5;
  MonitorElement* meEle_dtSignif_general_pT_6;
  MonitorElement* meEle_dtSignif_general_pT_7;
  MonitorElement* meEle_dtSignif_general_pT_8;
  MonitorElement* meEle_dtSignif_general_pT_9;

  MonitorElement* meEle_dt_general_eta_1;
  MonitorElement* meEle_dt_general_eta_2;
  MonitorElement* meEle_dt_general_eta_3;
  MonitorElement* meEle_dt_general_eta_4;
  MonitorElement* meEle_dt_general_eta_5;

  MonitorElement* meEle_dtSignif_general_eta_1;
  MonitorElement* meEle_dtSignif_general_eta_2;
  MonitorElement* meEle_dtSignif_general_eta_3;
  MonitorElement* meEle_dtSignif_general_eta_4;
  MonitorElement* meEle_dtSignif_general_eta_5;

  // promt part for histogram vectors
  std::vector<MonitorElement*> Ntracks_EB_list_Sig;
  std::vector<MonitorElement*> ch_iso_EB_list_Sig;
  std::vector<MonitorElement*> rel_ch_iso_EB_list_Sig;

  std::vector<MonitorElement*> Ntracks_EE_list_Sig;
  std::vector<MonitorElement*> ch_iso_EE_list_Sig;
  std::vector<MonitorElement*> rel_ch_iso_EE_list_Sig;

  std::vector<MonitorElement*> Ntracks_sim_EB_list_Sig;
  std::vector<MonitorElement*> ch_iso_sim_EB_list_Sig;
  std::vector<MonitorElement*> rel_ch_iso_sim_EB_list_Sig;

  std::vector<MonitorElement*> Ntracks_sim_EE_list_Sig;
  std::vector<MonitorElement*> ch_iso_sim_EE_list_Sig;
  std::vector<MonitorElement*> rel_ch_iso_sim_EE_list_Sig;

  std::vector<MonitorElement*> Ele_pT_MTD_EB_list_Sig;
  std::vector<MonitorElement*> Ele_pT_sim_MTD_EB_list_Sig;
  std::vector<MonitorElement*> Ele_eta_MTD_EB_list_Sig;
  std::vector<MonitorElement*> Ele_phi_MTD_EB_list_Sig;

  std::vector<MonitorElement*> Ele_pT_MTD_EE_list_Sig;
  std::vector<MonitorElement*> Ele_pT_sim_MTD_EE_list_Sig;
  std::vector<MonitorElement*> Ele_eta_MTD_EE_list_Sig;
  std::vector<MonitorElement*> Ele_phi_MTD_EE_list_Sig;

  std::vector<MonitorElement*> Ntracks_EB_list_Significance_Sig;
  std::vector<MonitorElement*> ch_iso_EB_list_Significance_Sig;
  std::vector<MonitorElement*> rel_ch_iso_EB_list_Significance_Sig;

  std::vector<MonitorElement*> Ntracks_EE_list_Significance_Sig;
  std::vector<MonitorElement*> ch_iso_EE_list_Significance_Sig;
  std::vector<MonitorElement*> rel_ch_iso_EE_list_Significance_Sig;

  std::vector<MonitorElement*> Ntracks_sim_EB_list_Significance_Sig;
  std::vector<MonitorElement*> ch_iso_sim_EB_list_Significance_Sig;
  std::vector<MonitorElement*> rel_ch_iso_sim_EB_list_Significance_Sig;

  std::vector<MonitorElement*> Ntracks_sim_EE_list_Significance_Sig;
  std::vector<MonitorElement*> ch_iso_sim_EE_list_Significance_Sig;
  std::vector<MonitorElement*> rel_ch_iso_sim_EE_list_Significance_Sig;

  std::vector<MonitorElement*> Ele_pT_MTD_EB_list_Significance_Sig;
  std::vector<MonitorElement*> Ele_pT_sim_MTD_EB_list_Significance_Sig;
  std::vector<MonitorElement*> Ele_eta_MTD_EB_list_Significance_Sig;
  std::vector<MonitorElement*> Ele_phi_MTD_EB_list_Significance_Sig;

  std::vector<MonitorElement*> Ele_pT_MTD_EE_list_Significance_Sig;
  std::vector<MonitorElement*> Ele_pT_sim_MTD_EE_list_Significance_Sig;
  std::vector<MonitorElement*> Ele_eta_MTD_EE_list_Significance_Sig;
  std::vector<MonitorElement*> Ele_phi_MTD_EE_list_Significance_Sig;

  // Non-promt part for histogram vectors
  std::vector<MonitorElement*> Ntracks_EB_list_Bkg;
  std::vector<MonitorElement*> ch_iso_EB_list_Bkg;
  std::vector<MonitorElement*> rel_ch_iso_EB_list_Bkg;

  std::vector<MonitorElement*> Ntracks_EE_list_Bkg;
  std::vector<MonitorElement*> ch_iso_EE_list_Bkg;
  std::vector<MonitorElement*> rel_ch_iso_EE_list_Bkg;

  std::vector<MonitorElement*> Ntracks_sim_EB_list_Bkg;
  std::vector<MonitorElement*> ch_iso_sim_EB_list_Bkg;
  std::vector<MonitorElement*> rel_ch_iso_sim_EB_list_Bkg;

  std::vector<MonitorElement*> Ntracks_sim_EE_list_Bkg;
  std::vector<MonitorElement*> ch_iso_sim_EE_list_Bkg;
  std::vector<MonitorElement*> rel_ch_iso_sim_EE_list_Bkg;

  std::vector<MonitorElement*> Ele_pT_MTD_EB_list_Bkg;
  std::vector<MonitorElement*> Ele_pT_sim_MTD_EB_list_Bkg;
  std::vector<MonitorElement*> Ele_eta_MTD_EB_list_Bkg;
  std::vector<MonitorElement*> Ele_phi_MTD_EB_list_Bkg;

  std::vector<MonitorElement*> Ele_pT_MTD_EE_list_Bkg;
  std::vector<MonitorElement*> Ele_pT_sim_MTD_EE_list_Bkg;
  std::vector<MonitorElement*> Ele_eta_MTD_EE_list_Bkg;
  std::vector<MonitorElement*> Ele_phi_MTD_EE_list_Bkg;

  std::vector<MonitorElement*> Ntracks_EB_list_Significance_Bkg;
  std::vector<MonitorElement*> ch_iso_EB_list_Significance_Bkg;
  std::vector<MonitorElement*> rel_ch_iso_EB_list_Significance_Bkg;

  std::vector<MonitorElement*> Ntracks_EE_list_Significance_Bkg;
  std::vector<MonitorElement*> ch_iso_EE_list_Significance_Bkg;
  std::vector<MonitorElement*> rel_ch_iso_EE_list_Significance_Bkg;

  std::vector<MonitorElement*> Ntracks_sim_EB_list_Significance_Bkg;
  std::vector<MonitorElement*> ch_iso_sim_EB_list_Significance_Bkg;
  std::vector<MonitorElement*> rel_ch_iso_sim_EB_list_Significance_Bkg;

  std::vector<MonitorElement*> Ntracks_sim_EE_list_Significance_Bkg;
  std::vector<MonitorElement*> ch_iso_sim_EE_list_Significance_Bkg;
  std::vector<MonitorElement*> rel_ch_iso_sim_EE_list_Significance_Bkg;

  std::vector<MonitorElement*> Ele_pT_MTD_EB_list_Significance_Bkg;
  std::vector<MonitorElement*> Ele_pT_sim_MTD_EB_list_Significance_Bkg;
  std::vector<MonitorElement*> Ele_eta_MTD_EB_list_Significance_Bkg;
  std::vector<MonitorElement*> Ele_phi_MTD_EB_list_Significance_Bkg;

  std::vector<MonitorElement*> Ele_pT_MTD_EE_list_Significance_Bkg;
  std::vector<MonitorElement*> Ele_pT_sim_MTD_EE_list_Significance_Bkg;
  std::vector<MonitorElement*> Ele_eta_MTD_EE_list_Significance_Bkg;
  std::vector<MonitorElement*> Ele_phi_MTD_EE_list_Significance_Bkg;

  // dt distribution part for histogram vectors
  std::vector<MonitorElement*> general_pT_list;
  std::vector<MonitorElement*> general_eta_list;

  std::vector<MonitorElement*> general_pT_Signif_list;
  std::vector<MonitorElement*> general_eta_Signif_list;
};

// ------------ constructor and destructor --------------
MtdEleIsoValidation::MtdEleIsoValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      trackMinPt_(iConfig.getParameter<double>("trackMinimumPt")),
      trackMinEta_(iConfig.getParameter<double>("trackMinimumEta")),
      trackMaxEta_(iConfig.getParameter<double>("trackMaximumEta")),
      rel_iso_cut_(iConfig.getParameter<double>("rel_iso_cut")),
      track_match_PV_(iConfig.getParameter<bool>("optionTrackMatchToPV")),
      dt_sig_vtx_(iConfig.getParameter<bool>("option_dtToPV")),
      dt_sig_track_(iConfig.getParameter<bool>("option_dtToTrack")),
      dt_distributions_(iConfig.getParameter<bool>("option_dtDistributions")) {
  //Ntracks_EB_list_Sig(iConfig.getParameter<std::vector<MonitorElement*>>("Ntracks_EB_list_Sig_test")) { // Example that does not work, but does work for double type

  GenRecTrackToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTagG"));
  RecTrackToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTagT"));
  RecVertexToken_ =
      consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("inputTag_vtx"));  // Vtx 4D collection

  GsfElectronToken_EB_ = consumes<reco::GsfElectronCollection>(
      iConfig.getParameter<edm::InputTag>("inputEle_EB"));  // Barrel electron collection input/token
  GsfElectronToken_EE_ = consumes<reco::GsfElectronCollection>(
      iConfig.getParameter<edm::InputTag>("inputEle_EE"));  // Endcap electron collection input/token
  GenParticleToken_ = consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("inputGenP"));

  HepMCProductToken_ = consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("inputTagH"));
  trackAssocToken_ = consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("trackAssocSrc"));
  pathLengthToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("pathLengthSrc"));
  tmtdToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("tmtd"));
  SigmatmtdToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmatmtd"));
  t0SrcToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("t0Src"));
  Sigmat0SrcToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmat0Src"));
  t0PidToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("t0PID"));
  Sigmat0PidToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmat0PID"));
  t0SafePidToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("t0SafePID"));
  Sigmat0SafePidToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmat0SafePID"));
  trackMVAQualToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("trackMVAQual"));

  trackingParticleCollectionToken_ =
      consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("SimTag"));  // From Aurora
  recoToSimAssociationToken_ =
      consumes<reco::RecoToSimCollection>(iConfig.getParameter<edm::InputTag>("TPtoRecoTrackAssoc"));  // From Aurora

  particleTableToken_ = esConsumes<HepPDT::ParticleDataTable, edm::DefaultRecord>();
}

MtdEleIsoValidation::~MtdEleIsoValidation() {}

// ------------ method called for each event  ------------
void MtdEleIsoValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace geant_units::operators;
  using namespace std;

  //auto GenRecTrackHandle = makeValid(iEvent.getHandle(GenRecTrackToken_));
  //edm::Handle<reco::TrackCollection> GenRecTrackHandle = iEvent.getHandle(GenRecTrackToken_);
  auto GenRecTrackHandle = iEvent.getHandle(GenRecTrackToken_);

  //auto RecVertexHandle_ = makeValid(iEvent.getHandle(RecVertexToken_));
  auto VertexHandle_ = iEvent.getHandle(RecVertexToken_);

  std::vector<reco::Vertex> vertices = *VertexHandle_;

  const auto& t0Pid = iEvent.get(t0PidToken_);
  const auto& Sigmat0Pid = iEvent.get(Sigmat0PidToken_);
  const auto& mtdQualMVA = iEvent.get(trackMVAQualToken_);
  //const auto& tMtd = iEvent.get(tmtdToken_);
  //const auto& SigmatMtd = iEvent.get(SigmatmtdToken_);
  //const auto& t0Src = iEvent.get(t0SrcToken_);
  //const auto& Sigmat0Src = iEvent.get(Sigmat0SrcToken_);
  //const auto& t0Safe = iEvent.get(t0SafePidToken_);
  //const auto& Sigmat0Safe = iEvent.get(Sigmat0SafePidToken_);
  //const auto& trackAssoc = iEvent.get(trackAssocToken_);
  //const auto& pathLength = iEvent.get(pathLengthToken_);

  auto eleHandle_EB = makeValid(iEvent.getHandle(GsfElectronToken_EB_));
  reco::GsfElectronCollection eleColl_EB = *(eleHandle_EB.product());

  auto eleHandle_EE = makeValid(iEvent.getHandle(GsfElectronToken_EE_));
  reco::GsfElectronCollection eleColl_EE = *(eleHandle_EE.product());

  auto GenPartHandle =
      makeValid(iEvent.getHandle(GenParticleToken_));  // Aurora uses getByToken, result should be the same
  reco::GenParticleCollection GenPartColl = *(GenPartHandle.product());

  auto recoToSimH = makeValid(iEvent.getHandle(recoToSimAssociationToken_));  // From Aurora
  const reco::RecoToSimCollection* r2s_ = recoToSimH.product();               // From Aurora

  // Creating combined electron collection

  std::vector<reco::GsfElectron> localEleCollection;
  localEleCollection.reserve(
      eleColl_EB.size() + eleColl_EE.size());  // From Aurora, added for saving memory and running time (reserve memory)

  for (const auto& ele_EB : eleColl_EB) {
    if (ele_EB.isEB()) {
      localEleCollection.emplace_back(ele_EB);  // changed to emplace_back instead of push_back (From Aurora)
    }
  }

  for (const auto& ele_EE : eleColl_EE) {
    if (ele_EE.isEE()) {
      localEleCollection.emplace_back(ele_EE);  // changed to emplace_back instead of push_back (From Aurora)
    }
  }
  localEleCollection.shrink_to_fit();  // From Aurora, optimize vector momery size

  // timing cut type check (if both true, only check cut wrt. vtx not electron track)
  if (dt_sig_vtx_ && dt_sig_track_) {
    dt_sig_track_ = false;
    std::cout
        << "Both timing cut types chosen!!! Fix this at the configuration setup. Default timing wrt. vtx chosen!!!!"
        << std::endl;
  }

  // Selecting the PV from 3D and 4D vertex collections

  reco::Vertex Vtx_chosen;

  // This part has to be included, because in ~1% of the events, the "good" vertex is the 1st one not the 0th one in the collection
  for (int iVtx = 0; iVtx < (int)vertices.size(); iVtx++) {
    const reco::Vertex& vertex = vertices.at(iVtx);

    if (!vertex.isFake() && vertex.ndof() >= 4) {
      Vtx_chosen = vertex;
      break;
    }
  }
  // Vertex selection ends

  // auto isPrompt = [](int pdg) {
  //   pdg = std::abs(pdg);
  //   return (pdg == 23 or pdg == 24 or pdg==15 or pdg==11); // some electrons are mothers to themselves?
  // };

  //for (const auto& ele : eleColl_EB){
  for (const auto& ele : localEleCollection) {
    bool ele_Promt = false;

    float ele_track_source_dz = fabs(ele.gsfTrack()->dz(Vtx_chosen.position()));
    float ele_track_source_dxy = fabs(ele.gsfTrack()->dxy(Vtx_chosen.position()));

    const reco::TrackRef ele_TrkRef = ele.core()->ctfTrack();
    double tsim = -1.;
    double ele_sim_pt = -1.;

    if (ele.pt() > 10 && fabs(ele.eta()) < 2.4 && ele_track_source_dz < max_dz_vtx_cut &&
        ele_track_source_dxy < max_dxy_vtx_cut) {  // selecting "good" RECO electrons

      // Do the RECO-GEN match through RECOtoSIMcollection link (From Aurora)

      const reco::TrackBaseRef trkrefb(ele_TrkRef);

      auto found = r2s_->find(trkrefb);  // find the track link?
      if (found != r2s_->end()) {
        const auto& tp = (found->val)[0];
        tsim = (tp.first)->parentVertex()->position().t() * 1e9;
        ele_sim_pt = (tp.first)->pt();
        // check that the genParticle vector is not empty
        if (((found->val)[0]).first->status() != -99) {
          const auto genParticle = *(tp.first->genParticles()[0]);
          // check if prompt (not from hadron, muon, or tau decay) and final state
          // or if is a direct decay product of a prompt tau and is final state
          //if ((genParticle.isPromptFinalState() or genParticle.isDirectPromptTauDecayProductFinalState()) and isPrompt(genParticle.mother()->pdgId())) {
          if ((genParticle.isPromptFinalState() or genParticle.isDirectPromptTauDecayProductFinalState()) and
              pdgCheck(genParticle.mother()->pdgId())) {
            ele_Promt = true;
          }
        }
      }

      meEle_test_check_->Fill(tsim);

      // old matching
      // math::XYZVector EleMomentum = ele.momentum();

      // for (const auto& genParticle : GenPartColl) {
      //   math::XYZVector GenPartMomentum = genParticle.momentum();
      //   double dr_match = reco::deltaR(GenPartMomentum, EleMomentum);

      //   if (((genParticle.pdgId() == 11 && ele.charge() == -1) || (genParticle.pdgId() == -11 && ele.charge() == 1)) &&
      //       dr_match < 0.10) {  // dR match check and charge match check

      //     if (genParticle.mother()->pdgId() == 23 || fabs(genParticle.mother()->pdgId()) == 24 ||
      //         fabs(genParticle.mother()->pdgId()) ==
      //             15) {  // check if ele mother is Z,W or tau (W->tau->ele decays, as these still should be quite isolated)
      //       ele_Promt = true;  // ele is defined as promt
      //     }
      //     break;

      //   } else {
      //     continue;
      //   }
      // }
    } else {
      continue;
    }

    math::XYZVector EleSigTrackMomentumAtVtx = ele.gsfTrack()->momentum();  // not sure if correct, but works
    double EleSigTrackEtaAtVtx = ele.gsfTrack()->eta();

    double ele_sigTrkTime = -1;  // electron signal track MTD information
    double ele_sigTrkTimeErr = -1;
    double ele_sigTrkMtdMva = -1;

    // Removed track dR matching as we have track references (From Aurora)
    if (ele_TrkRef.isNonnull()) {  // if we found a track match, we add MTD timing information for it

      bool Barrel_ele = ele.isEB();  // for track pT/dz cuts (Different for EB and EE in TDR)

      float min_pt_cut = Barrel_ele ? min_pt_cut_EB : min_pt_cut_EE;
      float max_dz_cut = Barrel_ele ? max_dz_cut_EB : max_dz_cut_EE;

      ele_sigTrkTime = t0Pid[ele_TrkRef];
      ele_sigTrkTimeErr = Sigmat0Pid[ele_TrkRef];
      ele_sigTrkMtdMva = mtdQualMVA[ele_TrkRef];
      ele_sigTrkTimeErr = (ele_sigTrkMtdMva > min_track_mtd_mva_cut) ? ele_sigTrkTimeErr : -1;

      meEle_avg_error_SigTrk_check_->Fill(ele_sigTrkTimeErr);

      if (ele_Promt) {
        // For signal (promt)
        if (Barrel_ele) {
          meEle_pt_tot_Sig_EB_->Fill(ele.pt());  // All selected electron information for efficiency plots later
          meEle_pt_sim_tot_Sig_EB_->Fill(ele_sim_pt);
          meEle_eta_tot_Sig_EB_->Fill(ele.eta());
          meEle_phi_tot_Sig_EB_->Fill(ele.phi());
        } else {
          meEle_pt_tot_Sig_EE_->Fill(ele.pt());  // All selected electron information for efficiency plots later
          meEle_pt_sim_tot_Sig_EE_->Fill(ele_sim_pt);
          meEle_eta_tot_Sig_EE_->Fill(ele.eta());
          meEle_phi_tot_Sig_EE_->Fill(ele.phi());
        }
      } else {
        // For background (non-promt)
        if (Barrel_ele) {
          meEle_pt_tot_Bkg_EB_->Fill(ele.pt());
          meEle_pt_sim_tot_Bkg_EB_->Fill(ele_sim_pt);
          meEle_eta_tot_Bkg_EB_->Fill(ele.eta());
          meEle_phi_tot_Bkg_EB_->Fill(ele.phi());
        } else {
          meEle_pt_tot_Bkg_EE_->Fill(ele.pt());
          meEle_pt_sim_tot_Bkg_EE_->Fill(ele_sim_pt);
          meEle_eta_tot_Bkg_EE_->Fill(ele.eta());
          meEle_phi_tot_Bkg_EE_->Fill(ele.phi());
        }
      }

      int N_tracks_noMTD = 0;  // values for no MTD case
      double pT_sum_noMTD = 0;
      double rel_pT_sum_noMTD = 0;
      std::vector<int> N_tracks_MTD{0, 0, 0, 0, 0, 0, 0};  // values for MTD case - 7 timing cuts // RECO
      std::vector<double> pT_sum_MTD{0, 0, 0, 0, 0, 0, 0};
      std::vector<double> rel_pT_sum_MTD{0, 0, 0, 0, 0, 0, 0};

      std::vector<int> N_tracks_sim_MTD{0, 0, 0, 0, 0, 0, 0};  // SIM
      std::vector<double> pT_sum_sim_MTD{0, 0, 0, 0, 0, 0, 0};
      std::vector<double> rel_pT_sum_sim_MTD{0, 0, 0, 0, 0, 0, 0};

      std::vector<int> N_tracks_MTD_significance{0, 0, 0};  // values for MTD case - 3 significance cuts // RECO
      std::vector<double> pT_sum_MTD_significance{0, 0, 0};
      std::vector<double> rel_pT_sum_MTD_significance{0, 0, 0};

      std::vector<int> N_tracks_sim_MTD_significance{0, 0, 0};  // SIM
      std::vector<double> pT_sum_sim_MTD_significance{0, 0, 0};
      std::vector<double> rel_pT_sum_sim_MTD_significance{0, 0, 0};

      int general_index = 0;
      for (const auto& trackGen : *GenRecTrackHandle) {
        const reco::TrackRef trackref_general(GenRecTrackHandle, general_index);
        general_index++;

        if (trackref_general == ele_TrkRef)  // Skip electron track (From Aurora)
          continue;

        if (trackGen.pt() < min_pt_cut) {  // track pT cut
          continue;
        }

        if (fabs(trackGen.vz() - ele.gsfTrack()->vz()) > max_dz_cut) {  // general track vs signal track dz cut
          continue;
        }

        if (track_match_PV_) {
          if (Vtx_chosen.trackWeight(trackref_general) < 0.5) {  // cut for general track matching to PV
            continue;
          }
        }

        double dr_check = reco::deltaR(trackGen.momentum(), EleSigTrackMomentumAtVtx);
        double deta = fabs(trackGen.eta() - EleSigTrackEtaAtVtx);

        if (dr_check > min_dR_cut && dr_check < max_dR_cut &&
            deta > min_strip_cut) {  // checking if the track is inside isolation cone

          ++N_tracks_noMTD;
          pT_sum_noMTD += trackGen.pt();
        } else {
          continue;
        }

        // checking the MTD timing cuts

        // Track SIM information (From Aurora)
        const reco::TrackBaseRef trkrefBase(trackref_general);

        auto TPmatched = r2s_->find(trkrefBase);
        double tsim_trk = -1.;
        double trk_ptSim = -1.;
        //int genMatched = 0;
        if (TPmatched != r2s_->end()) {
          // reco track matched to a TP
          const auto& tp = (TPmatched->val)[0];
          tsim_trk = (tp.first)->parentVertex()->position().t() * 1e9;
          trk_ptSim = (tp.first)->pt();
          //     // check that the genParticle vector is not empty
          //       if (((TPmatched->val)[0]).first->status() != -99) {
          //     genMatched = 1;
          //     }
        }

        //   meTrk_genMatch_check_->Fill(genMatched);

        meEle_test_check_->Fill(trk_ptSim);

        double TrkMTDTime = t0Pid[trackref_general];  // MTD timing info for particular track fron general tracks
        double TrkMTDTimeErr = Sigmat0Pid[trackref_general];
        double TrkMTDMva = mtdQualMVA[trackref_general];
        TrkMTDTimeErr = (TrkMTDMva > min_track_mtd_mva_cut) ? TrkMTDTimeErr : -1;  // track MTD MVA cut/check

        meEle_avg_error_PUTrk_check_->Fill(TrkMTDTimeErr);

        if (dt_sig_track_) {
          double dt_sigTrk = 0;  // dt regular track vs signal track (absolute value)
          double dt_sigTrk_signif = 0;

          double dt_sim_sigTrk = 0;  // for SIM case
          double dt_sim_sigTrk_signif = 0;

          // SIM CASE trk-SigTrk STARTS
          if (fabs(tsim_trk) > 0 && fabs(tsim) > 0 && trk_ptSim > 0) {
            dt_sim_sigTrk = fabs(tsim_trk - tsim);
            dt_sim_sigTrk_signif =
                dt_sim_sigTrk /
                std::sqrt(avg_sim_PUtrack_t_err * avg_sim_PUtrack_t_err +
                          avg_sim_sigTrk_t_err * avg_sim_sigTrk_t_err);  // FIX ERRORS FOR SIGNIFICANCE IN SIM CASE

            // absolute timing cuts
            for (long unsigned int i = 0; i < N_tracks_sim_MTD.size(); i++) {
              if (dt_sim_sigTrk < max_dt_track_cut[i]) {
                N_tracks_sim_MTD[i] = N_tracks_sim_MTD[i] + 1;
                pT_sum_sim_MTD[i] = pT_sum_sim_MTD[i] + trk_ptSim;
              }
            }
            // significance cuts
            for (long unsigned int i = 0; i < N_tracks_sim_MTD_significance.size(); i++) {
              if (dt_sim_sigTrk_signif < max_dt_significance_cut[i]) {
                N_tracks_sim_MTD_significance[i] = N_tracks_sim_MTD_significance[i] + 1;
                pT_sum_sim_MTD_significance[i] = pT_sum_sim_MTD_significance[i] + trk_ptSim;
              }
            }

          } else {  // if there is no error for MTD information, we count the MTD isolation case same as noMTD
            for (long unsigned int i = 0; i < N_tracks_sim_MTD.size(); i++) {
              N_tracks_sim_MTD[i] = N_tracks_sim_MTD[i] + 1;
              pT_sum_sim_MTD[i] = pT_sum_sim_MTD[i] + trk_ptSim;
            }

            for (long unsigned int i = 0; i < N_tracks_sim_MTD_significance.size(); i++) {
              N_tracks_sim_MTD_significance[i] = N_tracks_sim_MTD_significance[i] + 1;
              pT_sum_sim_MTD_significance[i] = pT_sum_sim_MTD_significance[i] + trk_ptSim;
            }
          }
          // SIM CASE trk-SigTrk ENDS

          // Regular RECO MTD check
          if (TrkMTDTimeErr > 0 && ele_sigTrkTimeErr > 0) {
            dt_sigTrk = fabs(TrkMTDTime - ele_sigTrkTime);
            dt_sigTrk_signif =
                dt_sigTrk / std::sqrt(TrkMTDTimeErr * TrkMTDTimeErr + ele_sigTrkTimeErr * ele_sigTrkTimeErr);

            meEle_no_dt_check_->Fill(1);
            // absolute timing cuts
            for (long unsigned int i = 0; i < N_tracks_MTD.size(); i++) {
              if (dt_sigTrk < max_dt_track_cut[i]) {
                N_tracks_MTD[i] = N_tracks_MTD[i] + 1;
                pT_sum_MTD[i] = pT_sum_MTD[i] + trackGen.pt();
              }
            }
            // significance cuts
            for (long unsigned int i = 0; i < N_tracks_MTD_significance.size(); i++) {
              if (dt_sigTrk_signif < max_dt_significance_cut[i]) {
                N_tracks_MTD_significance[i] = N_tracks_MTD_significance[i] + 1;
                pT_sum_MTD_significance[i] = pT_sum_MTD_significance[i] + trackGen.pt();
              }
            }

          } else {  // if there is no error for MTD information, we count the MTD isolation case same as noMTD
            for (long unsigned int i = 0; i < N_tracks_MTD.size(); i++) {
              N_tracks_MTD[i] = N_tracks_MTD[i] + 1;          // N_tracks_noMTD
              pT_sum_MTD[i] = pT_sum_MTD[i] + trackGen.pt();  // pT sum
            }

            for (long unsigned int i = 0; i < N_tracks_MTD_significance.size(); i++) {
              N_tracks_MTD_significance[i] = N_tracks_MTD_significance[i] + 1;          // N_tracks_noMTD
              pT_sum_MTD_significance[i] = pT_sum_MTD_significance[i] + trackGen.pt();  // pT sum
            }
            meEle_no_dt_check_->Fill(0);
          }

          if (dt_distributions_) {
            for (long unsigned int i = 0; i < (pT_bins_dt_distrb.size() - 1); i++) {
              //stuff general pT
              if (ele.pt() > pT_bins_dt_distrb[i] && ele.pt() < pT_bins_dt_distrb[i + 1]) {
                general_pT_list[i]->Fill(dt_sigTrk);
                general_pT_Signif_list[i]->Fill(dt_sigTrk_signif);
              }
            }

            for (long unsigned int i = 0; i < (eta_bins_dt_distrib.size() - 1); i++) {
              //stuff general eta
              if (fabs(ele.eta()) > eta_bins_dt_distrib[i] && fabs(ele.eta()) < eta_bins_dt_distrib[i + 1]) {
                general_eta_list[i]->Fill(dt_sigTrk);
                general_eta_Signif_list[i]->Fill(dt_sigTrk_signif);
              }
            }
          }  // End of optional dt distributions plots
        }

        if (dt_sig_vtx_) {
          double dt_vtx = 0;  // dt regular track vs vtx
          double dt_vtx_signif = 0;

          double dt_sim_vtx = 0;  // dt regular track vs vtx
          double dt_sim_vtx_signif = 0;

          //SIM CASE trk-vertex STARTS
          if (fabs(tsim_trk) > 0 && Vtx_chosen.tError() > 0 && trk_ptSim > 0) {
            dt_sim_vtx = fabs(tsim_trk - Vtx_chosen.t());
            dt_sim_vtx_signif = dt_sim_vtx / std::sqrt(avg_sim_PUtrack_t_err * avg_sim_PUtrack_t_err +
                                                       Vtx_chosen.tError() * Vtx_chosen.tError());
            // absolute timing cuts
            for (long unsigned int i = 0; i < N_tracks_sim_MTD.size(); i++) {
              if (dt_sim_vtx < max_dt_vtx_cut[i]) {
                N_tracks_sim_MTD[i] = N_tracks_sim_MTD[i] + 1;
                pT_sum_sim_MTD[i] = pT_sum_sim_MTD[i] + trk_ptSim;
              }
            }
            // significance timing cuts
            for (long unsigned int i = 0; i < N_tracks_sim_MTD_significance.size(); i++) {
              if (dt_sim_vtx_signif < max_dt_significance_cut[i]) {
                N_tracks_sim_MTD_significance[i] = N_tracks_sim_MTD_significance[i] + 1;
                pT_sum_sim_MTD_significance[i] = pT_sum_sim_MTD_significance[i] + trk_ptSim;
              }
            }

          } else {
            for (long unsigned int i = 0; i < N_tracks_sim_MTD.size(); i++) {
              N_tracks_sim_MTD[i] = N_tracks_sim_MTD[i] + 1;      // N_tracks_noMTD
              pT_sum_sim_MTD[i] = pT_sum_sim_MTD[i] + trk_ptSim;  // pT_sum_noMTD
            }

            for (long unsigned int i = 0; i < N_tracks_sim_MTD_significance.size(); i++) {
              N_tracks_sim_MTD_significance[i] = N_tracks_sim_MTD_significance[i] + 1;      // N_tracks_noMTD
              pT_sum_sim_MTD_significance[i] = pT_sum_sim_MTD_significance[i] + trk_ptSim;  // pT sum
            }
          }  // SIM CASE trk-vertex ENDS

          //Regular RECO MTD case
          if (TrkMTDTimeErr > 0 && Vtx_chosen.tError() > 0) {
            dt_vtx = fabs(TrkMTDTime - Vtx_chosen.t());
            dt_vtx_signif =
                dt_vtx / std::sqrt(TrkMTDTimeErr * TrkMTDTimeErr + Vtx_chosen.tError() * Vtx_chosen.tError());

            meEle_no_dt_check_->Fill(1);
            meEle_avg_error_vtx_check_->Fill(Vtx_chosen.tError());

            // absolute timing cuts
            for (long unsigned int i = 0; i < N_tracks_MTD.size(); i++) {
              if (dt_vtx < max_dt_vtx_cut[i]) {
                N_tracks_MTD[i] = N_tracks_MTD[i] + 1;
                pT_sum_MTD[i] = pT_sum_MTD[i] + trackGen.pt();
              }
            }
            // significance timing cuts
            for (long unsigned int i = 0; i < N_tracks_MTD_significance.size(); i++) {
              if (dt_vtx_signif < max_dt_significance_cut[i]) {
                N_tracks_MTD_significance[i] = N_tracks_MTD_significance[i] + 1;
                pT_sum_MTD_significance[i] = pT_sum_MTD_significance[i] + trackGen.pt();
              }
            }

          } else {
            for (long unsigned int i = 0; i < N_tracks_MTD.size(); i++) {
              N_tracks_MTD[i] = N_tracks_MTD[i] + 1;          // N_tracks_noMTD
              pT_sum_MTD[i] = pT_sum_MTD[i] + trackGen.pt();  // pT_sum_noMTD
            }

            for (long unsigned int i = 0; i < N_tracks_MTD_significance.size(); i++) {
              N_tracks_MTD_significance[i] = N_tracks_MTD_significance[i] + 1;          // N_tracks_noMTD
              pT_sum_MTD_significance[i] = pT_sum_MTD_significance[i] + trackGen.pt();  // pT sum
            }
            meEle_no_dt_check_->Fill(0);
          }

          /// Optional dt distribution plots
          if (dt_distributions_) {
            for (long unsigned int i = 0; i < (pT_bins_dt_distrb.size() - 1); i++) {
              //stuff general pT
              if (ele.pt() > pT_bins_dt_distrb[i] && ele.pt() < pT_bins_dt_distrb[i + 1]) {
                general_pT_list[i]->Fill(dt_vtx);
                general_pT_Signif_list[i]->Fill(dt_vtx_signif);
              }
            }

            for (long unsigned int i = 0; i < (eta_bins_dt_distrib.size() - 1); i++) {
              //stuff general eta
              if (fabs(ele.eta()) > eta_bins_dt_distrib[i] && fabs(ele.eta()) < eta_bins_dt_distrib[i + 1]) {
                general_eta_list[i]->Fill(dt_vtx);
                general_eta_Signif_list[i]->Fill(dt_vtx_signif);
              }
            }
          }  // End of optional dt distributions plots
        }
      }

      rel_pT_sum_noMTD = pT_sum_noMTD / ele.gsfTrack()->pt();  // rel_ch_iso calculation
      for (long unsigned int i = 0; i < N_tracks_MTD.size(); i++) {
        rel_pT_sum_MTD[i] = pT_sum_MTD[i] / ele.gsfTrack()->pt();
        rel_pT_sum_sim_MTD[i] = pT_sum_sim_MTD[i] / ele_sim_pt;
      }

      for (long unsigned int i = 0; i < N_tracks_MTD_significance.size(); i++) {
        rel_pT_sum_MTD_significance[i] = pT_sum_MTD_significance[i] / ele.gsfTrack()->pt();
        rel_pT_sum_sim_MTD_significance[i] = pT_sum_sim_MTD_significance[i] / ele_sim_pt;
      }

      if (ele_Promt) {  // promt part
        if (Barrel_ele) {
          meEleISO_Ntracks_Sig_EB_->Fill(N_tracks_noMTD);  // Filling hists for Ntraks and chIso sums for noMTD case //
          meEleISO_chIso_Sig_EB_->Fill(pT_sum_noMTD);
          meEleISO_rel_chIso_Sig_EB_->Fill(rel_pT_sum_noMTD);

          for (long unsigned int j = 0; j < Ntracks_EB_list_Sig.size(); j++) {
            Ntracks_EB_list_Sig[j]->Fill(N_tracks_MTD[j]);
            ch_iso_EB_list_Sig[j]->Fill(pT_sum_MTD[j]);
            rel_ch_iso_EB_list_Sig[j]->Fill(rel_pT_sum_MTD[j]);

            Ntracks_sim_EB_list_Sig[j]->Fill(N_tracks_sim_MTD[j]);
            ch_iso_sim_EB_list_Sig[j]->Fill(pT_sum_sim_MTD[j]);
            rel_ch_iso_sim_EB_list_Sig[j]->Fill(rel_pT_sum_sim_MTD[j]);
          }

          for (long unsigned int j = 0; j < Ntracks_EB_list_Significance_Sig.size(); j++) {
            Ntracks_EB_list_Significance_Sig[j]->Fill(N_tracks_MTD_significance[j]);
            ch_iso_EB_list_Significance_Sig[j]->Fill(pT_sum_MTD_significance[j]);
            rel_ch_iso_EB_list_Significance_Sig[j]->Fill(rel_pT_sum_MTD_significance[j]);

            Ntracks_sim_EB_list_Significance_Sig[j]->Fill(N_tracks_sim_MTD_significance[j]);
            ch_iso_sim_EB_list_Significance_Sig[j]->Fill(pT_sum_sim_MTD_significance[j]);
            rel_ch_iso_sim_EB_list_Significance_Sig[j]->Fill(rel_pT_sum_sim_MTD_significance[j]);
          }

          if (rel_pT_sum_noMTD < rel_iso_cut_) {  // filling hists for iso efficiency calculations
            meEle_pt_noMTD_Sig_EB_->Fill(ele.pt());
            meEle_eta_noMTD_Sig_EB_->Fill(ele.eta());
            meEle_phi_noMTD_Sig_EB_->Fill(ele.phi());
          }

          for (long unsigned int k = 0; k < Ntracks_EB_list_Sig.size(); k++) {
            if (rel_pT_sum_MTD[k] < rel_iso_cut_) {
              Ele_pT_MTD_EB_list_Sig[k]->Fill(ele.pt());
              Ele_eta_MTD_EB_list_Sig[k]->Fill(ele.eta());
              Ele_phi_MTD_EB_list_Sig[k]->Fill(ele.phi());

              Ele_pT_sim_MTD_EB_list_Sig[k]->Fill(ele_sim_pt);
            }
          }

          for (long unsigned int k = 0; k < Ntracks_EB_list_Significance_Sig.size(); k++) {
            if (rel_pT_sum_MTD_significance[k] < rel_iso_cut_) {
              Ele_pT_MTD_EB_list_Significance_Sig[k]->Fill(ele.pt());
              Ele_eta_MTD_EB_list_Significance_Sig[k]->Fill(ele.eta());
              Ele_phi_MTD_EB_list_Significance_Sig[k]->Fill(ele.phi());

              Ele_pT_sim_MTD_EB_list_Significance_Sig[k]->Fill(ele_sim_pt);
            }
          }

        } else {  // for endcap

          meEleISO_Ntracks_Sig_EE_->Fill(N_tracks_noMTD);  // Filling hists for Ntraks and chIso sums for noMTD case //
          meEleISO_chIso_Sig_EE_->Fill(pT_sum_noMTD);
          meEleISO_rel_chIso_Sig_EE_->Fill(rel_pT_sum_noMTD);

          for (long unsigned int j = 0; j < Ntracks_EE_list_Sig.size(); j++) {
            Ntracks_EE_list_Sig[j]->Fill(N_tracks_MTD[j]);
            ch_iso_EE_list_Sig[j]->Fill(pT_sum_MTD[j]);
            rel_ch_iso_EE_list_Sig[j]->Fill(rel_pT_sum_MTD[j]);

            Ntracks_sim_EE_list_Sig[j]->Fill(N_tracks_sim_MTD[j]);
            ch_iso_sim_EE_list_Sig[j]->Fill(pT_sum_sim_MTD[j]);
            rel_ch_iso_sim_EE_list_Sig[j]->Fill(rel_pT_sum_sim_MTD[j]);
          }

          for (long unsigned int j = 0; j < Ntracks_EE_list_Significance_Sig.size(); j++) {
            Ntracks_EE_list_Significance_Sig[j]->Fill(N_tracks_MTD_significance[j]);
            ch_iso_EE_list_Significance_Sig[j]->Fill(pT_sum_MTD_significance[j]);
            rel_ch_iso_EE_list_Significance_Sig[j]->Fill(rel_pT_sum_MTD_significance[j]);

            Ntracks_sim_EE_list_Significance_Sig[j]->Fill(N_tracks_sim_MTD_significance[j]);
            ch_iso_sim_EE_list_Significance_Sig[j]->Fill(pT_sum_sim_MTD_significance[j]);
            rel_ch_iso_sim_EE_list_Significance_Sig[j]->Fill(rel_pT_sum_sim_MTD_significance[j]);
          }

          if (rel_pT_sum_noMTD < rel_iso_cut_) {  // filling hists for iso efficiency calculations
            meEle_pt_noMTD_Sig_EE_->Fill(ele.pt());
            meEle_eta_noMTD_Sig_EE_->Fill(ele.eta());
            meEle_phi_noMTD_Sig_EE_->Fill(ele.phi());
          }

          for (long unsigned int k = 0; k < Ntracks_EE_list_Sig.size(); k++) {
            if (rel_pT_sum_MTD[k] < rel_iso_cut_) {
              Ele_pT_MTD_EE_list_Sig[k]->Fill(ele.pt());
              Ele_eta_MTD_EE_list_Sig[k]->Fill(ele.eta());
              Ele_phi_MTD_EE_list_Sig[k]->Fill(ele.phi());

              Ele_pT_sim_MTD_EE_list_Sig[k]->Fill(ele_sim_pt);
            }
          }

          for (long unsigned int k = 0; k < Ntracks_EE_list_Significance_Sig.size(); k++) {
            if (rel_pT_sum_MTD_significance[k] < rel_iso_cut_) {
              Ele_pT_MTD_EE_list_Significance_Sig[k]->Fill(ele.pt());
              Ele_eta_MTD_EE_list_Significance_Sig[k]->Fill(ele.eta());
              Ele_phi_MTD_EE_list_Significance_Sig[k]->Fill(ele.phi());

              Ele_pT_sim_MTD_EE_list_Significance_Sig[k]->Fill(ele_sim_pt);
            }
          }
        }
      } else {  // non-promt part
        if (Barrel_ele) {
          meEleISO_Ntracks_Bkg_EB_->Fill(N_tracks_noMTD);  // Filling hists for Ntraks and chIso sums for noMTD case //
          meEleISO_chIso_Bkg_EB_->Fill(pT_sum_noMTD);
          meEleISO_rel_chIso_Bkg_EB_->Fill(rel_pT_sum_noMTD);

          for (long unsigned int j = 0; j < Ntracks_EB_list_Bkg.size(); j++) {
            Ntracks_EB_list_Bkg[j]->Fill(N_tracks_MTD[j]);
            ch_iso_EB_list_Bkg[j]->Fill(pT_sum_MTD[j]);
            rel_ch_iso_EB_list_Bkg[j]->Fill(rel_pT_sum_MTD[j]);

            Ntracks_sim_EB_list_Bkg[j]->Fill(N_tracks_sim_MTD[j]);
            ch_iso_sim_EB_list_Bkg[j]->Fill(pT_sum_sim_MTD[j]);
            rel_ch_iso_sim_EB_list_Bkg[j]->Fill(rel_pT_sum_sim_MTD[j]);
          }

          for (long unsigned int j = 0; j < Ntracks_EB_list_Significance_Bkg.size(); j++) {
            Ntracks_EB_list_Significance_Bkg[j]->Fill(N_tracks_MTD_significance[j]);
            ch_iso_EB_list_Significance_Bkg[j]->Fill(pT_sum_MTD_significance[j]);
            rel_ch_iso_EB_list_Significance_Bkg[j]->Fill(rel_pT_sum_MTD_significance[j]);

            Ntracks_sim_EB_list_Significance_Bkg[j]->Fill(N_tracks_sim_MTD_significance[j]);
            ch_iso_sim_EB_list_Significance_Bkg[j]->Fill(pT_sum_sim_MTD_significance[j]);
            rel_ch_iso_sim_EB_list_Significance_Bkg[j]->Fill(rel_pT_sum_sim_MTD_significance[j]);
          }

          if (rel_pT_sum_noMTD < rel_iso_cut_) {  // filling hists for iso efficiency calculations
            meEle_pt_noMTD_Bkg_EB_->Fill(ele.pt());
            meEle_eta_noMTD_Bkg_EB_->Fill(ele.eta());
            meEle_phi_noMTD_Bkg_EB_->Fill(ele.phi());
          }

          for (long unsigned int k = 0; k < Ntracks_EB_list_Bkg.size(); k++) {
            if (rel_pT_sum_MTD[k] < rel_iso_cut_) {
              Ele_pT_MTD_EB_list_Bkg[k]->Fill(ele.pt());
              Ele_eta_MTD_EB_list_Bkg[k]->Fill(ele.eta());
              Ele_phi_MTD_EB_list_Bkg[k]->Fill(ele.phi());

              Ele_pT_sim_MTD_EB_list_Bkg[k]->Fill(ele_sim_pt);
            }
          }

          for (long unsigned int k = 0; k < Ntracks_EB_list_Significance_Bkg.size(); k++) {
            if (rel_pT_sum_MTD_significance[k] < rel_iso_cut_) {
              Ele_pT_MTD_EB_list_Significance_Bkg[k]->Fill(ele.pt());
              Ele_eta_MTD_EB_list_Significance_Bkg[k]->Fill(ele.eta());
              Ele_phi_MTD_EB_list_Significance_Bkg[k]->Fill(ele.phi());

              Ele_pT_sim_MTD_EB_list_Significance_Bkg[k]->Fill(ele_sim_pt);
            }
          }

        } else {                                           // for endcap
          meEleISO_Ntracks_Bkg_EE_->Fill(N_tracks_noMTD);  // Filling hists for Ntraks and chIso sums for noMTD case //
          meEleISO_chIso_Bkg_EE_->Fill(pT_sum_noMTD);
          meEleISO_rel_chIso_Bkg_EE_->Fill(rel_pT_sum_noMTD);

          for (long unsigned int j = 0; j < Ntracks_EE_list_Bkg.size(); j++) {
            Ntracks_EE_list_Bkg[j]->Fill(N_tracks_MTD[j]);
            ch_iso_EE_list_Bkg[j]->Fill(pT_sum_MTD[j]);
            rel_ch_iso_EE_list_Bkg[j]->Fill(rel_pT_sum_MTD[j]);

            Ntracks_sim_EE_list_Bkg[j]->Fill(N_tracks_sim_MTD[j]);
            ch_iso_sim_EE_list_Bkg[j]->Fill(pT_sum_sim_MTD[j]);
            rel_ch_iso_sim_EE_list_Bkg[j]->Fill(rel_pT_sum_sim_MTD[j]);
          }

          for (long unsigned int j = 0; j < Ntracks_EE_list_Significance_Bkg.size(); j++) {
            Ntracks_EE_list_Significance_Bkg[j]->Fill(N_tracks_MTD_significance[j]);
            ch_iso_EE_list_Significance_Bkg[j]->Fill(pT_sum_MTD_significance[j]);
            rel_ch_iso_EE_list_Significance_Bkg[j]->Fill(rel_pT_sum_MTD_significance[j]);

            Ntracks_sim_EE_list_Significance_Bkg[j]->Fill(N_tracks_sim_MTD_significance[j]);
            ch_iso_sim_EE_list_Significance_Bkg[j]->Fill(pT_sum_sim_MTD_significance[j]);
            rel_ch_iso_sim_EE_list_Significance_Bkg[j]->Fill(rel_pT_sum_sim_MTD_significance[j]);
          }

          if (rel_pT_sum_noMTD < rel_iso_cut_) {  // filling hists for iso efficiency calculations
            meEle_pt_noMTD_Bkg_EE_->Fill(ele.pt());
            meEle_eta_noMTD_Bkg_EE_->Fill(ele.eta());
            meEle_phi_noMTD_Bkg_EE_->Fill(ele.phi());
          }

          for (long unsigned int k = 0; k < Ntracks_EE_list_Bkg.size(); k++) {
            if (rel_pT_sum_MTD[k] < rel_iso_cut_) {
              Ele_pT_MTD_EE_list_Bkg[k]->Fill(ele.pt());
              Ele_eta_MTD_EE_list_Bkg[k]->Fill(ele.eta());
              Ele_phi_MTD_EE_list_Bkg[k]->Fill(ele.phi());

              Ele_pT_sim_MTD_EE_list_Bkg[k]->Fill(ele_sim_pt);
            }
          }
          for (long unsigned int k = 0; k < Ntracks_EE_list_Significance_Bkg.size(); k++) {
            if (rel_pT_sum_MTD_significance[k] < rel_iso_cut_) {
              Ele_pT_MTD_EE_list_Significance_Bkg[k]->Fill(ele.pt());
              Ele_eta_MTD_EE_list_Significance_Bkg[k]->Fill(ele.eta());
              Ele_phi_MTD_EE_list_Significance_Bkg[k]->Fill(ele.phi());

              Ele_pT_sim_MTD_EE_list_Significance_Bkg[k]->Fill(ele_sim_pt);
            }
          }
        }
      }
    }  // electron matched to a track
  }    // electron collection inside single event
}

// ------------ method for histogram booking ------------
void MtdEleIsoValidation::bookHistograms(DQMStore::IBooker& ibook, edm::Run const& run, edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  // histogram booking

  meEle_avg_error_SigTrk_check_ =
      ibook.book1D("SigTrk_avg_timming_err", "Average signal electron track MTD timing uncertainty", 200, 0, 2);
  meEle_avg_error_PUTrk_check_ =
      ibook.book1D("PUTrk_avg_timming_err", "Average PU track MTD timing uncertainty", 200, 0, 2);
  meEle_avg_error_vtx_check_ = ibook.book1D("Vtx_avg_timming_err", "Average veretex timing uncertainty", 200, 0, 2);

  meEle_test_check_ = ibook.book1D("test_hist", "test_hist", 102, -2, 100);

  meEle_no_dt_check_ =
      ibook.book1D("Track_dt_info_check",
                   "Tracks dt check - ratio between tracks with (value 1) and without (value 0) timing info",
                   2,
                   0,
                   2);

  meTrk_genMatch_check_ =
      ibook.book1D("Track_genMatch_info_check", "Track MTD genMatch check - tracks for MTD timing checked", 2, 0, 2);

  // signal
  meEleISO_Ntracks_Sig_EB_ = ibook.book1D("Ele_Iso_Ntracks_Sig_EB",
                                          "Tracks in isolation cone around electron track after basic cuts",
                                          20,
                                          0,
                                          20);  // hists for electrons

  meEleISO_chIso_Sig_EB_ = ibook.book1D(
      "Ele_chIso_sum_Sig_EB", "Track pT sum in isolation cone around electron track after basic cuts", 2000, 0, 20);

  meEleISO_rel_chIso_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_1_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_1_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons

  meEleISO_chIso_MTD_1_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_1_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_1_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_1_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_2_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_2_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons

  meEleISO_chIso_MTD_2_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_2_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_2_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_2_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_3_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_3_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_3_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_3_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_3_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_3_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_4_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_4_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_4_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_4_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_4_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_4_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_5_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_5_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_5_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_5_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_5_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_5_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_6_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_6_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_6_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_6_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_6_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_6_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_7_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_7_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_7_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_7_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_7_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_7_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_4sigma_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_4sigma_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_4sigma_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_4sigma_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_4sigma_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_4sigma_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_3sigma_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_3sigma_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_3sigma_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_3sigma_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_3sigma_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_3sigma_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_2sigma_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_2sigma_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_2sigma_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_2sigma_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_2sigma_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_2sigma_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_1_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_1_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons

  meEleISO_chIso_MTD_sim_1_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_1_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_1_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_1_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_2_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_2_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons

  meEleISO_chIso_MTD_sim_2_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_2_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_2_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_2_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_3_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_3_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_3_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_3_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_3_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_3_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_4_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_4_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_4_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_4_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_4_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_4_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_5_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_5_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_5_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_5_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_5_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_5_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_6_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_6_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_6_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_6_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_6_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_6_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_7_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_7_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_7_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_7_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_7_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_7_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_4sigma_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_4sigma_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_sim_4sigma_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_4sigma_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_4sigma_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_4sigma_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_3sigma_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_3sigma_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_sim_3sigma_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_3sigma_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_3sigma_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_3sigma_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_2sigma_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_2sigma_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_sim_2sigma_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_2sigma_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_2sigma_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_2sigma_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEle_pt_tot_Sig_EB_ =
      ibook.book1D("Ele_pT_tot_Sig_EB", "Electron pT tot", 30, 10, 100);  // hists for ele isto stuff start
  meEle_pt_noMTD_Sig_EB_ = ibook.book1D("Ele_pT_noMTD_Sig_EB", "Electron pT noMTD", 30, 10, 100);

  meEle_pt_sim_tot_Sig_EB_ =
      ibook.book1D("Ele_pT_sim_tot_Sig_EB", "Electron pT tot", 30, 10, 100);  // hists for ele isto stuff start

  meEle_eta_tot_Sig_EB_ = ibook.book1D("Ele_eta_tot_Sig_EB", "Electron eta tot", 128, -3.2, 3.2);
  meEle_eta_noMTD_Sig_EB_ = ibook.book1D("Ele_eta_noMTD_Sig_EB", "Electron eta noMTD", 128, -3.2, 3.2);

  meEle_phi_tot_Sig_EB_ = ibook.book1D("Ele_phi_tot_Sig_EB", "Electron phi tot", 128, -3.2, 3.2);
  meEle_phi_noMTD_Sig_EB_ = ibook.book1D("Ele_phi_noMTD_Sig_EB", "Electron phi noMTD", 128, -3.2, 3.2);

  meEle_pt_MTD_1_Sig_EB_ = ibook.book1D("Ele_pT_MTD_1_Sig_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_1_Sig_EB_ = ibook.book1D("Ele_eta_MTD_1_Sig_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_1_Sig_EB_ = ibook.book1D("Ele_phi_MTD_1_Sig_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_2_Sig_EB_ = ibook.book1D("Ele_pT_MTD_2_Sig_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_2_Sig_EB_ = ibook.book1D("Ele_eta_MTD_2_Sig_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_2_Sig_EB_ = ibook.book1D("Ele_phi_MTD_2_Sig_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_3_Sig_EB_ = ibook.book1D("Ele_pT_MTD_3_Sig_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_3_Sig_EB_ = ibook.book1D("Ele_eta_MTD_3_Sig_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_3_Sig_EB_ = ibook.book1D("Ele_phi_MTD_3_Sig_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_4_Sig_EB_ = ibook.book1D("Ele_pT_MTD_4_Sig_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_4_Sig_EB_ = ibook.book1D("Ele_eta_MTD_4_Sig_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_4_Sig_EB_ = ibook.book1D("Ele_phi_MTD_4_Sig_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_5_Sig_EB_ = ibook.book1D("Ele_pT_MTD_5_Sig_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_5_Sig_EB_ = ibook.book1D("Ele_eta_MTD_5_Sig_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_5_Sig_EB_ = ibook.book1D("Ele_phi_MTD_5_Sig_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_6_Sig_EB_ = ibook.book1D("Ele_pT_MTD_6_Sig_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_6_Sig_EB_ = ibook.book1D("Ele_eta_MTD_6_Sig_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_6_Sig_EB_ = ibook.book1D("Ele_phi_MTD_6_Sig_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_7_Sig_EB_ = ibook.book1D("Ele_pT_MTD_7_Sig_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_7_Sig_EB_ = ibook.book1D("Ele_eta_MTD_7_Sig_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_7_Sig_EB_ = ibook.book1D("Ele_phi_MTD_7_Sig_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_4sigma_Sig_EB_ = ibook.book1D("Ele_pT_MTD_4sigma_Sig_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_4sigma_Sig_EB_ = ibook.book1D("Ele_eta_MTD_4sigma_Sig_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_4sigma_Sig_EB_ = ibook.book1D("Ele_phi_MTD_4sigma_Sig_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_3sigma_Sig_EB_ = ibook.book1D("Ele_pT_MTD_3sigma_Sig_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_3sigma_Sig_EB_ = ibook.book1D("Ele_eta_MTD_3sigma_Sig_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_3sigma_Sig_EB_ = ibook.book1D("Ele_phi_MTD_3sigma_Sig_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_2sigma_Sig_EB_ = ibook.book1D("Ele_pT_MTD_2sigma_Sig_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_2sigma_Sig_EB_ = ibook.book1D("Ele_eta_MTD_2sigma_Sig_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_2sigma_Sig_EB_ = ibook.book1D("Ele_phi_MTD_2sigma_Sig_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_sim_MTD_1_Sig_EB_ = ibook.book1D("Ele_pT_sim_MTD_1_Sig_EB", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_2_Sig_EB_ = ibook.book1D("Ele_pT_sim_MTD_2_Sig_EB", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_3_Sig_EB_ = ibook.book1D("Ele_pT_sim_MTD_3_Sig_EB", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_4_Sig_EB_ = ibook.book1D("Ele_pT_sim_MTD_4_Sig_EB", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_5_Sig_EB_ = ibook.book1D("Ele_pT_sim_MTD_5_Sig_EB", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_6_Sig_EB_ = ibook.book1D("Ele_pT_sim_MTD_6_Sig_EB", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_7_Sig_EB_ = ibook.book1D("Ele_pT_sim_MTD_7_Sig_EB", "Electron pT MTD", 30, 10, 100);

  meEle_pt_sim_MTD_4sigma_Sig_EB_ = ibook.book1D("Ele_pT_sim_MTD_4sigma_Sig_EB", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_3sigma_Sig_EB_ = ibook.book1D("Ele_pT_sim_MTD_3sigma_Sig_EB", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_2sigma_Sig_EB_ = ibook.book1D("Ele_pT_sim_MTD_2sigma_Sig_EB", "Electron pT MTD", 30, 10, 100);

  meEleISO_Ntracks_Sig_EE_ = ibook.book1D("Ele_Iso_Ntracks_Sig_EE",
                                          "Tracks in isolation cone around electron track after basic cuts",
                                          20,
                                          0,
                                          20);  // hists for electrons
  meEleISO_chIso_Sig_EE_ = ibook.book1D(
      "Ele_chIso_sum_Sig_EE", "Track pT sum in isolation cone around electron track after basic cuts", 2000, 0, 20);
  meEleISO_rel_chIso_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_1_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_1_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_1_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_1_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_1_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_1_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_2_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_2_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_2_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_2_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_2_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_2_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_3_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_3_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_3_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_3_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_3_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_3_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_4_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_4_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_4_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_4_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_4_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_4_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_5_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_5_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_5_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_5_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_5_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_5_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_6_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_6_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_6_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_6_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_6_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_6_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_7_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_7_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_7_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_7_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_7_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_7_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_4sigma_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_4sigma_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_4sigma_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_4sigma_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_4sigma_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_4sigma_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_3sigma_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_3sigma_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_3sigma_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_3sigma_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_3sigma_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_3sigma_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_2sigma_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_2sigma_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_2sigma_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_2sigma_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_2sigma_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_2sigma_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_1_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_1_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_1_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_1_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_1_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_1_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_2_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_2_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_2_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_2_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_2_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_2_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_3_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_3_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_3_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_3_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_3_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_3_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_4_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_4_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_4_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_4_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_4_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_4_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_5_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_5_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_5_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_5_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_5_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_5_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_6_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_6_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_6_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_6_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_6_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_6_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_7_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_7_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_7_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_7_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_7_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_7_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_4sigma_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_4sigma_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_sim_4sigma_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_4sigma_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_4sigma_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_4sigma_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_3sigma_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_3sigma_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_sim_3sigma_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_3sigma_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_3sigma_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_3sigma_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_2sigma_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_2sigma_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_sim_2sigma_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_2sigma_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_2sigma_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_2sigma_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEle_pt_tot_Sig_EE_ =
      ibook.book1D("Ele_pT_tot_Sig_EE", "Electron pT tot", 30, 10, 100);  // hists for ele isto stuff start
  meEle_pt_noMTD_Sig_EE_ = ibook.book1D("Ele_pT_noMTD_Sig_EE", "Electron pT noMTD", 30, 10, 100);

  meEle_pt_sim_tot_Sig_EE_ =
      ibook.book1D("Ele_pT_sim_tot_Sig_EE", "Electron pT tot", 30, 10, 100);  // hists for ele isto stuff start

  meEle_eta_tot_Sig_EE_ = ibook.book1D("Ele_eta_tot_Sig_EE", "Electron eta tot", 128, -3.2, 3.2);
  meEle_eta_noMTD_Sig_EE_ = ibook.book1D("Ele_eta_noMTD_Sig_EE", "Electron eta noMTD", 128, -3.2, 3.2);

  meEle_phi_tot_Sig_EE_ = ibook.book1D("Ele_phi_tot_Sig_EE", "Electron phi tot", 128, -3.2, 3.2);
  meEle_phi_noMTD_Sig_EE_ = ibook.book1D("Ele_phi_noMTD_Sig_EE", "Electron phi noMTD", 128, -3.2, 3.2);

  meEle_pt_MTD_1_Sig_EE_ = ibook.book1D("Ele_pT_MTD_1_Sig_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_1_Sig_EE_ = ibook.book1D("Ele_eta_MTD_1_Sig_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_1_Sig_EE_ = ibook.book1D("Ele_phi_MTD_1_Sig_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_2_Sig_EE_ = ibook.book1D("Ele_pT_MTD_2_Sig_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_2_Sig_EE_ = ibook.book1D("Ele_eta_MTD_2_Sig_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_2_Sig_EE_ = ibook.book1D("Ele_phi_MTD_2_Sig_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_3_Sig_EE_ = ibook.book1D("Ele_pT_MTD_3_Sig_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_3_Sig_EE_ = ibook.book1D("Ele_eta_MTD_3_Sig_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_3_Sig_EE_ = ibook.book1D("Ele_phi_MTD_3_Sig_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_4_Sig_EE_ = ibook.book1D("Ele_pT_MTD_4_Sig_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_4_Sig_EE_ = ibook.book1D("Ele_eta_MTD_4_Sig_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_4_Sig_EE_ = ibook.book1D("Ele_phi_MTD_4_Sig_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_5_Sig_EE_ = ibook.book1D("Ele_pT_MTD_5_Sig_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_5_Sig_EE_ = ibook.book1D("Ele_eta_MTD_5_Sig_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_5_Sig_EE_ = ibook.book1D("Ele_phi_MTD_5_Sig_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_6_Sig_EE_ = ibook.book1D("Ele_pT_MTD_6_Sig_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_6_Sig_EE_ = ibook.book1D("Ele_eta_MTD_6_Sig_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_6_Sig_EE_ = ibook.book1D("Ele_phi_MTD_6_Sig_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_7_Sig_EE_ = ibook.book1D("Ele_pT_MTD_7_Sig_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_7_Sig_EE_ = ibook.book1D("Ele_eta_MTD_7_Sig_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_7_Sig_EE_ = ibook.book1D("Ele_phi_MTD_7_Sig_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_4sigma_Sig_EE_ = ibook.book1D("Ele_pT_MTD_4sigma_Sig_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_4sigma_Sig_EE_ = ibook.book1D("Ele_eta_MTD_4sigma_Sig_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_4sigma_Sig_EE_ = ibook.book1D("Ele_phi_MTD_4sigma_Sig_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_3sigma_Sig_EE_ = ibook.book1D("Ele_pT_MTD_3sigma_Sig_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_3sigma_Sig_EE_ = ibook.book1D("Ele_eta_MTD_3sigma_Sig_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_3sigma_Sig_EE_ = ibook.book1D("Ele_phi_MTD_3sigma_Sig_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_2sigma_Sig_EE_ = ibook.book1D("Ele_pT_MTD_2sigma_Sig_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_2sigma_Sig_EE_ = ibook.book1D("Ele_eta_MTD_2sigma_Sig_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_2sigma_Sig_EE_ = ibook.book1D("Ele_phi_MTD_2sigma_Sig_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_sim_MTD_1_Sig_EE_ = ibook.book1D("Ele_pT_sim_MTD_1_Sig_EE", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_2_Sig_EE_ = ibook.book1D("Ele_pT_sim_MTD_2_Sig_EE", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_3_Sig_EE_ = ibook.book1D("Ele_pT_sim_MTD_3_Sig_EE", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_4_Sig_EE_ = ibook.book1D("Ele_pT_sim_MTD_4_Sig_EE", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_5_Sig_EE_ = ibook.book1D("Ele_pT_sim_MTD_5_Sig_EE", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_6_Sig_EE_ = ibook.book1D("Ele_pT_sim_MTD_6_Sig_EE", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_7_Sig_EE_ = ibook.book1D("Ele_pT_sim_MTD_7_Sig_EE", "Electron pT MTD", 30, 10, 100);

  meEle_pt_sim_MTD_4sigma_Sig_EE_ = ibook.book1D("Ele_pT_sim_MTD_4sigma_Sig_EE", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_3sigma_Sig_EE_ = ibook.book1D("Ele_pT_sim_MTD_3sigma_Sig_EE", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_2sigma_Sig_EE_ = ibook.book1D("Ele_pT_sim_MTD_2sigma_Sig_EE", "Electron pT MTD", 30, 10, 100);

  // background
  meEleISO_Ntracks_Bkg_EB_ = ibook.book1D("Ele_Iso_Ntracks_Bkg_EB",
                                          "Tracks in isolation cone around electron track after basic cuts",
                                          20,
                                          0,
                                          20);  // hists for electrons
  meEleISO_chIso_Bkg_EB_ = ibook.book1D(
      "Ele_chIso_sum_Bkg_EB", "Track pT sum in isolation cone around electron track after basic cuts", 2000, 0, 20);
  meEleISO_rel_chIso_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_1_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_1_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_1_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_1_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_1_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_1_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_2_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_2_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_2_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_2_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_2_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_2_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_3_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_3_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_3_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_3_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_3_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_3_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_4_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_4_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_4_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_4_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_4_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_4_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_5_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_5_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_5_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_5_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_5_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_5_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_6_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_6_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_6_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_6_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_6_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_6_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_7_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_7_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_7_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_7_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_7_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_7_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_4sigma_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_4sigma_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_4sigma_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_4sigma_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_4sigma_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_4sigma_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_3sigma_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_3sigma_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_3sigma_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_3sigma_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_3sigma_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_3sigma_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_2sigma_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_2sigma_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_2sigma_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_2sigma_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_2sigma_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_2sigma_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_1_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_1_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_1_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_1_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_1_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_1_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_2_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_2_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_2_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_2_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_2_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_2_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_3_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_3_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_3_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_3_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_3_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_3_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_4_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_4_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_4_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_4_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_4_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_4_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_5_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_5_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_5_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_5_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_5_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_5_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_6_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_6_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_6_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_6_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_6_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_6_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_7_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_7_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_7_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_7_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_7_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_7_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_4sigma_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_4sigma_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_sim_4sigma_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_4sigma_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_4sigma_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_4sigma_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_3sigma_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_3sigma_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_sim_3sigma_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_3sigma_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_3sigma_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_3sigma_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_2sigma_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_2sigma_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_sim_2sigma_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_2sigma_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_2sigma_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_2sigma_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEle_pt_tot_Bkg_EB_ =
      ibook.book1D("Ele_pT_tot_Bkg_EB", "Electron pT tot", 30, 10, 100);  // hists for ele isto stuff start
  meEle_pt_noMTD_Bkg_EB_ = ibook.book1D("Ele_pT_noMTD_Bkg_EB", "Electron pT noMTD", 30, 10, 100);

  meEle_pt_sim_tot_Bkg_EB_ =
      ibook.book1D("Ele_pT_sim_tot_Bkg_EB", "Electron pT tot", 30, 10, 100);  // hists for ele isto stuff start

  meEle_eta_tot_Bkg_EB_ = ibook.book1D("Ele_eta_tot_Bkg_EB", "Electron eta tot", 128, -3.2, 3.2);
  meEle_eta_noMTD_Bkg_EB_ = ibook.book1D("Ele_eta_noMTD_Bkg_EB", "Electron eta noMTD", 128, -3.2, 3.2);

  meEle_phi_tot_Bkg_EB_ = ibook.book1D("Ele_phi_tot_Bkg_EB", "Electron phi tot", 128, -3.2, 3.2);
  meEle_phi_noMTD_Bkg_EB_ = ibook.book1D("Ele_phi_noMTD_Bkg_EB", "Electron phi noMTD", 128, -3.2, 3.2);

  meEle_pt_MTD_1_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_1_Bkg_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_1_Bkg_EB_ = ibook.book1D("Ele_eta_MTD_1_Bkg_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_1_Bkg_EB_ = ibook.book1D("Ele_phi_MTD_1_Bkg_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_2_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_2_Bkg_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_2_Bkg_EB_ = ibook.book1D("Ele_eta_MTD_2_Bkg_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_2_Bkg_EB_ = ibook.book1D("Ele_phi_MTD_2_Bkg_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_3_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_3_Bkg_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_3_Bkg_EB_ = ibook.book1D("Ele_eta_MTD_3_Bkg_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_3_Bkg_EB_ = ibook.book1D("Ele_phi_MTD_3_Bkg_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_4_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_4_Bkg_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_4_Bkg_EB_ = ibook.book1D("Ele_eta_MTD_4_Bkg_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_4_Bkg_EB_ = ibook.book1D("Ele_phi_MTD_4_Bkg_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_5_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_5_Bkg_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_5_Bkg_EB_ = ibook.book1D("Ele_eta_MTD_5_Bkg_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_5_Bkg_EB_ = ibook.book1D("Ele_phi_MTD_5_Bkg_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_6_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_6_Bkg_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_6_Bkg_EB_ = ibook.book1D("Ele_eta_MTD_6_Bkg_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_6_Bkg_EB_ = ibook.book1D("Ele_phi_MTD_6_Bkg_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_7_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_7_Bkg_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_7_Bkg_EB_ = ibook.book1D("Ele_eta_MTD_7_Bkg_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_7_Bkg_EB_ = ibook.book1D("Ele_phi_MTD_7_Bkg_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_4sigma_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_4sigma_Bkg_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_4sigma_Bkg_EB_ = ibook.book1D("Ele_eta_MTD_4sigma_Bkg_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_4sigma_Bkg_EB_ = ibook.book1D("Ele_phi_MTD_4sigma_Bkg_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_3sigma_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_3sigma_Bkg_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_3sigma_Bkg_EB_ = ibook.book1D("Ele_eta_MTD_3sigma_Bkg_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_3sigma_Bkg_EB_ = ibook.book1D("Ele_phi_MTD_3sigma_Bkg_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_2sigma_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_2sigma_Bkg_EB", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_2sigma_Bkg_EB_ = ibook.book1D("Ele_eta_MTD_2sigma_Bkg_EB", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_2sigma_Bkg_EB_ = ibook.book1D("Ele_phi_MTD_2sigma_Bkg_EB", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_sim_MTD_1_Bkg_EB_ = ibook.book1D("Ele_pT_sim_MTD_1_Bkg_EB", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_2_Bkg_EB_ = ibook.book1D("Ele_pT_sim_MTD_2_Bkg_EB", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_3_Bkg_EB_ = ibook.book1D("Ele_pT_sim_MTD_3_Bkg_EB", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_4_Bkg_EB_ = ibook.book1D("Ele_pT_sim_MTD_4_Bkg_EB", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_5_Bkg_EB_ = ibook.book1D("Ele_pT_sim_MTD_5_Bkg_EB", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_6_Bkg_EB_ = ibook.book1D("Ele_pT_sim_MTD_6_Bkg_EB", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_7_Bkg_EB_ = ibook.book1D("Ele_pT_sim_MTD_7_Bkg_EB", "Electron pT MTD", 30, 10, 100);

  meEle_pt_sim_MTD_4sigma_Bkg_EB_ = ibook.book1D("Ele_pT_sim_MTD_4sigma_Bkg_EB", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_3sigma_Bkg_EB_ = ibook.book1D("Ele_pT_sim_MTD_3sigma_Bkg_EB", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_2sigma_Bkg_EB_ = ibook.book1D("Ele_pT_sim_MTD_2sigma_Bkg_EB", "Electron pT MTD", 30, 10, 100);

  meEleISO_Ntracks_Bkg_EE_ = ibook.book1D("Ele_Iso_Ntracks_Bkg_EE",
                                          "Tracks in isolation cone around electron track after basic cuts",
                                          20,
                                          0,
                                          20);  // hists for electrons
  meEleISO_chIso_Bkg_EE_ = ibook.book1D(
      "Ele_chIso_sum_Bkg_EE", "Track pT sum in isolation cone around electron track after basic cuts", 2000, 0, 20);
  meEleISO_rel_chIso_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_1_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_1_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_1_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_1_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_1_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_1_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_2_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_2_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_2_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_2_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_2_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_2_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_3_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_3_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_3_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_3_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_3_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_3_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_4_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_4_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_4_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_4_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_4_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_4_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_5_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_5_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_5_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_5_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_5_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_5_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_6_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_6_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_6_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_6_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_6_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_6_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_7_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_7_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_7_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_7_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_7_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_7_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_4sigma_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_4sigma_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_4sigma_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_4sigma_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_4sigma_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_4sigma_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_3sigma_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_3sigma_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_3sigma_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_3sigma_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_3sigma_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_3sigma_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_2sigma_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_2sigma_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_2sigma_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_2sigma_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_2sigma_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_2sigma_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_1_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_1_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_1_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_1_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_1_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_1_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_2_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_2_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_2_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_2_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_2_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_2_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_3_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_3_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_3_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_3_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_3_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_3_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_4_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_4_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_4_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_4_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_4_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_4_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_5_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_5_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_5_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_5_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_5_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_5_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_6_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_6_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_6_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_6_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_6_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_6_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_7_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_7_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);  // hists for electrons
  meEleISO_chIso_MTD_sim_7_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_7_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_7_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_7_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_4sigma_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_4sigma_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_sim_4sigma_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_4sigma_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_4sigma_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_4sigma_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_3sigma_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_3sigma_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_sim_3sigma_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_3sigma_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_3sigma_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_3sigma_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEleISO_Ntracks_MTD_sim_2sigma_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_sim_2sigma_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_sim_2sigma_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_sim_2sigma_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic cuts with MTD",
                   2000,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_sim_2sigma_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_sim_2sigma_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD",
                   1000,
                   0,
                   4);

  meEle_pt_tot_Bkg_EE_ =
      ibook.book1D("Ele_pT_tot_Bkg_EE", "Electron pT tot", 30, 10, 100);  // hists for ele isto stuff start
  meEle_pt_noMTD_Bkg_EE_ = ibook.book1D("Ele_pT_noMTD_Bkg_EE", "Electron pT noMTD", 30, 10, 100);

  meEle_pt_sim_tot_Bkg_EE_ =
      ibook.book1D("Ele_pT_sim_tot_Bkg_EE", "Electron pT tot", 30, 10, 100);  // hists for ele isto stuff start

  meEle_eta_tot_Bkg_EE_ = ibook.book1D("Ele_eta_tot_Bkg_EE", "Electron eta tot", 128, -3.2, 3.2);
  meEle_eta_noMTD_Bkg_EE_ = ibook.book1D("Ele_eta_noMTD_Bkg_EE", "Electron eta noMTD", 128, -3.2, 3.2);

  meEle_phi_tot_Bkg_EE_ = ibook.book1D("Ele_phi_tot_Bkg_EE", "Electron phi tot", 128, -3.2, 3.2);
  meEle_phi_noMTD_Bkg_EE_ = ibook.book1D("Ele_phi_noMTD_Bkg_EE", "Electron phi noMTD", 128, -3.2, 3.2);

  meEle_pt_MTD_1_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_1_Bkg_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_1_Bkg_EE_ = ibook.book1D("Ele_eta_MTD_1_Bkg_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_1_Bkg_EE_ = ibook.book1D("Ele_phi_MTD_1_Bkg_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_2_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_2_Bkg_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_2_Bkg_EE_ = ibook.book1D("Ele_eta_MTD_2_Bkg_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_2_Bkg_EE_ = ibook.book1D("Ele_phi_MTD_2_Bkg_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_3_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_3_Bkg_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_3_Bkg_EE_ = ibook.book1D("Ele_eta_MTD_3_Bkg_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_3_Bkg_EE_ = ibook.book1D("Ele_phi_MTD_3_Bkg_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_4_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_4_Bkg_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_4_Bkg_EE_ = ibook.book1D("Ele_eta_MTD_4_Bkg_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_4_Bkg_EE_ = ibook.book1D("Ele_phi_MTD_4_Bkg_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_5_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_5_Bkg_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_5_Bkg_EE_ = ibook.book1D("Ele_eta_MTD_5_Bkg_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_5_Bkg_EE_ = ibook.book1D("Ele_phi_MTD_5_Bkg_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_6_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_6_Bkg_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_6_Bkg_EE_ = ibook.book1D("Ele_eta_MTD_6_Bkg_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_6_Bkg_EE_ = ibook.book1D("Ele_phi_MTD_6_Bkg_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_7_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_7_Bkg_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_7_Bkg_EE_ = ibook.book1D("Ele_eta_MTD_7_Bkg_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_7_Bkg_EE_ = ibook.book1D("Ele_phi_MTD_7_Bkg_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_4sigma_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_4sigma_Bkg_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_4sigma_Bkg_EE_ = ibook.book1D("Ele_eta_MTD_4sigma_Bkg_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_4sigma_Bkg_EE_ = ibook.book1D("Ele_phi_MTD_4sigma_Bkg_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_3sigma_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_3sigma_Bkg_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_3sigma_Bkg_EE_ = ibook.book1D("Ele_eta_MTD_3sigma_Bkg_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_3sigma_Bkg_EE_ = ibook.book1D("Ele_phi_MTD_3sigma_Bkg_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_MTD_2sigma_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_2sigma_Bkg_EE", "Electron pT MTD", 30, 10, 100);
  meEle_eta_MTD_2sigma_Bkg_EE_ = ibook.book1D("Ele_eta_MTD_2sigma_Bkg_EE", "Electron eta MTD", 128, -3.2, 3.2);
  meEle_phi_MTD_2sigma_Bkg_EE_ = ibook.book1D("Ele_phi_MTD_2sigma_Bkg_EE", "Electron phi MTD", 128, -3.2, 3.2);

  meEle_pt_sim_MTD_1_Bkg_EE_ = ibook.book1D("Ele_pT_sim_MTD_1_Bkg_EE", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_2_Bkg_EE_ = ibook.book1D("Ele_pT_sim_MTD_2_Bkg_EE", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_3_Bkg_EE_ = ibook.book1D("Ele_pT_sim_MTD_3_Bkg_EE", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_4_Bkg_EE_ = ibook.book1D("Ele_pT_sim_MTD_4_Bkg_EE", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_5_Bkg_EE_ = ibook.book1D("Ele_pT_sim_MTD_5_Bkg_EE", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_6_Bkg_EE_ = ibook.book1D("Ele_pT_sim_MTD_6_Bkg_EE", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_7_Bkg_EE_ = ibook.book1D("Ele_pT_sim_MTD_7_Bkg_EE", "Electron pT MTD", 30, 10, 100);

  meEle_pt_sim_MTD_4sigma_Bkg_EE_ = ibook.book1D("Ele_pT_sim_MTD_4sigma_Bkg_EE", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_3sigma_Bkg_EE_ = ibook.book1D("Ele_pT_sim_MTD_3sigma_Bkg_EE", "Electron pT MTD", 30, 10, 100);
  meEle_pt_sim_MTD_2sigma_Bkg_EE_ = ibook.book1D("Ele_pT_sim_MTD_2sigma_Bkg_EE", "Electron pT MTD", 30, 10, 100);

  meEle_dt_general_pT_1 =
      ibook.book1D("Iso_track_dt_general_pT_10_20", "Iso cone track dt distribution in pT bin 10-20 GeV ", 100, 0, 1);
  meEle_dt_general_pT_2 =
      ibook.book1D("Iso_track_dt_general_pT_20_30", "Iso cone track dt distribution in pT bin 20-30 GeV ", 100, 0, 1);
  meEle_dt_general_pT_3 =
      ibook.book1D("Iso_track_dt_general_pT_30_40", "Iso cone track dt distribution in pT bin 30-40 GeV ", 100, 0, 1);
  meEle_dt_general_pT_4 =
      ibook.book1D("Iso_track_dt_general_pT_40_50", "Iso cone track dt distribution in pT bin 40-50 GeV ", 100, 0, 1);
  meEle_dt_general_pT_5 =
      ibook.book1D("Iso_track_dt_general_pT_50_60", "Iso cone track dt distribution in pT bin 50-60 GeV ", 100, 0, 1);
  meEle_dt_general_pT_6 =
      ibook.book1D("Iso_track_dt_general_pT_60_70", "Iso cone track dt distribution in pT bin 60-70 GeV ", 100, 0, 1);
  meEle_dt_general_pT_7 =
      ibook.book1D("Iso_track_dt_general_pT_70_80", "Iso cone track dt distribution in pT bin 70-80 GeV ", 100, 0, 1);
  meEle_dt_general_pT_8 =
      ibook.book1D("Iso_track_dt_general_pT_80_90", "Iso cone track dt distribution in pT bin 80-90 GeV ", 100, 0, 1);
  meEle_dt_general_pT_9 =
      ibook.book1D("Iso_track_dt_general_pT_90_100", "Iso cone track dt distribution in pT bin 90-100 GeV ", 100, 0, 1);

  meEle_dtSignif_general_pT_1 = ibook.book1D(
      "Iso_track_dtSignif_general_pT_10_20", "Iso cone track dt distribution in pT bin 10-20 GeV ", 1000, 0, 10);
  meEle_dtSignif_general_pT_2 = ibook.book1D(
      "Iso_track_dtSignif_general_pT_20_30", "Iso cone track dt distribution in pT bin 20-30 GeV ", 1000, 0, 10);
  meEle_dtSignif_general_pT_3 = ibook.book1D(
      "Iso_track_dtSignif_general_pT_30_40", "Iso cone track dt distribution in pT bin 30-40 GeV ", 1000, 0, 10);
  meEle_dtSignif_general_pT_4 = ibook.book1D(
      "Iso_track_dtSignif_general_pT_40_50", "Iso cone track dt distribution in pT bin 40-50 GeV ", 1000, 0, 10);
  meEle_dtSignif_general_pT_5 = ibook.book1D(
      "Iso_track_dtSignif_general_pT_50_60", "Iso cone track dt distribution in pT bin 50-60 GeV ", 1000, 0, 10);
  meEle_dtSignif_general_pT_6 = ibook.book1D(
      "Iso_track_dtSignif_general_pT_60_70", "Iso cone track dt distribution in pT bin 60-70 GeV ", 1000, 0, 10);
  meEle_dtSignif_general_pT_7 = ibook.book1D(
      "Iso_track_dtSignif_general_pT_70_80", "Iso cone track dt distribution in pT bin 70-80 GeV ", 1000, 0, 10);
  meEle_dtSignif_general_pT_8 = ibook.book1D(
      "Iso_track_dtSignif_general_pT_80_90", "Iso cone track dt distribution in pT bin 80-90 GeV ", 1000, 0, 10);
  meEle_dtSignif_general_pT_9 = ibook.book1D(
      "Iso_track_dtSignif_general_pT_90_100", "Iso cone track dt distribution in pT bin 90-100 GeV ", 1000, 0, 10);

  meEle_dt_general_eta_1 =
      ibook.book1D("Iso_track_dt_general_eta_0_05", "Iso cone track dt distribution in eta bin 0.0-0.5 ", 100, 0, 1);
  meEle_dt_general_eta_2 =
      ibook.book1D("Iso_track_dt_general_eta_05_10", "Iso cone track dt distribution in eta bin 0.5-1.0 ", 100, 0, 1);
  meEle_dt_general_eta_3 =
      ibook.book1D("Iso_track_dt_general_eta_10_15", "Iso cone track dt distribution in eta bin 1.0-1.5 ", 100, 0, 1);
  meEle_dt_general_eta_4 =
      ibook.book1D("Iso_track_dt_general_eta_15_20", "Iso cone track dt distribution in eta bin 1.5-2.0 ", 100, 0, 1);
  meEle_dt_general_eta_5 =
      ibook.book1D("Iso_track_dt_general_eta_20_24", "Iso cone track dt distribution in eta bin 2.0-2.4 ", 100, 0, 1);

  meEle_dtSignif_general_eta_1 = ibook.book1D(
      "Iso_track_dtSignif_general_eta_0_05", "Iso cone track dt distribution in eta bin 0.0-0.5 ", 1000, 0, 10);
  meEle_dtSignif_general_eta_2 = ibook.book1D(
      "Iso_track_dtSignif_general_eta_05_10", "Iso cone track dt distribution in eta bin 0.5-1.0 ", 1000, 0, 10);
  meEle_dtSignif_general_eta_3 = ibook.book1D(
      "Iso_track_dtSignif_general_eta_10_15", "Iso cone track dt distribution in eta bin 1.0-1.5 ", 1000, 0, 10);
  meEle_dtSignif_general_eta_4 = ibook.book1D(
      "Iso_track_dtSignif_general_eta_15_20", "Iso cone track dt distribution in eta bin 1.5-2.0 ", 1000, 0, 10);
  meEle_dtSignif_general_eta_5 = ibook.book1D(
      "Iso_track_dtSignif_general_eta_20_24", "Iso cone track dt distribution in eta bin 2.0-2.4 ", 1000, 0, 10);

  // defining vectors for more efficient hist filling
  // Promt part
  Ntracks_EB_list_Sig = {meEleISO_Ntracks_MTD_1_Sig_EB_,
                         meEleISO_Ntracks_MTD_2_Sig_EB_,
                         meEleISO_Ntracks_MTD_3_Sig_EB_,
                         meEleISO_Ntracks_MTD_4_Sig_EB_,
                         meEleISO_Ntracks_MTD_5_Sig_EB_,
                         meEleISO_Ntracks_MTD_6_Sig_EB_,
                         meEleISO_Ntracks_MTD_7_Sig_EB_};
  ch_iso_EB_list_Sig = {meEleISO_chIso_MTD_1_Sig_EB_,
                        meEleISO_chIso_MTD_2_Sig_EB_,
                        meEleISO_chIso_MTD_3_Sig_EB_,
                        meEleISO_chIso_MTD_4_Sig_EB_,
                        meEleISO_chIso_MTD_5_Sig_EB_,
                        meEleISO_chIso_MTD_6_Sig_EB_,
                        meEleISO_chIso_MTD_7_Sig_EB_};
  rel_ch_iso_EB_list_Sig = {meEleISO_rel_chIso_MTD_1_Sig_EB_,
                            meEleISO_rel_chIso_MTD_2_Sig_EB_,
                            meEleISO_rel_chIso_MTD_3_Sig_EB_,
                            meEleISO_rel_chIso_MTD_4_Sig_EB_,
                            meEleISO_rel_chIso_MTD_5_Sig_EB_,
                            meEleISO_rel_chIso_MTD_6_Sig_EB_,
                            meEleISO_rel_chIso_MTD_7_Sig_EB_};

  Ntracks_EB_list_Significance_Sig = {
      meEleISO_Ntracks_MTD_4sigma_Sig_EB_, meEleISO_Ntracks_MTD_3sigma_Sig_EB_, meEleISO_Ntracks_MTD_2sigma_Sig_EB_};
  ch_iso_EB_list_Significance_Sig = {
      meEleISO_chIso_MTD_4sigma_Sig_EB_, meEleISO_chIso_MTD_3sigma_Sig_EB_, meEleISO_chIso_MTD_2sigma_Sig_EB_};
  rel_ch_iso_EB_list_Significance_Sig = {meEleISO_rel_chIso_MTD_4sigma_Sig_EB_,
                                         meEleISO_rel_chIso_MTD_3sigma_Sig_EB_,
                                         meEleISO_rel_chIso_MTD_2sigma_Sig_EB_};

  Ntracks_EE_list_Sig = {meEleISO_Ntracks_MTD_1_Sig_EE_,
                         meEleISO_Ntracks_MTD_2_Sig_EE_,
                         meEleISO_Ntracks_MTD_3_Sig_EE_,
                         meEleISO_Ntracks_MTD_4_Sig_EE_,
                         meEleISO_Ntracks_MTD_5_Sig_EE_,
                         meEleISO_Ntracks_MTD_6_Sig_EE_,
                         meEleISO_Ntracks_MTD_7_Sig_EE_};
  ch_iso_EE_list_Sig = {meEleISO_chIso_MTD_1_Sig_EE_,
                        meEleISO_chIso_MTD_2_Sig_EE_,
                        meEleISO_chIso_MTD_3_Sig_EE_,
                        meEleISO_chIso_MTD_4_Sig_EE_,
                        meEleISO_chIso_MTD_5_Sig_EE_,
                        meEleISO_chIso_MTD_6_Sig_EE_,
                        meEleISO_chIso_MTD_7_Sig_EE_};
  rel_ch_iso_EE_list_Sig = {meEleISO_rel_chIso_MTD_1_Sig_EE_,
                            meEleISO_rel_chIso_MTD_2_Sig_EE_,
                            meEleISO_rel_chIso_MTD_3_Sig_EE_,
                            meEleISO_rel_chIso_MTD_4_Sig_EE_,
                            meEleISO_rel_chIso_MTD_5_Sig_EE_,
                            meEleISO_rel_chIso_MTD_6_Sig_EE_,
                            meEleISO_rel_chIso_MTD_7_Sig_EE_};

  Ntracks_EE_list_Significance_Sig = {
      meEleISO_Ntracks_MTD_4sigma_Sig_EE_, meEleISO_Ntracks_MTD_3sigma_Sig_EE_, meEleISO_Ntracks_MTD_2sigma_Sig_EE_};
  ch_iso_EE_list_Significance_Sig = {
      meEleISO_chIso_MTD_4sigma_Sig_EE_, meEleISO_chIso_MTD_3sigma_Sig_EE_, meEleISO_chIso_MTD_2sigma_Sig_EE_};
  rel_ch_iso_EE_list_Significance_Sig = {meEleISO_rel_chIso_MTD_4sigma_Sig_EE_,
                                         meEleISO_rel_chIso_MTD_3sigma_Sig_EE_,
                                         meEleISO_rel_chIso_MTD_2sigma_Sig_EE_};

  Ele_pT_MTD_EB_list_Sig = {meEle_pt_MTD_1_Sig_EB_,
                            meEle_pt_MTD_2_Sig_EB_,
                            meEle_pt_MTD_3_Sig_EB_,
                            meEle_pt_MTD_4_Sig_EB_,
                            meEle_pt_MTD_5_Sig_EB_,
                            meEle_pt_MTD_6_Sig_EB_,
                            meEle_pt_MTD_7_Sig_EB_};
  Ele_eta_MTD_EB_list_Sig = {meEle_eta_MTD_1_Sig_EB_,
                             meEle_eta_MTD_2_Sig_EB_,
                             meEle_eta_MTD_3_Sig_EB_,
                             meEle_eta_MTD_4_Sig_EB_,
                             meEle_eta_MTD_5_Sig_EB_,
                             meEle_eta_MTD_6_Sig_EB_,
                             meEle_eta_MTD_7_Sig_EB_};
  Ele_phi_MTD_EB_list_Sig = {meEle_phi_MTD_1_Sig_EB_,
                             meEle_phi_MTD_2_Sig_EB_,
                             meEle_phi_MTD_3_Sig_EB_,
                             meEle_phi_MTD_4_Sig_EB_,
                             meEle_phi_MTD_5_Sig_EB_,
                             meEle_phi_MTD_6_Sig_EB_,
                             meEle_phi_MTD_7_Sig_EB_};

  Ele_pT_MTD_EB_list_Significance_Sig = {
      meEle_pt_MTD_4sigma_Sig_EB_, meEle_pt_MTD_3sigma_Sig_EB_, meEle_pt_MTD_2sigma_Sig_EB_};
  Ele_eta_MTD_EB_list_Significance_Sig = {
      meEle_eta_MTD_4sigma_Sig_EB_, meEle_eta_MTD_3sigma_Sig_EB_, meEle_eta_MTD_2sigma_Sig_EB_};
  Ele_phi_MTD_EB_list_Significance_Sig = {
      meEle_phi_MTD_4sigma_Sig_EB_, meEle_phi_MTD_3sigma_Sig_EB_, meEle_phi_MTD_2sigma_Sig_EB_};

  Ele_pT_MTD_EE_list_Sig = {meEle_pt_MTD_1_Sig_EE_,
                            meEle_pt_MTD_2_Sig_EE_,
                            meEle_pt_MTD_3_Sig_EE_,
                            meEle_pt_MTD_4_Sig_EE_,
                            meEle_pt_MTD_5_Sig_EE_,
                            meEle_pt_MTD_6_Sig_EE_,
                            meEle_pt_MTD_7_Sig_EE_};
  Ele_eta_MTD_EE_list_Sig = {meEle_eta_MTD_1_Sig_EE_,
                             meEle_eta_MTD_2_Sig_EE_,
                             meEle_eta_MTD_3_Sig_EE_,
                             meEle_eta_MTD_4_Sig_EE_,
                             meEle_eta_MTD_5_Sig_EE_,
                             meEle_eta_MTD_6_Sig_EE_,
                             meEle_eta_MTD_7_Sig_EE_};
  Ele_phi_MTD_EE_list_Sig = {meEle_phi_MTD_1_Sig_EE_,
                             meEle_phi_MTD_2_Sig_EE_,
                             meEle_phi_MTD_3_Sig_EE_,
                             meEle_phi_MTD_4_Sig_EE_,
                             meEle_phi_MTD_5_Sig_EE_,
                             meEle_phi_MTD_6_Sig_EE_,
                             meEle_phi_MTD_7_Sig_EE_};

  Ele_pT_MTD_EE_list_Significance_Sig = {
      meEle_pt_MTD_4sigma_Sig_EE_, meEle_pt_MTD_3sigma_Sig_EE_, meEle_pt_MTD_2sigma_Sig_EE_};
  Ele_eta_MTD_EE_list_Significance_Sig = {
      meEle_eta_MTD_4sigma_Sig_EE_, meEle_eta_MTD_3sigma_Sig_EE_, meEle_eta_MTD_2sigma_Sig_EE_};
  Ele_phi_MTD_EE_list_Significance_Sig = {
      meEle_phi_MTD_4sigma_Sig_EE_, meEle_phi_MTD_3sigma_Sig_EE_, meEle_phi_MTD_2sigma_Sig_EE_};

  // For SIM CASE

  Ntracks_sim_EB_list_Sig = {meEleISO_Ntracks_MTD_sim_1_Sig_EB_,
                             meEleISO_Ntracks_MTD_sim_2_Sig_EB_,
                             meEleISO_Ntracks_MTD_sim_3_Sig_EB_,
                             meEleISO_Ntracks_MTD_sim_4_Sig_EB_,
                             meEleISO_Ntracks_MTD_sim_5_Sig_EB_,
                             meEleISO_Ntracks_MTD_sim_6_Sig_EB_,
                             meEleISO_Ntracks_MTD_sim_7_Sig_EB_};
  ch_iso_sim_EB_list_Sig = {meEleISO_chIso_MTD_sim_1_Sig_EB_,
                            meEleISO_chIso_MTD_sim_2_Sig_EB_,
                            meEleISO_chIso_MTD_sim_3_Sig_EB_,
                            meEleISO_chIso_MTD_sim_4_Sig_EB_,
                            meEleISO_chIso_MTD_sim_5_Sig_EB_,
                            meEleISO_chIso_MTD_sim_6_Sig_EB_,
                            meEleISO_chIso_MTD_sim_7_Sig_EB_};
  rel_ch_iso_sim_EB_list_Sig = {meEleISO_rel_chIso_MTD_sim_1_Sig_EB_,
                                meEleISO_rel_chIso_MTD_sim_2_Sig_EB_,
                                meEleISO_rel_chIso_MTD_sim_3_Sig_EB_,
                                meEleISO_rel_chIso_MTD_sim_4_Sig_EB_,
                                meEleISO_rel_chIso_MTD_sim_5_Sig_EB_,
                                meEleISO_rel_chIso_MTD_sim_6_Sig_EB_,
                                meEleISO_rel_chIso_MTD_sim_7_Sig_EB_};

  Ntracks_sim_EB_list_Significance_Sig = {meEleISO_Ntracks_MTD_sim_4sigma_Sig_EB_,
                                          meEleISO_Ntracks_MTD_sim_3sigma_Sig_EB_,
                                          meEleISO_Ntracks_MTD_sim_2sigma_Sig_EB_};
  ch_iso_sim_EB_list_Significance_Sig = {meEleISO_chIso_MTD_sim_4sigma_Sig_EB_,
                                         meEleISO_chIso_MTD_sim_3sigma_Sig_EB_,
                                         meEleISO_chIso_MTD_sim_2sigma_Sig_EB_};
  rel_ch_iso_sim_EB_list_Significance_Sig = {meEleISO_rel_chIso_MTD_sim_4sigma_Sig_EB_,
                                             meEleISO_rel_chIso_MTD_sim_3sigma_Sig_EB_,
                                             meEleISO_rel_chIso_MTD_sim_2sigma_Sig_EB_};

  Ntracks_sim_EE_list_Sig = {meEleISO_Ntracks_MTD_sim_1_Sig_EE_,
                             meEleISO_Ntracks_MTD_sim_2_Sig_EE_,
                             meEleISO_Ntracks_MTD_sim_3_Sig_EE_,
                             meEleISO_Ntracks_MTD_sim_4_Sig_EE_,
                             meEleISO_Ntracks_MTD_sim_5_Sig_EE_,
                             meEleISO_Ntracks_MTD_sim_6_Sig_EE_,
                             meEleISO_Ntracks_MTD_sim_7_Sig_EE_};
  ch_iso_sim_EE_list_Sig = {meEleISO_chIso_MTD_sim_1_Sig_EE_,
                            meEleISO_chIso_MTD_sim_2_Sig_EE_,
                            meEleISO_chIso_MTD_sim_3_Sig_EE_,
                            meEleISO_chIso_MTD_sim_4_Sig_EE_,
                            meEleISO_chIso_MTD_sim_5_Sig_EE_,
                            meEleISO_chIso_MTD_sim_6_Sig_EE_,
                            meEleISO_chIso_MTD_sim_7_Sig_EE_};
  rel_ch_iso_sim_EE_list_Sig = {meEleISO_rel_chIso_MTD_sim_1_Sig_EE_,
                                meEleISO_rel_chIso_MTD_sim_2_Sig_EE_,
                                meEleISO_rel_chIso_MTD_sim_3_Sig_EE_,
                                meEleISO_rel_chIso_MTD_sim_4_Sig_EE_,
                                meEleISO_rel_chIso_MTD_sim_5_Sig_EE_,
                                meEleISO_rel_chIso_MTD_sim_6_Sig_EE_,
                                meEleISO_rel_chIso_MTD_sim_7_Sig_EE_};

  Ntracks_sim_EE_list_Significance_Sig = {meEleISO_Ntracks_MTD_sim_4sigma_Sig_EE_,
                                          meEleISO_Ntracks_MTD_sim_3sigma_Sig_EE_,
                                          meEleISO_Ntracks_MTD_sim_2sigma_Sig_EE_};
  ch_iso_sim_EE_list_Significance_Sig = {meEleISO_chIso_MTD_sim_4sigma_Sig_EE_,
                                         meEleISO_chIso_MTD_sim_3sigma_Sig_EE_,
                                         meEleISO_chIso_MTD_sim_2sigma_Sig_EE_};
  rel_ch_iso_sim_EE_list_Significance_Sig = {meEleISO_rel_chIso_MTD_sim_4sigma_Sig_EE_,
                                             meEleISO_rel_chIso_MTD_sim_3sigma_Sig_EE_,
                                             meEleISO_rel_chIso_MTD_sim_2sigma_Sig_EE_};

  Ele_pT_sim_MTD_EB_list_Sig = {meEle_pt_sim_MTD_1_Sig_EB_,
                                meEle_pt_sim_MTD_2_Sig_EB_,
                                meEle_pt_sim_MTD_3_Sig_EB_,
                                meEle_pt_sim_MTD_4_Sig_EB_,
                                meEle_pt_sim_MTD_5_Sig_EB_,
                                meEle_pt_sim_MTD_6_Sig_EB_,
                                meEle_pt_sim_MTD_7_Sig_EB_};

  Ele_pT_sim_MTD_EB_list_Significance_Sig = {
      meEle_pt_sim_MTD_4sigma_Sig_EB_, meEle_pt_sim_MTD_3sigma_Sig_EB_, meEle_pt_sim_MTD_2sigma_Sig_EB_};

  Ele_pT_sim_MTD_EE_list_Sig = {meEle_pt_sim_MTD_1_Sig_EE_,
                                meEle_pt_sim_MTD_2_Sig_EE_,
                                meEle_pt_sim_MTD_3_Sig_EE_,
                                meEle_pt_sim_MTD_4_Sig_EE_,
                                meEle_pt_sim_MTD_5_Sig_EE_,
                                meEle_pt_sim_MTD_6_Sig_EE_,
                                meEle_pt_sim_MTD_7_Sig_EE_};

  Ele_pT_sim_MTD_EE_list_Significance_Sig = {
      meEle_pt_sim_MTD_4sigma_Sig_EE_, meEle_pt_sim_MTD_3sigma_Sig_EE_, meEle_pt_sim_MTD_2sigma_Sig_EE_};

  // Non-promt part
  Ntracks_EB_list_Bkg = {meEleISO_Ntracks_MTD_1_Bkg_EB_,
                         meEleISO_Ntracks_MTD_2_Bkg_EB_,
                         meEleISO_Ntracks_MTD_3_Bkg_EB_,
                         meEleISO_Ntracks_MTD_4_Bkg_EB_,
                         meEleISO_Ntracks_MTD_5_Bkg_EB_,
                         meEleISO_Ntracks_MTD_6_Bkg_EB_,
                         meEleISO_Ntracks_MTD_7_Bkg_EB_};
  ch_iso_EB_list_Bkg = {meEleISO_chIso_MTD_1_Bkg_EB_,
                        meEleISO_chIso_MTD_2_Bkg_EB_,
                        meEleISO_chIso_MTD_3_Bkg_EB_,
                        meEleISO_chIso_MTD_4_Bkg_EB_,
                        meEleISO_chIso_MTD_5_Bkg_EB_,
                        meEleISO_chIso_MTD_6_Bkg_EB_,
                        meEleISO_chIso_MTD_7_Bkg_EB_};
  rel_ch_iso_EB_list_Bkg = {meEleISO_rel_chIso_MTD_1_Bkg_EB_,
                            meEleISO_rel_chIso_MTD_2_Bkg_EB_,
                            meEleISO_rel_chIso_MTD_3_Bkg_EB_,
                            meEleISO_rel_chIso_MTD_4_Bkg_EB_,
                            meEleISO_rel_chIso_MTD_5_Bkg_EB_,
                            meEleISO_rel_chIso_MTD_6_Bkg_EB_,
                            meEleISO_rel_chIso_MTD_7_Bkg_EB_};

  Ntracks_EB_list_Significance_Bkg = {
      meEleISO_Ntracks_MTD_4sigma_Bkg_EB_, meEleISO_Ntracks_MTD_3sigma_Bkg_EB_, meEleISO_Ntracks_MTD_2sigma_Bkg_EB_};
  ch_iso_EB_list_Significance_Bkg = {
      meEleISO_chIso_MTD_4sigma_Bkg_EB_, meEleISO_chIso_MTD_3sigma_Bkg_EB_, meEleISO_chIso_MTD_2sigma_Bkg_EB_};
  rel_ch_iso_EB_list_Significance_Bkg = {meEleISO_rel_chIso_MTD_4sigma_Bkg_EB_,
                                         meEleISO_rel_chIso_MTD_3sigma_Bkg_EB_,
                                         meEleISO_rel_chIso_MTD_2sigma_Bkg_EB_};

  Ntracks_EE_list_Bkg = {meEleISO_Ntracks_MTD_1_Bkg_EE_,
                         meEleISO_Ntracks_MTD_2_Bkg_EE_,
                         meEleISO_Ntracks_MTD_3_Bkg_EE_,
                         meEleISO_Ntracks_MTD_4_Bkg_EE_,
                         meEleISO_Ntracks_MTD_5_Bkg_EE_,
                         meEleISO_Ntracks_MTD_6_Bkg_EE_,
                         meEleISO_Ntracks_MTD_7_Bkg_EE_};
  ch_iso_EE_list_Bkg = {meEleISO_chIso_MTD_1_Bkg_EE_,
                        meEleISO_chIso_MTD_2_Bkg_EE_,
                        meEleISO_chIso_MTD_3_Bkg_EE_,
                        meEleISO_chIso_MTD_4_Bkg_EE_,
                        meEleISO_chIso_MTD_5_Bkg_EE_,
                        meEleISO_chIso_MTD_6_Bkg_EE_,
                        meEleISO_chIso_MTD_7_Bkg_EE_};
  rel_ch_iso_EE_list_Bkg = {meEleISO_rel_chIso_MTD_1_Bkg_EE_,
                            meEleISO_rel_chIso_MTD_2_Bkg_EE_,
                            meEleISO_rel_chIso_MTD_3_Bkg_EE_,
                            meEleISO_rel_chIso_MTD_4_Bkg_EE_,
                            meEleISO_rel_chIso_MTD_5_Bkg_EE_,
                            meEleISO_rel_chIso_MTD_6_Bkg_EE_,
                            meEleISO_rel_chIso_MTD_7_Bkg_EE_};

  Ntracks_EE_list_Significance_Bkg = {
      meEleISO_Ntracks_MTD_4sigma_Bkg_EE_, meEleISO_Ntracks_MTD_3sigma_Bkg_EE_, meEleISO_Ntracks_MTD_2sigma_Bkg_EE_};
  ch_iso_EE_list_Significance_Bkg = {
      meEleISO_chIso_MTD_4sigma_Bkg_EE_, meEleISO_chIso_MTD_3sigma_Bkg_EE_, meEleISO_chIso_MTD_2sigma_Bkg_EE_};
  rel_ch_iso_EE_list_Significance_Bkg = {meEleISO_rel_chIso_MTD_4sigma_Bkg_EE_,
                                         meEleISO_rel_chIso_MTD_3sigma_Bkg_EE_,
                                         meEleISO_rel_chIso_MTD_2sigma_Bkg_EE_};

  Ele_pT_MTD_EB_list_Bkg = {meEle_pt_MTD_1_Bkg_EB_,
                            meEle_pt_MTD_2_Bkg_EB_,
                            meEle_pt_MTD_3_Bkg_EB_,
                            meEle_pt_MTD_4_Bkg_EB_,
                            meEle_pt_MTD_5_Bkg_EB_,
                            meEle_pt_MTD_6_Bkg_EB_,
                            meEle_pt_MTD_7_Bkg_EB_};
  Ele_eta_MTD_EB_list_Bkg = {meEle_eta_MTD_1_Bkg_EB_,
                             meEle_eta_MTD_2_Bkg_EB_,
                             meEle_eta_MTD_3_Bkg_EB_,
                             meEle_eta_MTD_4_Bkg_EB_,
                             meEle_eta_MTD_5_Bkg_EB_,
                             meEle_eta_MTD_6_Bkg_EB_,
                             meEle_eta_MTD_7_Bkg_EB_};
  Ele_phi_MTD_EB_list_Bkg = {meEle_phi_MTD_1_Bkg_EB_,
                             meEle_phi_MTD_2_Bkg_EB_,
                             meEle_phi_MTD_3_Bkg_EB_,
                             meEle_phi_MTD_4_Bkg_EB_,
                             meEle_phi_MTD_5_Bkg_EB_,
                             meEle_phi_MTD_6_Bkg_EB_,
                             meEle_phi_MTD_7_Bkg_EB_};

  Ele_pT_MTD_EB_list_Significance_Bkg = {
      meEle_pt_MTD_4sigma_Bkg_EB_, meEle_pt_MTD_3sigma_Bkg_EB_, meEle_pt_MTD_2sigma_Bkg_EB_};
  Ele_eta_MTD_EB_list_Significance_Bkg = {
      meEle_eta_MTD_4sigma_Bkg_EB_, meEle_eta_MTD_3sigma_Bkg_EB_, meEle_eta_MTD_2sigma_Bkg_EB_};
  Ele_phi_MTD_EB_list_Significance_Bkg = {
      meEle_phi_MTD_4sigma_Bkg_EB_, meEle_phi_MTD_3sigma_Bkg_EB_, meEle_phi_MTD_2sigma_Bkg_EB_};

  Ele_pT_MTD_EE_list_Bkg = {meEle_pt_MTD_1_Bkg_EE_,
                            meEle_pt_MTD_2_Bkg_EE_,
                            meEle_pt_MTD_3_Bkg_EE_,
                            meEle_pt_MTD_4_Bkg_EE_,
                            meEle_pt_MTD_5_Bkg_EE_,
                            meEle_pt_MTD_6_Bkg_EE_,
                            meEle_pt_MTD_7_Bkg_EE_};
  Ele_eta_MTD_EE_list_Bkg = {meEle_eta_MTD_1_Bkg_EE_,
                             meEle_eta_MTD_2_Bkg_EE_,
                             meEle_eta_MTD_3_Bkg_EE_,
                             meEle_eta_MTD_4_Bkg_EE_,
                             meEle_eta_MTD_5_Bkg_EE_,
                             meEle_eta_MTD_6_Bkg_EE_,
                             meEle_eta_MTD_7_Bkg_EE_};
  Ele_phi_MTD_EE_list_Bkg = {meEle_phi_MTD_1_Bkg_EE_,
                             meEle_phi_MTD_2_Bkg_EE_,
                             meEle_phi_MTD_3_Bkg_EE_,
                             meEle_phi_MTD_4_Bkg_EE_,
                             meEle_phi_MTD_5_Bkg_EE_,
                             meEle_phi_MTD_6_Bkg_EE_,
                             meEle_phi_MTD_7_Bkg_EE_};

  Ele_pT_MTD_EE_list_Significance_Bkg = {
      meEle_pt_MTD_4sigma_Bkg_EE_, meEle_pt_MTD_3sigma_Bkg_EE_, meEle_pt_MTD_2sigma_Bkg_EE_};
  Ele_eta_MTD_EE_list_Significance_Bkg = {
      meEle_eta_MTD_4sigma_Bkg_EE_, meEle_eta_MTD_3sigma_Bkg_EE_, meEle_eta_MTD_2sigma_Bkg_EE_};
  Ele_phi_MTD_EE_list_Significance_Bkg = {
      meEle_phi_MTD_4sigma_Bkg_EE_, meEle_phi_MTD_3sigma_Bkg_EE_, meEle_phi_MTD_2sigma_Bkg_EE_};

  // SIM CASE

  Ntracks_sim_EB_list_Bkg = {meEleISO_Ntracks_MTD_sim_1_Bkg_EB_,
                             meEleISO_Ntracks_MTD_sim_2_Bkg_EB_,
                             meEleISO_Ntracks_MTD_sim_3_Bkg_EB_,
                             meEleISO_Ntracks_MTD_sim_4_Bkg_EB_,
                             meEleISO_Ntracks_MTD_sim_5_Bkg_EB_,
                             meEleISO_Ntracks_MTD_sim_6_Bkg_EB_,
                             meEleISO_Ntracks_MTD_sim_7_Bkg_EB_};
  ch_iso_sim_EB_list_Bkg = {meEleISO_chIso_MTD_sim_1_Bkg_EB_,
                            meEleISO_chIso_MTD_sim_2_Bkg_EB_,
                            meEleISO_chIso_MTD_sim_3_Bkg_EB_,
                            meEleISO_chIso_MTD_sim_4_Bkg_EB_,
                            meEleISO_chIso_MTD_sim_5_Bkg_EB_,
                            meEleISO_chIso_MTD_sim_6_Bkg_EB_,
                            meEleISO_chIso_MTD_sim_7_Bkg_EB_};
  rel_ch_iso_sim_EB_list_Bkg = {meEleISO_rel_chIso_MTD_sim_1_Bkg_EB_,
                                meEleISO_rel_chIso_MTD_sim_2_Bkg_EB_,
                                meEleISO_rel_chIso_MTD_sim_3_Bkg_EB_,
                                meEleISO_rel_chIso_MTD_sim_4_Bkg_EB_,
                                meEleISO_rel_chIso_MTD_sim_5_Bkg_EB_,
                                meEleISO_rel_chIso_MTD_sim_6_Bkg_EB_,
                                meEleISO_rel_chIso_MTD_sim_7_Bkg_EB_};

  Ntracks_sim_EB_list_Significance_Bkg = {meEleISO_Ntracks_MTD_sim_4sigma_Bkg_EB_,
                                          meEleISO_Ntracks_MTD_sim_3sigma_Bkg_EB_,
                                          meEleISO_Ntracks_MTD_sim_2sigma_Bkg_EB_};
  ch_iso_sim_EB_list_Significance_Bkg = {meEleISO_chIso_MTD_sim_4sigma_Bkg_EB_,
                                         meEleISO_chIso_MTD_sim_3sigma_Bkg_EB_,
                                         meEleISO_chIso_MTD_sim_2sigma_Bkg_EB_};
  rel_ch_iso_sim_EB_list_Significance_Bkg = {meEleISO_rel_chIso_MTD_sim_4sigma_Bkg_EB_,
                                             meEleISO_rel_chIso_MTD_sim_3sigma_Bkg_EB_,
                                             meEleISO_rel_chIso_MTD_sim_2sigma_Bkg_EB_};

  Ntracks_sim_EE_list_Bkg = {meEleISO_Ntracks_MTD_sim_1_Bkg_EE_,
                             meEleISO_Ntracks_MTD_sim_2_Bkg_EE_,
                             meEleISO_Ntracks_MTD_sim_3_Bkg_EE_,
                             meEleISO_Ntracks_MTD_sim_4_Bkg_EE_,
                             meEleISO_Ntracks_MTD_sim_5_Bkg_EE_,
                             meEleISO_Ntracks_MTD_sim_6_Bkg_EE_,
                             meEleISO_Ntracks_MTD_sim_7_Bkg_EE_};
  ch_iso_sim_EE_list_Bkg = {meEleISO_chIso_MTD_sim_1_Bkg_EE_,
                            meEleISO_chIso_MTD_sim_2_Bkg_EE_,
                            meEleISO_chIso_MTD_sim_3_Bkg_EE_,
                            meEleISO_chIso_MTD_sim_4_Bkg_EE_,
                            meEleISO_chIso_MTD_sim_5_Bkg_EE_,
                            meEleISO_chIso_MTD_sim_6_Bkg_EE_,
                            meEleISO_chIso_MTD_sim_7_Bkg_EE_};
  rel_ch_iso_sim_EE_list_Bkg = {meEleISO_rel_chIso_MTD_sim_1_Bkg_EE_,
                                meEleISO_rel_chIso_MTD_sim_2_Bkg_EE_,
                                meEleISO_rel_chIso_MTD_sim_3_Bkg_EE_,
                                meEleISO_rel_chIso_MTD_sim_4_Bkg_EE_,
                                meEleISO_rel_chIso_MTD_sim_5_Bkg_EE_,
                                meEleISO_rel_chIso_MTD_sim_6_Bkg_EE_,
                                meEleISO_rel_chIso_MTD_sim_7_Bkg_EE_};

  Ntracks_sim_EE_list_Significance_Bkg = {meEleISO_Ntracks_MTD_sim_4sigma_Bkg_EE_,
                                          meEleISO_Ntracks_MTD_sim_3sigma_Bkg_EE_,
                                          meEleISO_Ntracks_MTD_sim_2sigma_Bkg_EE_};
  ch_iso_sim_EE_list_Significance_Bkg = {meEleISO_chIso_MTD_sim_4sigma_Bkg_EE_,
                                         meEleISO_chIso_MTD_sim_3sigma_Bkg_EE_,
                                         meEleISO_chIso_MTD_sim_2sigma_Bkg_EE_};
  rel_ch_iso_sim_EE_list_Significance_Bkg = {meEleISO_rel_chIso_MTD_sim_4sigma_Bkg_EE_,
                                             meEleISO_rel_chIso_MTD_sim_3sigma_Bkg_EE_,
                                             meEleISO_rel_chIso_MTD_sim_2sigma_Bkg_EE_};

  Ele_pT_sim_MTD_EB_list_Bkg = {meEle_pt_sim_MTD_1_Bkg_EB_,
                                meEle_pt_sim_MTD_2_Bkg_EB_,
                                meEle_pt_sim_MTD_3_Bkg_EB_,
                                meEle_pt_sim_MTD_4_Bkg_EB_,
                                meEle_pt_sim_MTD_5_Bkg_EB_,
                                meEle_pt_sim_MTD_6_Bkg_EB_,
                                meEle_pt_sim_MTD_7_Bkg_EB_};

  Ele_pT_sim_MTD_EB_list_Significance_Bkg = {
      meEle_pt_sim_MTD_4sigma_Bkg_EB_, meEle_pt_sim_MTD_3sigma_Bkg_EB_, meEle_pt_sim_MTD_2sigma_Bkg_EB_};

  Ele_pT_sim_MTD_EE_list_Bkg = {meEle_pt_sim_MTD_1_Bkg_EE_,
                                meEle_pt_sim_MTD_2_Bkg_EE_,
                                meEle_pt_sim_MTD_3_Bkg_EE_,
                                meEle_pt_sim_MTD_4_Bkg_EE_,
                                meEle_pt_sim_MTD_5_Bkg_EE_,
                                meEle_pt_sim_MTD_6_Bkg_EE_,
                                meEle_pt_sim_MTD_7_Bkg_EE_};

  Ele_pT_sim_MTD_EE_list_Significance_Bkg = {
      meEle_pt_sim_MTD_4sigma_Bkg_EE_, meEle_pt_sim_MTD_3sigma_Bkg_EE_, meEle_pt_sim_MTD_2sigma_Bkg_EE_};

  // dt distribution hist vecotrs

  general_pT_list = {meEle_dt_general_pT_1,
                     meEle_dt_general_pT_2,
                     meEle_dt_general_pT_3,
                     meEle_dt_general_pT_4,
                     meEle_dt_general_pT_5,
                     meEle_dt_general_pT_6,
                     meEle_dt_general_pT_7,
                     meEle_dt_general_pT_8,
                     meEle_dt_general_pT_9};

  general_pT_Signif_list = {meEle_dtSignif_general_pT_1,
                            meEle_dtSignif_general_pT_2,
                            meEle_dtSignif_general_pT_3,
                            meEle_dtSignif_general_pT_4,
                            meEle_dtSignif_general_pT_5,
                            meEle_dtSignif_general_pT_6,
                            meEle_dtSignif_general_pT_7,
                            meEle_dtSignif_general_pT_8,
                            meEle_dtSignif_general_pT_9};

  general_eta_list = {meEle_dt_general_eta_1,
                      meEle_dt_general_eta_2,
                      meEle_dt_general_eta_3,
                      meEle_dt_general_eta_4,
                      meEle_dt_general_eta_5};

  general_eta_Signif_list = {meEle_dtSignif_general_eta_1,
                             meEle_dtSignif_general_eta_2,
                             meEle_dtSignif_general_eta_3,
                             meEle_dtSignif_general_eta_4,
                             meEle_dtSignif_general_eta_5};
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------

void MtdEleIsoValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/ElectronIso");
  desc.add<edm::InputTag>("inputTagG", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("inputTagT", edm::InputTag("trackExtenderWithMTD"));
  desc.add<edm::InputTag>(
      "inputTag_vtx",
      edm::InputTag("offlinePrimaryVertices4D"));  //  "offlinePrimaryVertices4D" or "offlinePrimaryVertices" (3D case)
  desc.add<edm::InputTag>("inputTagH", edm::InputTag("generatorSmeared"));

  desc.add<edm::InputTag>(
      "inputEle_EB",
      edm::InputTag("gedGsfElectrons"));  // Adding the elecollection, barrel ecal and track driven electrons
  //desc.add<edm::InputTag>("inputEle", edm::InputTag("ecalDrivenGsfElectrons")); // barrel + endcap, but without track seeded electrons
  desc.add<edm::InputTag>("inputEle_EE", edm::InputTag("ecalDrivenGsfElectronsHGC"));  // only endcap electrons
  desc.add<edm::InputTag>("inputGenP", edm::InputTag("genParticles"));
  desc.add<edm::InputTag>("SimTag", edm::InputTag("mix", "MergedTrackTruth"));                            // From Aurora
  desc.add<edm::InputTag>("TPtoRecoTrackAssoc", edm::InputTag("trackingParticleRecoTrackAsssociation"));  // From Aurora

  desc.add<edm::InputTag>("tmtd", edm::InputTag("trackExtenderWithMTD:generalTracktmtd"));
  desc.add<edm::InputTag>("sigmatmtd", edm::InputTag("trackExtenderWithMTD:generalTracksigmatmtd"));
  desc.add<edm::InputTag>("t0Src", edm::InputTag("trackExtenderWithMTD:generalTrackt0"));
  desc.add<edm::InputTag>("sigmat0Src", edm::InputTag("trackExtenderWithMTD:generalTracksigmat0"));
  desc.add<edm::InputTag>("trackAssocSrc", edm::InputTag("trackExtenderWithMTD:generalTrackassoc"))
      ->setComment("Association between General and MTD Extended tracks");
  desc.add<edm::InputTag>("pathLengthSrc", edm::InputTag("trackExtenderWithMTD:generalTrackPathLength"));
  desc.add<edm::InputTag>("t0SafePID", edm::InputTag("tofPID:t0safe"));
  desc.add<edm::InputTag>("sigmat0SafePID", edm::InputTag("tofPID:sigmat0safe"));
  desc.add<edm::InputTag>("sigmat0PID", edm::InputTag("tofPID:sigmat0"));
  desc.add<edm::InputTag>("t0PID", edm::InputTag("tofPID:t0"));
  desc.add<edm::InputTag>("trackMVAQual", edm::InputTag("mtdTrackQualityMVA:mtdQualMVA"));
  desc.add<double>("trackMinimumPt", 1.0);  // [GeV]
  desc.add<double>("trackMinimumEta", 1.5);
  desc.add<double>("trackMaximumEta", 3.2);
  desc.add<double>("rel_iso_cut", 0.08);
  //desc.add<std::vector<MonitorElement*>>("Ntracks_EB_list_Sig_test", {meEleISO_Ntracks_MTD_1_Sig_EB_,meEleISO_Ntracks_MTD_2_Sig_EB_,meEleISO_Ntracks_MTD_3_Sig_EB_,meEleISO_Ntracks_MTD_4_Sig_EB_,meEleISO_Ntracks_MTD_5_Sig_EB_,meEleISO_Ntracks_MTD_6_Sig_EB_,meEleISO_Ntracks_MTD_7_Sig_EB_}); // example that does not work...
  desc.add<bool>("optionTrackMatchToPV", false);
  desc.add<bool>("option_dtToPV", false);
  desc.add<bool>("option_dtToTrack", true);
  desc.add<bool>("option_dtDistributions", false);

  descriptions.add("mtdEleIsoValid", desc);
}

bool MtdEleIsoValidation::pdgCheck(int pdg) {
  bool pass;
  pdg = std::abs(pdg);
  if (pdg == 11 || pdg == 15 || pdg == 23 || pdg == 24) {
    pass = true;
  } else {
    pass = false;
  }
  return pass;
}

DEFINE_FWK_MODULE(MtdEleIsoValidation);

//*/
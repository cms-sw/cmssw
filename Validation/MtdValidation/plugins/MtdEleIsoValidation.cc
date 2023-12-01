#include <string>
#include <numeric>
#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>

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
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

// Adding header files for electrons
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

// eff vs PU test libraries
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimTracker/VertexAssociation/interface/calculateVertexSharedTracks.h"
#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"
#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"

class MtdEleIsoValidation : public DQMEDAnalyzer {
public:
  explicit MtdEleIsoValidation(const edm::ParameterSet&);
  ~MtdEleIsoValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ------------ member data ------------

  const std::string folder_;
  const float trackMinPt_;
  const float trackMinEta_;
  const float trackMaxEta_;
  const double rel_iso_cut_;

  const bool track_match_PV_;
  const bool dt_sig_track_;
  const bool optionalPlots_;

  const float min_dR_cut;
  const float max_dR_cut;
  const float min_pt_cut_EB;
  const float min_pt_cut_EE;
  const float max_dz_cut_EB;
  const float max_dz_cut_EE;
  const float max_dz_vtx_cut;
  const float max_dxy_vtx_cut;
  const float min_strip_cut;
  const float min_track_mtd_mva_cut;
  const std::vector<double> max_dt_vtx_cut{0.30, 0.27, 0.24, 0.21, 0.18, 0.15, 0.12};
  const std::vector<double> max_dt_track_cut{0.30, 0.27, 0.24, 0.21, 0.18, 0.15, 0.12};
  const std::vector<double> max_dt_significance_cut{4.0, 3.0, 2.0};
  const std::vector<double> pT_bins_dt_distrb{10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  const std::vector<double> eta_bins_dt_distrib{0.0, 0.5, 1.0, 1.5, 2.0, 2.4, 2.7, 3};
  static constexpr double avg_sim_sigTrk_t_err = 0.03239;
  static constexpr double avg_sim_PUtrack_t_err = 0.03465;

  edm::EDGetTokenT<reco::TrackCollection> GenRecTrackToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> RecVertexToken_;

  edm::EDGetTokenT<reco::GsfElectronCollection> GsfElectronToken_EB_;
  edm::EDGetTokenT<reco::GsfElectronCollection> GsfElectronToken_EE_;
  edm::EDGetTokenT<reco::GenParticleCollection> GenParticleToken_;

  edm::EDGetTokenT<edm::ValueMap<float>> t0PidToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> Sigmat0PidToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> trackMVAQualToken_;

  edm::EDGetTokenT<reco::RecoToSimCollection> recoToSimAssociationToken_;

  // Signal histograms

  MonitorElement* meEle_no_dt_check_;
  MonitorElement* meTrk_genMatch_check_;

  MonitorElement* meEle_avg_error_SigTrk_check_;
  MonitorElement* meEle_avg_error_PUTrk_check_;
  MonitorElement* meEle_avg_error_vtx_check_;

  // Adding histograms for barrel electrons
  MonitorElement* meEleISO_Ntracks_Sig_EB_;
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

  MonitorElement* meEleISO_Ntracks_gen_Sig_EB_;
  MonitorElement* meEleISO_chIso_gen_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_gen_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_2sigma_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_2sigma_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_2sigma_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_3sigma_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_3sigma_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_3sigma_Sig_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_4sigma_Sig_EB_;
  MonitorElement* meEleISO_chIso_MTD_4sigma_Sig_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_4sigma_Sig_EB_;

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
  MonitorElement* meEle_pt_sim_tot_Sig_EB_;  // for GEN case is the same
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

  MonitorElement* meEle_pt_gen_Sig_EB_;
  MonitorElement* meEle_eta_gen_Sig_EB_;
  MonitorElement* meEle_phi_gen_Sig_EB_;

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

  // Adding histograms for endcap electrons
  MonitorElement* meEleISO_Ntracks_Sig_EE_;
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

  MonitorElement* meEleISO_Ntracks_gen_Sig_EE_;
  MonitorElement* meEleISO_chIso_gen_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_gen_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_2sigma_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_2sigma_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_2sigma_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_3sigma_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_3sigma_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_3sigma_Sig_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_4sigma_Sig_EE_;
  MonitorElement* meEleISO_chIso_MTD_4sigma_Sig_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_4sigma_Sig_EE_;

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

  MonitorElement* meEle_pt_gen_Sig_EE_;
  MonitorElement* meEle_eta_gen_Sig_EE_;
  MonitorElement* meEle_phi_gen_Sig_EE_;

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
  // Adding histograms for barrel electrons
  MonitorElement* meEleISO_Ntracks_Bkg_EB_;
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

  MonitorElement* meEleISO_Ntracks_gen_Bkg_EB_;
  MonitorElement* meEleISO_chIso_gen_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_gen_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_2sigma_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_2sigma_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_2sigma_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_3sigma_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_3sigma_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_3sigma_Bkg_EB_;

  MonitorElement* meEleISO_Ntracks_MTD_4sigma_Bkg_EB_;
  MonitorElement* meEleISO_chIso_MTD_4sigma_Bkg_EB_;
  MonitorElement* meEleISO_rel_chIso_MTD_4sigma_Bkg_EB_;

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

  MonitorElement* meEle_pt_gen_Bkg_EB_;
  MonitorElement* meEle_eta_gen_Bkg_EB_;
  MonitorElement* meEle_phi_gen_Bkg_EB_;

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

  // Adding histograms for endcap electrons
  MonitorElement* meEleISO_Ntracks_Bkg_EE_;
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

  MonitorElement* meEleISO_Ntracks_gen_Bkg_EE_;
  MonitorElement* meEleISO_chIso_gen_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_gen_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_2sigma_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_2sigma_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_2sigma_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_3sigma_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_3sigma_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_3sigma_Bkg_EE_;

  MonitorElement* meEleISO_Ntracks_MTD_4sigma_Bkg_EE_;
  MonitorElement* meEleISO_chIso_MTD_4sigma_Bkg_EE_;
  MonitorElement* meEleISO_rel_chIso_MTD_4sigma_Bkg_EE_;

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

  MonitorElement* meEle_pt_gen_Bkg_EE_;
  MonitorElement* meEle_eta_gen_Bkg_EE_;
  MonitorElement* meEle_phi_gen_Bkg_EE_;

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
      dt_sig_track_(iConfig.getParameter<bool>("option_dtToTrack")),
      optionalPlots_(iConfig.getParameter<bool>("option_plots")),
      min_dR_cut(iConfig.getParameter<double>("min_dR_cut")),
      max_dR_cut(iConfig.getParameter<double>("max_dR_cut")),
      min_pt_cut_EB(iConfig.getParameter<double>("min_pt_cut_EB")),
      min_pt_cut_EE(iConfig.getParameter<double>("min_pt_cut_EE")),
      max_dz_cut_EB(iConfig.getParameter<double>("max_dz_cut_EB")),
      max_dz_cut_EE(iConfig.getParameter<double>("max_dz_cut_EE")),
      max_dz_vtx_cut(iConfig.getParameter<double>("max_dz_vtx_cut")),
      max_dxy_vtx_cut(iConfig.getParameter<double>("max_dxy_vtx_cut")),
      min_strip_cut(iConfig.getParameter<double>("min_strip_cut")),
      min_track_mtd_mva_cut(iConfig.getParameter<double>("min_track_mtd_mva_cut")) {
  GenRecTrackToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTagG"));
  RecVertexToken_ =
      consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("inputTag_vtx"));  // Vtx 4D collection

  GsfElectronToken_EB_ = consumes<reco::GsfElectronCollection>(
      iConfig.getParameter<edm::InputTag>("inputEle_EB"));  // Barrel electron collection input/token
  GsfElectronToken_EE_ = consumes<reco::GsfElectronCollection>(
      iConfig.getParameter<edm::InputTag>("inputEle_EE"));  // Endcap electron collection input/token

  t0PidToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("t0PID"));
  Sigmat0PidToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmat0PID"));
  trackMVAQualToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("trackMVAQual"));

  recoToSimAssociationToken_ =
      consumes<reco::RecoToSimCollection>(iConfig.getParameter<edm::InputTag>("TPtoRecoTrackAssoc"));
}

MtdEleIsoValidation::~MtdEleIsoValidation() {}

// ------------ method called for each event  ------------
void MtdEleIsoValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace geant_units::operators;
  using namespace std;

  auto GenRecTrackHandle = iEvent.getHandle(GenRecTrackToken_);

  auto VertexHandle = iEvent.getHandle(RecVertexToken_);
  std::vector<reco::Vertex> vertices = *VertexHandle;

  const auto& t0Pid = iEvent.get(t0PidToken_);
  const auto& Sigmat0Pid = iEvent.get(Sigmat0PidToken_);
  const auto& mtdQualMVA = iEvent.get(trackMVAQualToken_);

  auto eleHandle_EB = makeValid(iEvent.getHandle(GsfElectronToken_EB_));
  reco::GsfElectronCollection eleColl_EB = *(eleHandle_EB.product());

  auto eleHandle_EE = makeValid(iEvent.getHandle(GsfElectronToken_EE_));
  reco::GsfElectronCollection eleColl_EE = *(eleHandle_EE.product());

  auto recoToSimH = makeValid(iEvent.getHandle(recoToSimAssociationToken_));
  const reco::RecoToSimCollection* r2s_ = recoToSimH.product();

  // Creating combined electron collection
  std::vector<reco::GsfElectron> localEleCollection;
  localEleCollection.reserve(eleColl_EB.size() + eleColl_EE.size());
  for (const auto& ele_EB : eleColl_EB) {
    if (ele_EB.isEB()) {
      localEleCollection.emplace_back(ele_EB);
    }
  }
  for (const auto& ele_EE : eleColl_EE) {
    if (ele_EE.isEE()) {
      localEleCollection.emplace_back(ele_EE);
    }
  }
  localEleCollection.shrink_to_fit();

  reco::Vertex Vtx_chosen;
  // This part has to be included, because in ~1% of the events, the "good" vertex is the 1st one not the 0th one in the collection
  for (int iVtx = 0; iVtx < (int)vertices.size(); iVtx++) {
    const reco::Vertex& vertex = vertices.at(iVtx);
    if (!vertex.isFake() && vertex.ndof() >= 4) {
      Vtx_chosen = vertex;
      break;
    }
  }

  auto pdgCheck = [](int pdg) {
    pdg = std::abs(pdg);
    return (pdg == 23 or pdg == 24 or pdg == 15 or pdg == 11);  // some electrons are mothers to themselves?
  };

  for (const auto& ele : localEleCollection) {
    bool ele_Promt = false;

    float ele_track_source_dz = std::abs(ele.gsfTrack()->dz(Vtx_chosen.position()));
    float ele_track_source_dxy = std::abs(ele.gsfTrack()->dxy(Vtx_chosen.position()));

    const reco::TrackRef ele_TrkRef = ele.core()->ctfTrack();
    double tsim_ele = -1.;
    double ele_sim_pt = -1.;
    double ele_sim_phi = -1.;
    double ele_sim_eta = -1.;

    // selecting "good" RECO electrons
    // PARAM
    if (ele.pt() < 10 || std::abs(ele.eta()) > 2.4 || ele_track_source_dz > max_dz_vtx_cut ||
        ele_track_source_dxy > max_dxy_vtx_cut)
      continue;

    // association with tracking particle to have sim info
    const reco::TrackBaseRef trkrefb(ele_TrkRef);
    auto found = r2s_->find(trkrefb);
    if (found != r2s_->end()) {
      const auto& tp = (found->val)[0];
      tsim_ele = (tp.first)->parentVertex()->position().t() * 1e9;
      ele_sim_pt = (tp.first)->pt();
      ele_sim_phi = (tp.first)->phi();
      ele_sim_eta = (tp.first)->eta();
      // check that the genParticle vector is not empty
      if (tp.first->status() != -99) {
        const auto genParticle = *(tp.first->genParticles()[0]);
        // check if prompt (not from hadron, muon, or tau decay) and final state
        // or if is a direct decay product of a prompt tau and is final state
        if ((genParticle.isPromptFinalState() or genParticle.isDirectPromptTauDecayProductFinalState()) and
            pdgCheck(genParticle.mother()->pdgId())) {
          ele_Promt = true;
          // TODO get simtrackster from mtd, simtrack to tp and check that a recocluster was there
        }
      }
    }

    math::XYZVector EleSigTrackMomentumAtVtx = ele.gsfTrack()->momentum();
    double EleSigTrackEtaAtVtx = ele.gsfTrack()->eta();

    double ele_sigTrkTime = -1;
    double ele_sigTrkTimeErr = -1;
    double ele_sigTrkMtdMva = -1;

    // if we found a track match, we add MTD timing information for it
    if (ele_TrkRef.isNonnull()) {
      // track pT/dz cuts
      bool Barrel_ele = ele.isEB();
      float min_pt_cut = Barrel_ele ? min_pt_cut_EB : min_pt_cut_EE;
      float max_dz_cut = Barrel_ele ? max_dz_cut_EB : max_dz_cut_EE;

      ele_sigTrkTime = t0Pid[ele_TrkRef];
      ele_sigTrkMtdMva = mtdQualMVA[ele_TrkRef];
      ele_sigTrkTimeErr = (ele_sigTrkMtdMva > min_track_mtd_mva_cut) ? Sigmat0Pid[ele_TrkRef] : -1;

      meEle_avg_error_SigTrk_check_->Fill(ele_sigTrkTimeErr);

      if (ele_Promt) {
        // For signal (promt)
        if (Barrel_ele) {
          // All selected electron information for efficiency plots later
          meEle_pt_tot_Sig_EB_->Fill(ele.pt());
          meEle_pt_sim_tot_Sig_EB_->Fill(ele_sim_pt);
          meEle_eta_tot_Sig_EB_->Fill(std::abs(ele.eta()));
          meEle_phi_tot_Sig_EB_->Fill(ele.phi());
        } else {
          // All selected electron information for efficiency plots later
          meEle_pt_tot_Sig_EE_->Fill(ele.pt());
          meEle_pt_sim_tot_Sig_EE_->Fill(ele_sim_pt);
          meEle_eta_tot_Sig_EE_->Fill(std::abs(ele.eta()));
          meEle_phi_tot_Sig_EE_->Fill(ele.phi());
        }
      } else {
        // For background (non-promt)
        if (Barrel_ele) {
          meEle_pt_tot_Bkg_EB_->Fill(ele.pt());
          meEle_pt_sim_tot_Bkg_EB_->Fill(ele_sim_pt);
          meEle_eta_tot_Bkg_EB_->Fill(std::abs(ele.eta()));
          meEle_phi_tot_Bkg_EB_->Fill(ele.phi());
        } else {
          meEle_pt_tot_Bkg_EE_->Fill(ele.pt());
          meEle_pt_sim_tot_Bkg_EE_->Fill(ele_sim_pt);
          meEle_eta_tot_Bkg_EE_->Fill(std::abs(ele.eta()));
          meEle_phi_tot_Bkg_EE_->Fill(ele.phi());
        }
      }

      int N_tracks_noMTD = 0;
      double pT_sum_noMTD = 0;
      double rel_pT_sum_noMTD = 0;
      std::vector<int> N_tracks_MTD{0, 0, 0, 0, 0, 0, 0};
      std::vector<double> pT_sum_MTD{0, 0, 0, 0, 0, 0, 0};
      std::vector<double> rel_pT_sum_MTD{0, 0, 0, 0, 0, 0, 0};

      std::vector<int> N_tracks_sim_MTD{0, 0, 0, 0, 0, 0, 0};
      std::vector<double> pT_sum_sim_MTD{0, 0, 0, 0, 0, 0, 0};
      std::vector<double> rel_pT_sum_sim_MTD{0, 0, 0, 0, 0, 0, 0};
      int N_tracks_gen = 0;
      double pT_sum_gen = 0;
      double rel_pT_sum_gen = 0;

      std::vector<int> N_tracks_MTD_significance{0, 0, 0};
      std::vector<double> pT_sum_MTD_significance{0, 0, 0};
      std::vector<double> rel_pT_sum_MTD_significance{0, 0, 0};

      std::vector<int> N_tracks_sim_MTD_significance{0, 0, 0};
      std::vector<double> pT_sum_sim_MTD_significance{0, 0, 0};
      std::vector<double> rel_pT_sum_sim_MTD_significance{0, 0, 0};

      int general_index = 0;
      for (const auto& trackGen : *GenRecTrackHandle) {
        const reco::TrackRef trackref_general(GenRecTrackHandle, general_index);
        general_index++;

        // Skip electron track
        if (trackref_general == ele_TrkRef)
          continue;

        if (trackGen.pt() < min_pt_cut) {
          continue;
        }
        if (std::abs(trackGen.vz() - ele.gsfTrack()->vz()) > max_dz_cut) {
          continue;
        }

        // cut for general track matching to PV
        if (track_match_PV_) {
          if (Vtx_chosen.trackWeight(trackref_general) < 0.5) {
            continue;
          }
        }

        double dR = reco::deltaR(trackGen.momentum(), EleSigTrackMomentumAtVtx);
        double deta = std::abs(trackGen.eta() - EleSigTrackEtaAtVtx);

        // restrict to tracks in the isolation cone
        if (dR < min_dR_cut || dR > max_dR_cut || deta < min_strip_cut)
          continue;

        // no MTD case
        ++N_tracks_noMTD;
        pT_sum_noMTD += trackGen.pt();

        // MTD case
        const reco::TrackBaseRef trkrefBase(trackref_general);
        auto TPmatched = r2s_->find(trkrefBase);
        double tsim_trk = -1.;
        double trk_ptSim = -1.;
        bool genMatched = false;
        if (TPmatched != r2s_->end()) {
          // reco track matched to a TP
          const auto& tp = (TPmatched->val)[0];
          tsim_trk = (tp.first)->parentVertex()->position().t() * 1e9;
          trk_ptSim = (tp.first)->pt();
          // check that the genParticle vector is not empty
          if (tp.first->status() != -99) {
            genMatched = true;
            meTrk_genMatch_check_->Fill(1);
          } else {
            meTrk_genMatch_check_->Fill(0);
          }
        }

        double TrkMTDTime = t0Pid[trackref_general];
        double TrkMTDMva = mtdQualMVA[trackref_general];
        double TrkMTDTimeErr = (TrkMTDMva > min_track_mtd_mva_cut) ? Sigmat0Pid[trackref_general] : -1;

        meEle_avg_error_PUTrk_check_->Fill(TrkMTDTimeErr);

        // MTD GEN case
        if (genMatched) {
          N_tracks_gen++;
          pT_sum_gen += trk_ptSim;
        }

        // dt with the track
        if (dt_sig_track_) {
          double dt_sigTrk = 0;
          double dt_sigTrk_signif = 0;
          double dt_sim_sigTrk = 0;
          double dt_sim_sigTrk_signif = 0;

          // MTD SIM CASE
          if (std::abs(tsim_trk) > 0 && std::abs(tsim_ele) > 0 && trk_ptSim > 0) {
            dt_sim_sigTrk = std::abs(tsim_trk - tsim_ele);
            dt_sim_sigTrk_signif = dt_sim_sigTrk / std::sqrt(avg_sim_PUtrack_t_err * avg_sim_PUtrack_t_err +
                                                             avg_sim_sigTrk_t_err * avg_sim_sigTrk_t_err);

            if (optionalPlots_) {
              // absolute timing cuts
              for (long unsigned int i = 0; i < N_tracks_sim_MTD.size(); i++) {
                if (dt_sim_sigTrk < max_dt_track_cut[i]) {
                  N_tracks_sim_MTD[i] = N_tracks_sim_MTD[i] + 1;
                  pT_sum_sim_MTD[i] = pT_sum_sim_MTD[i] + trk_ptSim;
                }
              }
            }
            // significance cuts
            for (long unsigned int i = 0; i < N_tracks_sim_MTD_significance.size(); i++) {
              if (dt_sim_sigTrk_signif < max_dt_significance_cut[i]) {
                N_tracks_sim_MTD_significance[i]++;
                pT_sum_sim_MTD_significance[i] += trk_ptSim;
              }
            }

          } else {
            // if there is no error for MTD information, we count the MTD isolation case same as noMTD
            if (optionalPlots_) {
              for (long unsigned int i = 0; i < N_tracks_sim_MTD.size(); i++) {
                N_tracks_sim_MTD[i] = N_tracks_sim_MTD[i] + 1;
                pT_sum_sim_MTD[i] = pT_sum_sim_MTD[i] + trk_ptSim;
              }
            }
            for (long unsigned int i = 0; i < N_tracks_sim_MTD_significance.size(); i++) {
              N_tracks_sim_MTD_significance[i]++;
              pT_sum_sim_MTD_significance[i] += trk_ptSim;
            }
          }

          // MTD reco case
          if (TrkMTDTimeErr > 0 && ele_sigTrkTimeErr > 0) {
            dt_sigTrk = std::abs(TrkMTDTime - ele_sigTrkTime);
            dt_sigTrk_signif =
                dt_sigTrk / std::sqrt(TrkMTDTimeErr * TrkMTDTimeErr + ele_sigTrkTimeErr * ele_sigTrkTimeErr);

            meEle_no_dt_check_->Fill(1);
            if (optionalPlots_) {
              // absolute timing cuts
              for (long unsigned int i = 0; i < N_tracks_MTD.size(); i++) {
                if (dt_sigTrk < max_dt_track_cut[i]) {
                  N_tracks_MTD[i] = N_tracks_MTD[i] + 1;
                  pT_sum_MTD[i] = pT_sum_MTD[i] + trackGen.pt();
                }
              }
            }
            // significance cuts
            for (long unsigned int i = 0; i < N_tracks_MTD_significance.size(); i++) {
              if (dt_sigTrk_signif < max_dt_significance_cut[i]) {
                N_tracks_MTD_significance[i]++;
                pT_sum_MTD_significance[i] += trackGen.pt();
              }
            }

          } else {
            // if there is no error for MTD information, we count the MTD isolation case same as noMTD
            if (optionalPlots_) {
              for (long unsigned int i = 0; i < N_tracks_MTD.size(); i++) {
                N_tracks_MTD[i] = N_tracks_MTD[i] + 1;          // N_tracks_noMTD
                pT_sum_MTD[i] = pT_sum_MTD[i] + trackGen.pt();  // pT sum
              }
            }
            for (long unsigned int i = 0; i < N_tracks_MTD_significance.size(); i++) {
              N_tracks_MTD_significance[i]++;
              pT_sum_MTD_significance[i] += trackGen.pt();
            }
            meEle_no_dt_check_->Fill(0);
          }

          if (optionalPlots_) {
            for (long unsigned int i = 0; i < (pT_bins_dt_distrb.size() - 1); i++) {
              //stuff general pT
              if (ele.pt() > pT_bins_dt_distrb[i] && ele.pt() < pT_bins_dt_distrb[i + 1]) {
                general_pT_list[i]->Fill(dt_sigTrk);
                general_pT_Signif_list[i]->Fill(dt_sigTrk_signif);
              }
            }

            for (long unsigned int i = 0; i < (eta_bins_dt_distrib.size() - 1); i++) {
              //stuff general eta
              if (std::abs(ele.eta()) > eta_bins_dt_distrib[i] && std::abs(ele.eta()) < eta_bins_dt_distrib[i + 1]) {
                general_eta_list[i]->Fill(dt_sigTrk);
                general_eta_Signif_list[i]->Fill(dt_sigTrk_signif);
              }
            }
          }  // End of optional dt distributions plots

          // dt with the vertex
        } else {
          double dt_vtx = 0;  // dt regular track vs vtx
          double dt_vtx_signif = 0;

          double dt_sim_vtx = 0;  // dt regular track vs vtx
          double dt_sim_vtx_signif = 0;

          // MTD SIM case
          if (std::abs(tsim_trk) > 0 && Vtx_chosen.tError() > 0 && trk_ptSim > 0) {
            dt_sim_vtx = std::abs(tsim_trk - Vtx_chosen.t());
            dt_sim_vtx_signif = dt_sim_vtx / std::sqrt(avg_sim_PUtrack_t_err * avg_sim_PUtrack_t_err +
                                                       Vtx_chosen.tError() * Vtx_chosen.tError());
            if (optionalPlots_) {
              // absolute timing cuts
              for (long unsigned int i = 0; i < N_tracks_sim_MTD.size(); i++) {
                if (dt_sim_vtx < max_dt_vtx_cut[i]) {
                  N_tracks_sim_MTD[i] = N_tracks_sim_MTD[i] + 1;
                  pT_sum_sim_MTD[i] = pT_sum_sim_MTD[i] + trk_ptSim;
                }
              }
            }
            // significance timing cuts
            for (long unsigned int i = 0; i < N_tracks_sim_MTD_significance.size(); i++) {
              if (dt_sim_vtx_signif < max_dt_significance_cut[i]) {
                N_tracks_sim_MTD_significance[i]++;
                pT_sum_sim_MTD_significance[i] += trk_ptSim;
              }
            }
          } else {
            if (optionalPlots_) {
              for (long unsigned int i = 0; i < N_tracks_sim_MTD.size(); i++) {
                N_tracks_sim_MTD[i] = N_tracks_sim_MTD[i] + 1;      // N_tracks_noMTD
                pT_sum_sim_MTD[i] = pT_sum_sim_MTD[i] + trk_ptSim;  // pT_sum_noMTD
              }
            }
            for (long unsigned int i = 0; i < N_tracks_sim_MTD_significance.size(); i++) {
              N_tracks_sim_MTD_significance[i]++;
              pT_sum_sim_MTD_significance[i] += trk_ptSim;
            }
          }

          // MTD RECO case
          if (TrkMTDTimeErr > 0 && Vtx_chosen.tError() > 0) {
            dt_vtx = std::abs(TrkMTDTime - Vtx_chosen.t());
            dt_vtx_signif =
                dt_vtx / std::sqrt(TrkMTDTimeErr * TrkMTDTimeErr + Vtx_chosen.tError() * Vtx_chosen.tError());

            meEle_no_dt_check_->Fill(1);
            meEle_avg_error_vtx_check_->Fill(Vtx_chosen.tError());
            if (optionalPlots_) {
              // absolute timing cuts
              for (long unsigned int i = 0; i < N_tracks_MTD.size(); i++) {
                if (dt_vtx < max_dt_vtx_cut[i]) {
                  N_tracks_MTD[i] = N_tracks_MTD[i] + 1;
                  pT_sum_MTD[i] = pT_sum_MTD[i] + trackGen.pt();
                }
              }
            }
            // significance timing cuts
            for (long unsigned int i = 0; i < N_tracks_MTD_significance.size(); i++) {
              if (dt_vtx_signif < max_dt_significance_cut[i]) {
                N_tracks_MTD_significance[i]++;
                pT_sum_MTD_significance[i] += trackGen.pt();
              }
            }
          } else {
            if (optionalPlots_) {
              for (long unsigned int i = 0; i < N_tracks_MTD.size(); i++) {
                N_tracks_MTD[i] = N_tracks_MTD[i] + 1;          // N_tracks_noMTD
                pT_sum_MTD[i] = pT_sum_MTD[i] + trackGen.pt();  // pT_sum_noMTD
              }
            }
            for (long unsigned int i = 0; i < N_tracks_MTD_significance.size(); i++) {
              N_tracks_MTD_significance[i]++;
              pT_sum_MTD_significance[i] += trackGen.pt();
            }
            meEle_no_dt_check_->Fill(0);
          }

          // Optional dt distribution plots
          if (optionalPlots_) {
            for (long unsigned int i = 0; i < (pT_bins_dt_distrb.size() - 1); i++) {
              //stuff general pT
              if (ele.pt() > pT_bins_dt_distrb[i] && ele.pt() < pT_bins_dt_distrb[i + 1]) {
                general_pT_list[i]->Fill(dt_vtx);
                general_pT_Signif_list[i]->Fill(dt_vtx_signif);
              }
            }

            for (long unsigned int i = 0; i < (eta_bins_dt_distrib.size() - 1); i++) {
              //stuff general eta
              if (std::abs(ele.eta()) > eta_bins_dt_distrib[i] && std::abs(ele.eta()) < eta_bins_dt_distrib[i + 1]) {
                general_eta_list[i]->Fill(dt_vtx);
                general_eta_Signif_list[i]->Fill(dt_vtx_signif);
              }
            }
          }  // End of optional dt distributions plots
        }
      }
      rel_pT_sum_noMTD = pT_sum_noMTD / ele.gsfTrack()->pt();  // rel_ch_iso calculation
      if (optionalPlots_) {
        for (long unsigned int i = 0; i < N_tracks_MTD.size(); i++) {
          rel_pT_sum_MTD[i] = pT_sum_MTD[i] / ele.gsfTrack()->pt();
          rel_pT_sum_sim_MTD[i] = pT_sum_sim_MTD[i] / ele_sim_pt;
        }
        // now compute the isolation
        rel_pT_sum_noMTD = pT_sum_noMTD / ele.gsfTrack()->pt();

        rel_pT_sum_gen = pT_sum_gen / ele.gsfTrack()->pt();
      }

      for (long unsigned int i = 0; i < N_tracks_MTD_significance.size(); i++) {
        rel_pT_sum_MTD_significance[i] = pT_sum_MTD_significance[i] / ele.gsfTrack()->pt();
        rel_pT_sum_sim_MTD_significance[i] = pT_sum_sim_MTD_significance[i] / ele_sim_pt;
      }

      if (ele_Promt) {  // promt part
        if (Barrel_ele) {
          meEleISO_Ntracks_Sig_EB_->Fill(N_tracks_noMTD);
          meEleISO_chIso_Sig_EB_->Fill(pT_sum_noMTD);
          meEleISO_rel_chIso_Sig_EB_->Fill(rel_pT_sum_noMTD);
          if (optionalPlots_) {
            for (long unsigned int j = 0; j < Ntracks_EB_list_Sig.size(); j++) {
              Ntracks_EB_list_Sig[j]->Fill(N_tracks_MTD[j]);
              ch_iso_EB_list_Sig[j]->Fill(pT_sum_MTD[j]);
              rel_ch_iso_EB_list_Sig[j]->Fill(rel_pT_sum_MTD[j]);

              Ntracks_sim_EB_list_Sig[j]->Fill(N_tracks_sim_MTD[j]);
              ch_iso_sim_EB_list_Sig[j]->Fill(pT_sum_sim_MTD[j]);
              rel_ch_iso_sim_EB_list_Sig[j]->Fill(rel_pT_sum_sim_MTD[j]);
            }
            meEleISO_Ntracks_gen_Sig_EB_->Fill(N_tracks_gen);
            meEleISO_chIso_gen_Sig_EB_->Fill(pT_sum_gen);
            meEleISO_rel_chIso_gen_Sig_EB_->Fill(rel_pT_sum_gen);
          }

          for (long unsigned int j = 0; j < Ntracks_EB_list_Significance_Sig.size(); j++) {
            Ntracks_EB_list_Significance_Sig[j]->Fill(N_tracks_MTD_significance[j]);
            ch_iso_EB_list_Significance_Sig[j]->Fill(pT_sum_MTD_significance[j]);
            rel_ch_iso_EB_list_Significance_Sig[j]->Fill(rel_pT_sum_MTD_significance[j]);

            if (optionalPlots_) {
              Ntracks_sim_EB_list_Significance_Sig[j]->Fill(N_tracks_sim_MTD_significance[j]);
              ch_iso_sim_EB_list_Significance_Sig[j]->Fill(pT_sum_sim_MTD_significance[j]);
              rel_ch_iso_sim_EB_list_Significance_Sig[j]->Fill(rel_pT_sum_sim_MTD_significance[j]);
            }
          }

          if (rel_pT_sum_noMTD < rel_iso_cut_) {  // filling hists for iso efficiency calculations
            meEle_pt_noMTD_Sig_EB_->Fill(ele.pt());
            meEle_eta_noMTD_Sig_EB_->Fill(std::abs(ele.eta()));
            meEle_phi_noMTD_Sig_EB_->Fill(ele.phi());
          }
          if (optionalPlots_) {
            for (long unsigned int k = 0; k < Ntracks_EB_list_Sig.size(); k++) {
              if (rel_pT_sum_MTD[k] < rel_iso_cut_) {
                Ele_pT_MTD_EB_list_Sig[k]->Fill(ele.pt());
                Ele_eta_MTD_EB_list_Sig[k]->Fill(std::abs(ele.eta()));
                Ele_phi_MTD_EB_list_Sig[k]->Fill(ele.phi());

                Ele_pT_sim_MTD_EB_list_Sig[k]->Fill(ele_sim_pt);
              }
            }
            if (rel_pT_sum_gen < rel_iso_cut_) {
              meEle_pt_gen_Sig_EB_->Fill(ele_sim_pt);
              meEle_eta_gen_Sig_EB_->Fill(ele_sim_eta);
              meEle_phi_gen_Sig_EB_->Fill(ele_sim_phi);
            }
          }

          for (long unsigned int k = 0; k < Ntracks_EB_list_Significance_Sig.size(); k++) {
            if (rel_pT_sum_MTD_significance[k] < rel_iso_cut_) {
              Ele_pT_MTD_EB_list_Significance_Sig[k]->Fill(ele.pt());
              Ele_eta_MTD_EB_list_Significance_Sig[k]->Fill(std::abs(ele.eta()));
              Ele_phi_MTD_EB_list_Significance_Sig[k]->Fill(ele.phi());
            }
            if (optionalPlots_ and rel_pT_sum_sim_MTD_significance[k] < rel_iso_cut_)
              Ele_pT_sim_MTD_EB_list_Significance_Sig[k]->Fill(ele_sim_pt);
          }

        } else {  // for endcap

          meEleISO_Ntracks_Sig_EE_->Fill(N_tracks_noMTD);
          meEleISO_chIso_Sig_EE_->Fill(pT_sum_noMTD);
          meEleISO_rel_chIso_Sig_EE_->Fill(rel_pT_sum_noMTD);
          if (optionalPlots_) {
            for (long unsigned int j = 0; j < Ntracks_EE_list_Sig.size(); j++) {
              Ntracks_EE_list_Sig[j]->Fill(N_tracks_MTD[j]);
              ch_iso_EE_list_Sig[j]->Fill(pT_sum_MTD[j]);
              rel_ch_iso_EE_list_Sig[j]->Fill(rel_pT_sum_MTD[j]);

              Ntracks_sim_EE_list_Sig[j]->Fill(N_tracks_sim_MTD[j]);
              ch_iso_sim_EE_list_Sig[j]->Fill(pT_sum_sim_MTD[j]);
              rel_ch_iso_sim_EE_list_Sig[j]->Fill(rel_pT_sum_sim_MTD[j]);
            }
            meEleISO_Ntracks_gen_Sig_EE_->Fill(N_tracks_gen);
            meEleISO_chIso_gen_Sig_EE_->Fill(pT_sum_gen);
            meEleISO_rel_chIso_gen_Sig_EE_->Fill(rel_pT_sum_gen);
          }

          for (long unsigned int j = 0; j < Ntracks_EE_list_Significance_Sig.size(); j++) {
            Ntracks_EE_list_Significance_Sig[j]->Fill(N_tracks_MTD_significance[j]);
            ch_iso_EE_list_Significance_Sig[j]->Fill(pT_sum_MTD_significance[j]);
            rel_ch_iso_EE_list_Significance_Sig[j]->Fill(rel_pT_sum_MTD_significance[j]);

            if (optionalPlots_) {
              Ntracks_sim_EE_list_Significance_Sig[j]->Fill(N_tracks_sim_MTD_significance[j]);
              ch_iso_sim_EE_list_Significance_Sig[j]->Fill(pT_sum_sim_MTD_significance[j]);
              rel_ch_iso_sim_EE_list_Significance_Sig[j]->Fill(rel_pT_sum_sim_MTD_significance[j]);
            }
          }

          if (rel_pT_sum_noMTD < rel_iso_cut_) {  // filling hists for iso efficiency calculations
            meEle_pt_noMTD_Sig_EE_->Fill(ele.pt());
            meEle_eta_noMTD_Sig_EE_->Fill(std::abs(ele.eta()));
            meEle_phi_noMTD_Sig_EE_->Fill(ele.phi());
          }
          if (optionalPlots_) {
            for (long unsigned int k = 0; k < Ntracks_EE_list_Sig.size(); k++) {
              if (rel_pT_sum_MTD[k] < rel_iso_cut_) {
                Ele_pT_MTD_EE_list_Sig[k]->Fill(ele.pt());
                Ele_eta_MTD_EE_list_Sig[k]->Fill(std::abs(ele.eta()));
                Ele_phi_MTD_EE_list_Sig[k]->Fill(ele.phi());

                Ele_pT_sim_MTD_EE_list_Sig[k]->Fill(ele_sim_pt);
              }
            }
            if (rel_pT_sum_gen < rel_iso_cut_) {
              meEle_pt_gen_Sig_EE_->Fill(ele_sim_pt);
              meEle_eta_gen_Sig_EE_->Fill(ele_sim_eta);
              meEle_phi_gen_Sig_EE_->Fill(ele_sim_phi);
            }
          }
          for (long unsigned int k = 0; k < Ntracks_EE_list_Significance_Sig.size(); k++) {
            if (rel_pT_sum_MTD_significance[k] < rel_iso_cut_) {
              Ele_pT_MTD_EE_list_Significance_Sig[k]->Fill(ele.pt());
              Ele_eta_MTD_EE_list_Significance_Sig[k]->Fill(std::abs(ele.eta()));
              Ele_phi_MTD_EE_list_Significance_Sig[k]->Fill(ele.phi());

              if (optionalPlots_ and rel_pT_sum_sim_MTD_significance[k] < rel_iso_cut_)
                Ele_pT_sim_MTD_EE_list_Significance_Sig[k]->Fill(ele_sim_pt);
            }
          }
        }
      } else {  // non-promt part
        if (Barrel_ele) {
          meEleISO_Ntracks_Bkg_EB_->Fill(N_tracks_noMTD);
          meEleISO_chIso_Bkg_EB_->Fill(pT_sum_noMTD);
          meEleISO_rel_chIso_Bkg_EB_->Fill(rel_pT_sum_noMTD);
          if (optionalPlots_) {
            for (long unsigned int j = 0; j < Ntracks_EB_list_Bkg.size(); j++) {
              Ntracks_EB_list_Bkg[j]->Fill(N_tracks_MTD[j]);
              ch_iso_EB_list_Bkg[j]->Fill(pT_sum_MTD[j]);
              rel_ch_iso_EB_list_Bkg[j]->Fill(rel_pT_sum_MTD[j]);

              Ntracks_sim_EB_list_Bkg[j]->Fill(N_tracks_sim_MTD[j]);
              ch_iso_sim_EB_list_Bkg[j]->Fill(pT_sum_sim_MTD[j]);
              rel_ch_iso_sim_EB_list_Bkg[j]->Fill(rel_pT_sum_sim_MTD[j]);
            }
            meEleISO_Ntracks_gen_Bkg_EB_->Fill(N_tracks_gen);
            meEleISO_chIso_gen_Bkg_EB_->Fill(pT_sum_gen);
            meEleISO_rel_chIso_gen_Bkg_EB_->Fill(rel_pT_sum_gen);
          }

          for (long unsigned int j = 0; j < Ntracks_EB_list_Significance_Bkg.size(); j++) {
            Ntracks_EB_list_Significance_Bkg[j]->Fill(N_tracks_MTD_significance[j]);
            ch_iso_EB_list_Significance_Bkg[j]->Fill(pT_sum_MTD_significance[j]);
            rel_ch_iso_EB_list_Significance_Bkg[j]->Fill(rel_pT_sum_MTD_significance[j]);

            if (optionalPlots_) {
              Ntracks_sim_EB_list_Significance_Bkg[j]->Fill(N_tracks_sim_MTD_significance[j]);
              ch_iso_sim_EB_list_Significance_Bkg[j]->Fill(pT_sum_sim_MTD_significance[j]);
              rel_ch_iso_sim_EB_list_Significance_Bkg[j]->Fill(rel_pT_sum_sim_MTD_significance[j]);
            }
          }

          if (rel_pT_sum_noMTD < rel_iso_cut_) {  // filling hists for iso efficiency calculations
            meEle_pt_noMTD_Bkg_EB_->Fill(ele.pt());
            meEle_eta_noMTD_Bkg_EB_->Fill(std::abs(ele.eta()));
            meEle_phi_noMTD_Bkg_EB_->Fill(ele.phi());
          }
          if (optionalPlots_) {
            for (long unsigned int k = 0; k < Ntracks_EB_list_Bkg.size(); k++) {
              if (rel_pT_sum_MTD[k] < rel_iso_cut_) {
                Ele_pT_MTD_EB_list_Bkg[k]->Fill(ele.pt());
                Ele_eta_MTD_EB_list_Bkg[k]->Fill(std::abs(ele.eta()));
                Ele_phi_MTD_EB_list_Bkg[k]->Fill(ele.phi());

                Ele_pT_sim_MTD_EB_list_Bkg[k]->Fill(ele_sim_pt);
              }
            }
            if (rel_pT_sum_gen < rel_iso_cut_) {
              meEle_pt_gen_Bkg_EB_->Fill(ele_sim_pt);
              meEle_eta_gen_Bkg_EB_->Fill(ele_sim_eta);
              meEle_phi_gen_Bkg_EB_->Fill(ele_sim_phi);
            }
          }
          for (long unsigned int k = 0; k < Ntracks_EB_list_Significance_Bkg.size(); k++) {
            if (rel_pT_sum_MTD_significance[k] < rel_iso_cut_) {
              Ele_pT_MTD_EB_list_Significance_Bkg[k]->Fill(ele.pt());
              Ele_eta_MTD_EB_list_Significance_Bkg[k]->Fill(std::abs(ele.eta()));
              Ele_phi_MTD_EB_list_Significance_Bkg[k]->Fill(ele.phi());

              if (optionalPlots_ and rel_pT_sum_sim_MTD_significance[k] < rel_iso_cut_)
                Ele_pT_sim_MTD_EB_list_Significance_Bkg[k]->Fill(ele_sim_pt);
            }
          }

        } else {  // for endcap
          meEleISO_Ntracks_Bkg_EE_->Fill(N_tracks_noMTD);
          meEleISO_chIso_Bkg_EE_->Fill(pT_sum_noMTD);
          meEleISO_rel_chIso_Bkg_EE_->Fill(rel_pT_sum_noMTD);
          if (optionalPlots_) {
            for (long unsigned int j = 0; j < Ntracks_EE_list_Bkg.size(); j++) {
              Ntracks_EE_list_Bkg[j]->Fill(N_tracks_MTD[j]);
              ch_iso_EE_list_Bkg[j]->Fill(pT_sum_MTD[j]);
              rel_ch_iso_EE_list_Bkg[j]->Fill(rel_pT_sum_MTD[j]);

              Ntracks_sim_EE_list_Bkg[j]->Fill(N_tracks_sim_MTD[j]);
              ch_iso_sim_EE_list_Bkg[j]->Fill(pT_sum_sim_MTD[j]);
              rel_ch_iso_sim_EE_list_Bkg[j]->Fill(rel_pT_sum_sim_MTD[j]);
            }
            meEleISO_Ntracks_gen_Bkg_EE_->Fill(N_tracks_gen);
            meEleISO_chIso_gen_Bkg_EE_->Fill(pT_sum_gen);
            meEleISO_rel_chIso_gen_Bkg_EE_->Fill(rel_pT_sum_gen);
          }

          for (long unsigned int j = 0; j < Ntracks_EE_list_Significance_Bkg.size(); j++) {
            Ntracks_EE_list_Significance_Bkg[j]->Fill(N_tracks_MTD_significance[j]);
            ch_iso_EE_list_Significance_Bkg[j]->Fill(pT_sum_MTD_significance[j]);
            rel_ch_iso_EE_list_Significance_Bkg[j]->Fill(rel_pT_sum_MTD_significance[j]);

            if (optionalPlots_) {
              Ntracks_sim_EE_list_Significance_Bkg[j]->Fill(N_tracks_sim_MTD_significance[j]);
              ch_iso_sim_EE_list_Significance_Bkg[j]->Fill(pT_sum_sim_MTD_significance[j]);
              rel_ch_iso_sim_EE_list_Significance_Bkg[j]->Fill(rel_pT_sum_sim_MTD_significance[j]);
            }
          }

          if (rel_pT_sum_noMTD < rel_iso_cut_) {  // filling hists for iso efficiency calculations
            meEle_pt_noMTD_Bkg_EE_->Fill(ele.pt());
            meEle_eta_noMTD_Bkg_EE_->Fill(std::abs(ele.eta()));
            meEle_phi_noMTD_Bkg_EE_->Fill(ele.phi());
          }
          if (optionalPlots_) {
            for (long unsigned int k = 0; k < Ntracks_EE_list_Bkg.size(); k++) {
              if (rel_pT_sum_MTD[k] < rel_iso_cut_) {
                Ele_pT_MTD_EE_list_Bkg[k]->Fill(ele.pt());
                Ele_eta_MTD_EE_list_Bkg[k]->Fill(std::abs(ele.eta()));
                Ele_phi_MTD_EE_list_Bkg[k]->Fill(ele.phi());

                Ele_pT_sim_MTD_EE_list_Bkg[k]->Fill(ele_sim_pt);
              }
            }
            if (rel_pT_sum_gen < rel_iso_cut_) {
              meEle_pt_gen_Bkg_EE_->Fill(ele_sim_pt);
              meEle_eta_gen_Bkg_EE_->Fill(ele_sim_eta);
              meEle_phi_gen_Bkg_EE_->Fill(ele_sim_phi);
            }
          }

          for (long unsigned int k = 0; k < Ntracks_EE_list_Significance_Bkg.size(); k++) {
            if (rel_pT_sum_MTD_significance[k] < rel_iso_cut_) {
              Ele_pT_MTD_EE_list_Significance_Bkg[k]->Fill(ele.pt());
              Ele_eta_MTD_EE_list_Significance_Bkg[k]->Fill(std::abs(ele.eta()));
              Ele_phi_MTD_EE_list_Significance_Bkg[k]->Fill(ele.phi());

              if (optionalPlots_ and rel_pT_sum_sim_MTD_significance[k] < rel_iso_cut_)
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

  // for regular Validation use a reduced binning, for detailed analysis and ROC curves use the larger one
  int nbin_1 = 40;
  int nbin_2 = 40;
  if (optionalPlots_) {
    nbin_1 = 1000;
    nbin_2 = 2000;
  }

  // histogram booking

  meEle_avg_error_SigTrk_check_ =
      ibook.book1D("SigTrk_avg_timing_err",
                   "Average signal electron track MTD timing uncertainty;Time Error (ns);Counts",
                   200,
                   0,
                   0.1);
  meEle_avg_error_PUTrk_check_ = ibook.book1D(
      "PUTrk_avg_timing_err", "Average PU track MTD timing uncertainty;Time Error (ns);Counts", 200, 0, 0.1);
  meEle_avg_error_vtx_check_ =
      ibook.book1D("Vtx_avg_timing_err", "Average vertex timing uncertainty;Time Error (ns);Counts", 200, 0, 0.1);

  meEle_no_dt_check_ =
      ibook.book1D("Track_dt_info_check",
                   "Tracks dt check - ratio between tracks with (value 1) and without (value 0) timing info",
                   2,
                   0,
                   2);

  meTrk_genMatch_check_ = ibook.book1D(
      "Track_genMatch_info_check", "Check on tracks matched with a GenParticle (matched=1, non matched=0)", 2, 0, 2);

  // signal
  meEleISO_Ntracks_Sig_EB_ = ibook.book1D("Ele_Iso_Ntracks_Sig_EB",
                                          "Number of tracks in isolation cone around electron track after basic cuts - "
                                          "Signal Barrel;Number of tracks;Counts",
                                          20,
                                          0,
                                          20);

  meEleISO_chIso_Sig_EB_ = ibook.book1D(
      "Ele_chIso_sum_Sig_EB",
      "Track pT sum in isolation cone around electron track after basic cuts - Signal Barrel;p_{T} (GeV);Counts",
      nbin_2,
      0,
      20);

  meEleISO_rel_chIso_Sig_EB_ = ibook.book1D(
      "Ele_rel_chIso_sum_Sig_EB",
      "Track relative pT sum in isolation cone around electron track after basic cuts - Signal Barrel;Isolation;Counts",
      nbin_1,
      0,
      4);
  if (optionalPlots_) {
    meEleISO_Ntracks_MTD_1_Sig_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_1_Sig_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);

    meEleISO_chIso_MTD_1_Sig_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_1_Sig_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_1_Sig_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_1_Sig_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);
    // gen
    meEleISO_Ntracks_gen_Sig_EB_ = ibook.book1D("Ele_Iso_Ntracks_gen_Sig_EB",
                                                "Number of tracks in isolation cone around electron track after basic "
                                                "cuts using genInfo - Signal Barrel;Number of tracks;Counts",
                                                20,
                                                0,
                                                20);

    meEleISO_chIso_gen_Sig_EB_ = ibook.book1D("Ele_chIso_sum_gen_Sig_EB",
                                              "Track pT sum in isolation cone around electron track after basic cuts "
                                              "using genInfo - Signal Barrel;p_{T} (GeV);Counts",
                                              nbin_2,
                                              0,
                                              20);

    meEleISO_rel_chIso_gen_Sig_EB_ = ibook.book1D("Ele_rel_chIso_sum_gen_Sig_EB",
                                                  "Track relative pT sum in isolation cone around electron track after "
                                                  "basic cuts using genInfo - Signal Barrel;Isolation;Counts",
                                                  nbin_1,
                                                  0,
                                                  4);

    meEleISO_Ntracks_MTD_2_Sig_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_2_Sig_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);

    meEleISO_chIso_MTD_2_Sig_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_2_Sig_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_2_Sig_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_2_Sig_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_3_Sig_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_3_Sig_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_3_Sig_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_3_Sig_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_3_Sig_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_3_Sig_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_4_Sig_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_4_Sig_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_4_Sig_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_4_Sig_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_4_Sig_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_4_Sig_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_5_Sig_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_5_Sig_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_5_Sig_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_5_Sig_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_5_Sig_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_5_Sig_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_6_Sig_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_6_Sig_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_6_Sig_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_6_Sig_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_6_Sig_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_6_Sig_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_7_Sig_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_7_Sig_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_7_Sig_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_7_Sig_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_7_Sig_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_7_Sig_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_1_Sig_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_1_Sig_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);

    meEleISO_chIso_MTD_sim_1_Sig_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_1_Sig_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_1_Sig_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_1_Sig_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_2_Sig_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_2_Sig_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);

    meEleISO_chIso_MTD_sim_2_Sig_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_2_Sig_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_2_Sig_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_2_Sig_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_3_Sig_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_3_Sig_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_3_Sig_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_3_Sig_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_3_Sig_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_3_Sig_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_4_Sig_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_4_Sig_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_4_Sig_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_4_Sig_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_4_Sig_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_4_Sig_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_5_Sig_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_5_Sig_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_5_Sig_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_5_Sig_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_5_Sig_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_5_Sig_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_6_Sig_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_6_Sig_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_6_Sig_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_6_Sig_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_6_Sig_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_6_Sig_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_7_Sig_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_7_Sig_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_7_Sig_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_7_Sig_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_7_Sig_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_7_Sig_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);
  }
  meEleISO_Ntracks_MTD_4sigma_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_4sigma_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD - 4 sigma compatibiliy - "
                   "Signal Barrel;Number of tracks;Counts",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_4sigma_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_4sigma_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic "
                   "cuts with MTD - 4 sigma compatibiliy - Signal Barrel;p_{T} (GeV);Counts",
                   nbin_2,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_4sigma_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_4sigma_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD - 4 sigma "
                   "compatibiliy - Signal Barrel;Isolation;Counts",
                   nbin_1,
                   0,
                   4);

  meEleISO_Ntracks_MTD_3sigma_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_3sigma_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD - 3 sigma compatibiliy - "
                   "Signal Barrel;Number of tracks;Counts",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_3sigma_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_3sigma_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic "
                   "cuts with MTD - 3 sigma compatibiliy - Signal Barrel;p_{T} (GeV);Counts",
                   nbin_2,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_3sigma_Sig_EB_ = ibook.book1D("Ele_rel_chIso_sum_MTD_3sigma_Sig_EB",
                                                       "Track relative pT sum in isolation cone around electron track "
                                                       "after basic cuts with MTD - 3 sigma;Isolation;Counts"
                                                       "compatibiliy - Signal Barrel",
                                                       nbin_1,
                                                       0,
                                                       4);

  meEleISO_Ntracks_MTD_2sigma_Sig_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_2sigma_Sig_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD - 2 sigma compatibiliy - "
                   "Signal Barrel;Number of tracks;Counts",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_2sigma_Sig_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_2sigma_Sig_EB",
                   "Track pT sum in isolation cone around electron track after basic "
                   "cuts with MTD - 2 sigma compatibiliy - Signal Barrel;p_{T} (GeV);Counts",
                   nbin_2,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_2sigma_Sig_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_2sigma_Sig_EB",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD - 2 sigma "
                   "compatibiliy - Signal Barrel;Isolation;Counts",
                   nbin_1,
                   0,
                   4);

  meEle_pt_tot_Sig_EB_ =
      ibook.book1D("Ele_pT_tot_Sig_EB", "Electron pT tot - Signal Barrel;p_{T} (GeV);Counts", 30, 10, 100);
  meEle_pt_noMTD_Sig_EB_ =
      ibook.book1D("Ele_pT_noMTD_Sig_EB", "Electron pT noMTD - Signal Barrel;p_{T} (GeV);Counts", 30, 10, 100);

  meEle_pt_sim_tot_Sig_EB_ =
      ibook.book1D("Ele_pT_sim_tot_Sig_EB", "Electron SIM pT tot - Signal Barrel;p_{T} (GeV);Counts", 30, 10, 100);

  meEle_eta_tot_Sig_EB_ =
      ibook.book1D("Ele_eta_tot_Sig_EB", "Electron eta tot - Signal Barrel;#eta;Counts", 32, 0., 1.6);
  meEle_eta_noMTD_Sig_EB_ =
      ibook.book1D("Ele_eta_noMTD_Sig_EB", "Electron eta noMTD - Signal Barrel;#eta;Counts", 32, 0., 1.6);

  meEle_phi_tot_Sig_EB_ =
      ibook.book1D("Ele_phi_tot_Sig_EB", "Electron phi tot - Signal Barrel;#phi;Counts", 64, -3.2, 3.2);
  meEle_phi_noMTD_Sig_EB_ =
      ibook.book1D("Ele_phi_noMTD_Sig_EB", "Electron phi noMTD - Signal Barrel;#phi;Counts", 64, -3.2, 3.2);

  if (optionalPlots_) {
    meEleISO_Ntracks_MTD_sim_4sigma_Sig_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_4sigma_Sig_EB",
                     "Number of tracks in isolation cone around electron track after basic cuts with MTD - 4 sigma "
                     "compatibiliy - Signal Barrel;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_4sigma_Sig_EB_ =
        ibook.book1D("Ele_chIso_sum_MTD_sim_4sigma_Sig_EB",
                     "Track pT sum in isolation cone around electron track after "
                     "basic cuts with MTD - 4 sigma compatibiliy - Signal Barrel;p_{T} (GeV);Counts",
                     nbin_2,
                     0,
                     20);
    meEleISO_rel_chIso_MTD_sim_4sigma_Sig_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_4sigma_Sig_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD - 4 sigma "
        "compatibiliy - Signal Barrel;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_3sigma_Sig_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_3sigma_Sig_EB",
                     "Tracks in isolation cone around electron track after basic "
                     "cuts with MTD  - 3 sigma compatibiliy - Signal Barrel;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_3sigma_Sig_EB_ =
        ibook.book1D("Ele_chIso_sum_MTD_sim_3sigma_Sig_EB",
                     "Track pT sum in isolation cone around electron track after "
                     "basic cuts with MTD - 3 sigma compatibiliy - Signal Barrel;p_{T} (GeV);Counts",
                     nbin_2,
                     0,
                     20);
    meEleISO_rel_chIso_MTD_sim_3sigma_Sig_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_3sigma_Sig_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD - 3 sigma "
        "compatibiliy - Signal Barrel;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_2sigma_Sig_EB_ = ibook.book1D(
        "Ele_Iso_Ntracks_MTD_sim_2sigma_Sig_EB",
        "Tracks in isolation cone around electron track after basic cuts with MTD - 2 sigma compatibiliy - "
        "Signal Barrel;Number of tracks;Counts",
        20,
        0,
        20);
    meEleISO_chIso_MTD_sim_2sigma_Sig_EB_ =
        ibook.book1D("Ele_chIso_sum_MTD_sim_2sigma_Sig_EB",
                     "Track pT sum in isolation cone around electron track after "
                     "basic cuts with MTD - 2 sigma compatibiliy - Signal Barrel;p_{T} (GeV);Counts",
                     nbin_2,
                     0,
                     20);
    meEleISO_rel_chIso_MTD_sim_2sigma_Sig_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_2sigma_Sig_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD - 2 sigma "
        "compatibiliy - Signal Barrel;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEle_pt_gen_Sig_EB_ =
        ibook.book1D("Ele_pT_gen_Sig_EB", "Electron pT genInfo - Signal Barrel;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_gen_Sig_EB_ =
        ibook.book1D("Ele_eta_gen_Sig_EB", "Electron eta genInfo - Signal Barrel;#eta;Counts", 32, 0., 1.6);
    meEle_phi_gen_Sig_EB_ =
        ibook.book1D("Ele_phi_gen_Sig_EB", "Electron phi genInfo - Signal Barrel;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_1_Sig_EB_ = ibook.book1D("Ele_pT_MTD_1_Sig_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_1_Sig_EB_ = ibook.book1D("Ele_eta_MTD_1_Sig_EB", "Electron eta MTD;#eta;Counts", 32, 0., 1.6);
    meEle_phi_MTD_1_Sig_EB_ = ibook.book1D("Ele_phi_MTD_1_Sig_EB", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_2_Sig_EB_ = ibook.book1D("Ele_pT_MTD_2_Sig_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_2_Sig_EB_ = ibook.book1D("Ele_eta_MTD_2_Sig_EB", "Electron eta MTD;#eta;Counts", 32, 0., 1.6);
    meEle_phi_MTD_2_Sig_EB_ = ibook.book1D("Ele_phi_MTD_2_Sig_EB", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_3_Sig_EB_ = ibook.book1D("Ele_pT_MTD_3_Sig_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_3_Sig_EB_ = ibook.book1D("Ele_eta_MTD_3_Sig_EB", "Electron eta MTD;#eta;Counts", 32, 0., 1.6);
    meEle_phi_MTD_3_Sig_EB_ = ibook.book1D("Ele_phi_MTD_3_Sig_EB", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_4_Sig_EB_ = ibook.book1D("Ele_pT_MTD_4_Sig_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_4_Sig_EB_ = ibook.book1D("Ele_eta_MTD_4_Sig_EB", "Electron eta MTD;#eta;Counts", 32, 0., 1.6);
    meEle_phi_MTD_4_Sig_EB_ = ibook.book1D("Ele_phi_MTD_4_Sig_EB", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_5_Sig_EB_ = ibook.book1D("Ele_pT_MTD_5_Sig_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_5_Sig_EB_ = ibook.book1D("Ele_eta_MTD_5_Sig_EB", "Electron eta MTD;#eta;Counts", 32, 0., 1.6);
    meEle_phi_MTD_5_Sig_EB_ = ibook.book1D("Ele_phi_MTD_5_Sig_EB", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_6_Sig_EB_ = ibook.book1D("Ele_pT_MTD_6_Sig_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_6_Sig_EB_ = ibook.book1D("Ele_eta_MTD_6_Sig_EB", "Electron eta MTD;#eta;Counts", 32, 0., 1.6);
    meEle_phi_MTD_6_Sig_EB_ = ibook.book1D("Ele_phi_MTD_6_Sig_EB", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_7_Sig_EB_ = ibook.book1D("Ele_pT_MTD_7_Sig_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_7_Sig_EB_ = ibook.book1D("Ele_eta_MTD_7_Sig_EB", "Electron eta MTD;#eta;Counts", 32, 0., 1.6);
    meEle_phi_MTD_7_Sig_EB_ = ibook.book1D("Ele_phi_MTD_7_Sig_EB", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_sim_MTD_1_Sig_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_1_Sig_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_2_Sig_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_2_Sig_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_3_Sig_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_3_Sig_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_4_Sig_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_4_Sig_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_5_Sig_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_5_Sig_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_6_Sig_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_6_Sig_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_7_Sig_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_7_Sig_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
  }

  meEle_pt_MTD_4sigma_Sig_EB_ =
      ibook.book1D("Ele_pT_MTD_4sigma_Sig_EB",
                   "Electron pT MTD - 4 sigma compatibility - Signal Barrel;p_{T} (GeV);Counts",
                   30,
                   10,
                   100);
  meEle_eta_MTD_4sigma_Sig_EB_ = ibook.book1D(
      "Ele_eta_MTD_4sigma_Sig_EB", "Electron eta MTD - 4 sigma compatibility - Signal Barrel;#eta;Counts", 32, 0., 1.6);
  meEle_phi_MTD_4sigma_Sig_EB_ = ibook.book1D("Ele_phi_MTD_4sigma_Sig_EB",
                                              "Electron phi MTD - 4 sigma compatibility - Signal Barrel;#phi;Counts",
                                              64,
                                              -3.2,
                                              3.2);

  meEle_pt_MTD_3sigma_Sig_EB_ =
      ibook.book1D("Ele_pT_MTD_3sigma_Sig_EB",
                   "Electron pT MTD - 3 sigma compatibility - Signal Barrel;p_{T} (GeV);Counts",
                   30,
                   10,
                   100);
  meEle_eta_MTD_3sigma_Sig_EB_ = ibook.book1D(
      "Ele_eta_MTD_3sigma_Sig_EB", "Electron eta MTD - 3 sigma compatibility - Signal Barrel;#eta;Counts", 32, 0., 1.6);
  meEle_phi_MTD_3sigma_Sig_EB_ = ibook.book1D("Ele_phi_MTD_3sigma_Sig_EB",
                                              "Electron phi MTD - 3 sigma compatibility - Signal Barrel;#phi;Counts",
                                              64,
                                              -3.2,
                                              3.2);

  meEle_pt_MTD_2sigma_Sig_EB_ =
      ibook.book1D("Ele_pT_MTD_2sigma_Sig_EB",
                   "Electron pT MTD - 2 sigma compatibility - Signal Barrel;p_{T} (GeV);Counts",
                   30,
                   10,
                   100);
  meEle_eta_MTD_2sigma_Sig_EB_ = ibook.book1D(
      "Ele_eta_MTD_2sigma_Sig_EB", "Electron eta MTD - 2 sigma compatibility - Signal Barrel;#eta;Counts", 32, 0., 1.6);
  meEle_phi_MTD_2sigma_Sig_EB_ = ibook.book1D("Ele_phi_MTD_2sigma_Sig_EB",
                                              "Electron phi MTD - 2 sigma compatibility - Signal Barrel;#phi;Counts",
                                              64,
                                              -3.2,
                                              3.2);

  meEleISO_Ntracks_Sig_EE_ = ibook.book1D("Ele_Iso_Ntracks_Sig_EE",
                                          "Number of tracks in isolation cone around electron track after basic cuts - "
                                          "Signal Endcap;Number of tracks;Counts",
                                          20,
                                          0,
                                          20);
  meEleISO_chIso_Sig_EE_ = ibook.book1D(
      "Ele_chIso_sum_Sig_EE",
      "Track pT sum in isolation cone around electron track after basic cuts - Signal Endcap;p_{T} (GeV);Counts",
      nbin_2,
      0,
      20);
  meEleISO_rel_chIso_Sig_EE_ = ibook.book1D(
      "Ele_rel_chIso_sum_Sig_EE",
      "Track relative pT sum in isolation cone around electron track after basic cuts - Signal Endcap;Isolation;Counts",
      nbin_1,
      0,
      4);

  if (optionalPlots_) {
    meEle_pt_sim_MTD_4sigma_Sig_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_4sigma_Sig_EB",
                     "Electron pT MTD SIM - 4 sigma compatibility - Signal Barrel;p_{T} (GeV);Counts",
                     30,
                     10,
                     100);
    meEle_pt_sim_MTD_3sigma_Sig_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_3sigma_Sig_EB",
                     "Electron pT MTD SIM - 3 sigma compatibility - Signal Barrel;p_{T} (GeV);Counts",
                     30,
                     10,
                     100);
    meEle_pt_sim_MTD_2sigma_Sig_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_2sigma_Sig_EB",
                     "Electron pT MTD SIM - 2 sigma compatibility - Signal Barrel;p_{T} (GeV);Counts",
                     30,
                     10,
                     100);

    meEleISO_Ntracks_MTD_1_Sig_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_1_Sig_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_1_Sig_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_1_Sig_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_1_Sig_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_1_Sig_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_2_Sig_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_2_Sig_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_2_Sig_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_2_Sig_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_2_Sig_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_2_Sig_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_gen_Sig_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_gen_Sig_EE",
                     "Tracks in isolation cone around electron track after basic cuts using genInfo - Signal Endcap",
                     20,
                     0,
                     20);
    meEleISO_chIso_gen_Sig_EE_ =
        ibook.book1D("Ele_chIso_sum_gen_Sig_EE",
                     "Track pT sum in isolation cone around electron track after basic cuts - Signal Endcap",
                     nbin_2,
                     0,
                     20);
    meEleISO_rel_chIso_gen_Sig_EE_ =
        ibook.book1D("Ele_rel_chIso_sum_gen_Sig_EE",
                     "Track relative pT sum in isolation cone around electron track after basic cuts - Signal Endcap",
                     nbin_1,
                     0,
                     4);

    meEleISO_Ntracks_MTD_3_Sig_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_3_Sig_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_3_Sig_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_3_Sig_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_3_Sig_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_3_Sig_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_4_Sig_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_4_Sig_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_4_Sig_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_4_Sig_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_4_Sig_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_4_Sig_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_5_Sig_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_5_Sig_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_5_Sig_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_5_Sig_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_5_Sig_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_5_Sig_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_6_Sig_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_6_Sig_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_6_Sig_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_6_Sig_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_6_Sig_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_6_Sig_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_7_Sig_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_7_Sig_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_7_Sig_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_7_Sig_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_7_Sig_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_7_Sig_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_1_Sig_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_1_Sig_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_1_Sig_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_1_Sig_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_1_Sig_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_1_Sig_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_2_Sig_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_2_Sig_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_2_Sig_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_2_Sig_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_2_Sig_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_2_Sig_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_3_Sig_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_3_Sig_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_3_Sig_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_3_Sig_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_3_Sig_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_3_Sig_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_4_Sig_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_4_Sig_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_4_Sig_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_4_Sig_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_4_Sig_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_4_Sig_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_5_Sig_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_5_Sig_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_5_Sig_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_5_Sig_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_5_Sig_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_5_Sig_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_6_Sig_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_6_Sig_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_6_Sig_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_6_Sig_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_6_Sig_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_6_Sig_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_7_Sig_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_7_Sig_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_7_Sig_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_7_Sig_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_7_Sig_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_7_Sig_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);
  }
  meEleISO_Ntracks_MTD_4sigma_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_4sigma_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD - 4 sigma significance - "
                   "Signal Endcap;Number of tracks;Counts",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_4sigma_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_4sigma_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic "
                   "cuts with MTD - 4 sigma significance - Signal Endcap;p_{T} (GeV);Counts",
                   nbin_2,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_4sigma_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_4sigma_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD - 4 sigma "
                   "significance - Signal Endcap;Isolation;Counts",
                   nbin_1,
                   0,
                   4);

  meEleISO_Ntracks_MTD_3sigma_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_3sigma_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD - 3 sigma significance - "
                   "Signal Endcap;Number of tracks;Counts",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_3sigma_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_3sigma_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic "
                   "cuts with MTD - 3 sigma significance - Signal Endcap;p_{T} (GeV);Counts",
                   nbin_2,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_3sigma_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_3sigma_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD - 3 sigma "
                   "significance - Signal Endcap;Isolation;Counts",
                   nbin_1,
                   0,
                   4);

  meEleISO_Ntracks_MTD_2sigma_Sig_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_2sigma_Sig_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD - 2 sigma significance - "
                   "Signal Endcap;Number of tracks;Counts",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_2sigma_Sig_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_2sigma_Sig_EE",
                   "Track pT sum in isolation cone around electron track after basic "
                   "cuts with MTD - 2 sigma significance - Signal Endcap;p_{T} (GeV);Counts",
                   nbin_2,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_2sigma_Sig_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_2sigma_Sig_EE",
                   "Track relative pT sum in isolation cone around electron track after basic cuts with MTD - 2 sigma "
                   "significance - Signal Endcap;Isolation;Counts",
                   nbin_1,
                   0,
                   4);

  meEle_pt_tot_Sig_EE_ =
      ibook.book1D("Ele_pT_tot_Sig_EE", "Electron pT tot - Signal Endcap;p_{T} (GeV);Counts", 30, 10, 100);
  meEle_pt_noMTD_Sig_EE_ =
      ibook.book1D("Ele_pT_noMTD_Sig_EE", "Electron pT noMTD - Signal Endcap;p_{T} (GeV);Counts", 30, 10, 100);

  meEle_pt_sim_tot_Sig_EE_ =
      ibook.book1D("Ele_pT_sim_tot_Sig_EE", "Electron pT tot - Signal Endcap;p_{T} (GeV);Counts", 30, 10, 100);

  meEle_eta_tot_Sig_EE_ =
      ibook.book1D("Ele_eta_tot_Sig_EE", "Electron eta tot - Signal Endcap;#eta;Counts", 32, 1.6, 3.2);
  meEle_eta_noMTD_Sig_EE_ =
      ibook.book1D("Ele_eta_noMTD_Sig_EE", "Electron eta noMTD - Signal Endcap;#eta;Counts", 32, 1.6, 3.2);

  meEle_phi_tot_Sig_EE_ =
      ibook.book1D("Ele_phi_tot_Sig_EE", "Electron phi tot - Signal Endcap;#phi;Counts", 64, -3.2, 3.2);
  meEle_phi_noMTD_Sig_EE_ =
      ibook.book1D("Ele_phi_noMTD_Sig_EE", "Electron phi noMTD - Signal Endcap;#phi;Counts", 64, -3.2, 3.2);

  if (optionalPlots_) {
    meEleISO_Ntracks_MTD_sim_4sigma_Sig_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_4sigma_Sig_EE",
                     "Number of tracks in isolation cone around electron track after basic cuts with MTD SIM - 4 sigma "
                     "significance - Signal Endcap;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_4sigma_Sig_EE_ =
        ibook.book1D("Ele_chIso_sum_MTD_sim_4sigma_Sig_EE",
                     "Track pT sum in isolation cone around electron track after "
                     "basic cuts with MTD SIM - 4 sigma significance - Signal Endcap;p_{T} (GeV);Counts",
                     nbin_2,
                     0,
                     20);
    meEleISO_rel_chIso_MTD_sim_4sigma_Sig_EE_ =
        ibook.book1D("Ele_rel_chIso_sum_MTD_sim_4sigma_Sig_EE",
                     "Track relative pT sum in isolation cone around electron track after basic cuts with MTD SIM - 4 "
                     "sigma significance - Signal Endcap;Isolation;Counts",
                     nbin_1,
                     0,
                     4);

    meEleISO_Ntracks_MTD_sim_3sigma_Sig_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_3sigma_Sig_EE",
                     "Tracks in isolation cone around electron track after basic "
                     "cuts with MTD SIM - 3 sigma significance - Signal Endcap;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_3sigma_Sig_EE_ =
        ibook.book1D("Ele_chIso_sum_MTD_sim_3sigma_Sig_EE",
                     "Track pT sum in isolation cone around electron track after "
                     "basic cuts with MTD SIM - 3 sigma significance - Signal Endcap;p_{T} (GeV);Counts",
                     nbin_2,
                     0,
                     20);
    meEleISO_rel_chIso_MTD_sim_3sigma_Sig_EE_ =
        ibook.book1D("Ele_rel_chIso_sum_MTD_sim_3sigma_Sig_EE",
                     "Track relative pT sum in isolation cone around electron track after basic cuts with MTD SIM - 3 "
                     "sigma significance - Signal Endcap;Isolation;Counts",
                     nbin_1,
                     0,
                     4);

    meEleISO_Ntracks_MTD_sim_2sigma_Sig_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_2sigma_Sig_EE",
                     "Tracks in isolation cone around electron track after basic "
                     "cuts with MTD SIM - 2 sigma significance - Signal Endcap;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_2sigma_Sig_EE_ =
        ibook.book1D("Ele_chIso_sum_MTD_sim_2sigma_Sig_EE",
                     "Track pT sum in isolation cone around electron track after "
                     "basic cuts with MTD SIM - 2 sigma significance - Signal Endcap;p_{T} (GeV);Counts",
                     nbin_2,
                     0,
                     20);
    meEleISO_rel_chIso_MTD_sim_2sigma_Sig_EE_ =
        ibook.book1D("Ele_rel_chIso_sum_MTD_sim_2sigma_Sig_EE",
                     "Track relative pT sum in isolation cone around electron track after basic cuts with MTD SIM - 2 "
                     "sigma significance - Signal Endcap;Isolation;Counts",
                     nbin_1,
                     0,
                     4);

    meEle_pt_MTD_1_Sig_EE_ = ibook.book1D("Ele_pT_MTD_1_Sig_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_1_Sig_EE_ = ibook.book1D("Ele_eta_MTD_1_Sig_EE", "Electron eta MTD;#eta;Counts", 32, 1.6, 3.2);
    meEle_phi_MTD_1_Sig_EE_ = ibook.book1D("Ele_phi_MTD_1_Sig_EE", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);
    meEle_pt_gen_Sig_EE_ =
        ibook.book1D("Ele_pT_gen_Sig_EE", "Electron pT genInfo - Signal Endcap;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_gen_Sig_EE_ =
        ibook.book1D("Ele_eta_gen_Sig_EE", "Electron eta genInfo - Signal Endcap;#eta;Counts", 32, 1.6, 3.2);
    meEle_phi_gen_Sig_EE_ =
        ibook.book1D("Ele_phi_gen_Sig_EE", "Electron phi genInfo - Signal Endcap;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_2_Sig_EE_ = ibook.book1D("Ele_pT_MTD_2_Sig_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_2_Sig_EE_ = ibook.book1D("Ele_eta_MTD_2_Sig_EE", "Electron eta MTD;#eta;Counts", 32, 1.6, 3.2);
    meEle_phi_MTD_2_Sig_EE_ = ibook.book1D("Ele_phi_MTD_2_Sig_EE", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_3_Sig_EE_ = ibook.book1D("Ele_pT_MTD_3_Sig_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_3_Sig_EE_ = ibook.book1D("Ele_eta_MTD_3_Sig_EE", "Electron eta MTD;#eta;Counts", 32, 1.6, 3.2);
    meEle_phi_MTD_3_Sig_EE_ = ibook.book1D("Ele_phi_MTD_3_Sig_EE", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_4_Sig_EE_ = ibook.book1D("Ele_pT_MTD_4_Sig_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_4_Sig_EE_ = ibook.book1D("Ele_eta_MTD_4_Sig_EE", "Electron eta MTD;#eta;Counts", 32, 1.6, 3.2);
    meEle_phi_MTD_4_Sig_EE_ = ibook.book1D("Ele_phi_MTD_4_Sig_EE", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_5_Sig_EE_ = ibook.book1D("Ele_pT_MTD_5_Sig_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_5_Sig_EE_ = ibook.book1D("Ele_eta_MTD_5_Sig_EE", "Electron eta MTD;#eta;Counts", 32, 1.6, 3.2);
    meEle_phi_MTD_5_Sig_EE_ = ibook.book1D("Ele_phi_MTD_5_Sig_EE", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_6_Sig_EE_ = ibook.book1D("Ele_pT_MTD_6_Sig_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_6_Sig_EE_ = ibook.book1D("Ele_eta_MTD_6_Sig_EE", "Electron eta MTD;#eta;Counts", 32, 1.6, 3.2);
    meEle_phi_MTD_6_Sig_EE_ = ibook.book1D("Ele_phi_MTD_6_Sig_EE", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_7_Sig_EE_ = ibook.book1D("Ele_pT_MTD_7_Sig_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_7_Sig_EE_ = ibook.book1D("Ele_eta_MTD_7_Sig_EE", "Electron eta MTD;#eta;Counts", 32, 1.6, 3.2);
    meEle_phi_MTD_7_Sig_EE_ = ibook.book1D("Ele_phi_MTD_7_Sig_EE", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_sim_MTD_1_Sig_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_1_Sig_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_2_Sig_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_2_Sig_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_3_Sig_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_3_Sig_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_4_Sig_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_4_Sig_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_5_Sig_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_5_Sig_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_6_Sig_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_6_Sig_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_7_Sig_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_7_Sig_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);

    meEle_pt_sim_MTD_4sigma_Sig_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_4sigma_Sig_EE",
                     "Electron pT MTD SIM - 4 sigma significance - Signal Endcap;p_{T} (GeV);Counts",
                     30,
                     10,
                     100);
    meEle_pt_sim_MTD_3sigma_Sig_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_3sigma_Sig_EE",
                     "Electron pT MTD SIM - 3 sigma significance - Signal Endcap;p_{T} (GeV);Counts",
                     30,
                     10,
                     100);
    meEle_pt_sim_MTD_2sigma_Sig_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_2sigma_Sig_EE",
                     "Electron pT MTD SIM - 2 sigma significance - Signal Endcap;p_{T} (GeV);Counts",
                     30,
                     10,
                     100);
  }

  meEle_pt_MTD_4sigma_Sig_EE_ =
      ibook.book1D("Ele_pT_MTD_4sigma_Sig_EE",
                   "Electron pT MTD - 4 sigma significance - Signal Endcap;p_{T} (GeV);Counts",
                   30,
                   10,
                   100);
  meEle_eta_MTD_4sigma_Sig_EE_ = ibook.book1D(
      "Ele_eta_MTD_4sigma_Sig_EE", "Electron eta MTD - 4 sigma significance - Signal Endcap;#eta;Counts", 32, 1.6, 3.2);
  meEle_phi_MTD_4sigma_Sig_EE_ = ibook.book1D(
      "Ele_phi_MTD_4sigma_Sig_EE", "Electron phi MTD - 4 sigma significance - Signal Endcap;#phi;Counts", 64, -3.2, 3.2);

  meEle_pt_MTD_3sigma_Sig_EE_ =
      ibook.book1D("Ele_pT_MTD_3sigma_Sig_EE",
                   "Electron pT MTD - 3 sigma significance - Signal Endcap;p_{T} (GeV);Counts",
                   30,
                   10,
                   100);
  meEle_eta_MTD_3sigma_Sig_EE_ = ibook.book1D(
      "Ele_eta_MTD_3sigma_Sig_EE", "Electron eta MTD - 3 sigma significance - Signal Endcap;#eta;Counts", 32, 1.6, 3.2);
  meEle_phi_MTD_3sigma_Sig_EE_ = ibook.book1D(
      "Ele_phi_MTD_3sigma_Sig_EE", "Electron phi MTD - 3 sigma significance - Signal Endcap;#phi;Counts", 64, -3.2, 3.2);

  meEle_pt_MTD_2sigma_Sig_EE_ =
      ibook.book1D("Ele_pT_MTD_2sigma_Sig_EE",
                   "Electron pT MTD - 2 sigma significance - Signal Endcap;p_{T} (GeV);Counts",
                   30,
                   10,
                   100);
  meEle_eta_MTD_2sigma_Sig_EE_ = ibook.book1D(
      "Ele_eta_MTD_2sigma_Sig_EE", "Electron eta MTD - 2 sigma significance - Signal Endcap;#eta;Counts", 32, 1.6, 3.2);
  meEle_phi_MTD_2sigma_Sig_EE_ = ibook.book1D(
      "Ele_phi_MTD_2sigma_Sig_EE", "Electron phi MTD - 2 sigma significance - Signal Endcap;#phi;Counts", 64, -3.2, 3.2);

  // background
  meEleISO_Ntracks_Bkg_EB_ = ibook.book1D(
      "Ele_Iso_Ntracks_Bkg_EB",
      "Number of tracks in isolation cone around electron track after basic cuts - Bkg Barrel;Number of tracks;Counts",
      20,
      0,
      20);
  meEleISO_chIso_Bkg_EB_ = ibook.book1D(
      "Ele_chIso_sum_Bkg_EB",
      "Track pT sum in isolation cone around electron track after basic cuts - Bkg Barrel;p_{T} (GeV);Counts",
      nbin_2,
      0,
      20);
  meEleISO_rel_chIso_Bkg_EB_ = ibook.book1D(
      "Ele_rel_chIso_sum_Bkg_EB",
      "Track relative pT sum in isolation cone around electron track after basic cuts - Bkg Barrel;Isolation;Counts",
      nbin_1,
      0,
      4);
  if (optionalPlots_) {
    meEleISO_Ntracks_MTD_1_Bkg_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_1_Bkg_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_1_Bkg_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_1_Bkg_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_1_Bkg_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_1_Bkg_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_2_Bkg_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_2_Bkg_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_2_Bkg_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_2_Bkg_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_2_Bkg_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_2_Bkg_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);
    meEleISO_Ntracks_gen_Bkg_EB_ = ibook.book1D("Ele_Iso_Ntracks_gen_Bkg_EB",
                                                "Tracks in isolation cone around electron track after basic cuts using "
                                                "genInfo - Bkg Barrel;Number of tracks;Counts",
                                                20,
                                                0,
                                                20);
    meEleISO_chIso_gen_Bkg_EB_ = ibook.book1D("Ele_chIso_sum_gen_Bkg_EB",
                                              "Track pT sum in isolation cone around electron track after basic cuts "
                                              "using genInfo - Bkg Barrel;p_{T} (GeV);Counts",
                                              nbin_2,
                                              0,
                                              20);
    meEleISO_rel_chIso_gen_Bkg_EB_ = ibook.book1D("Ele_rel_chIso_sum_gen_Bkg_EB",
                                                  "Track relative pT sum in isolation cone around electron track after "
                                                  "basic cuts using genInfo - Bkg Barrel;Isolation;Counts",
                                                  nbin_1,
                                                  0,
                                                  4);
    meEleISO_Ntracks_MTD_3_Bkg_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_3_Bkg_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_3_Bkg_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_3_Bkg_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_3_Bkg_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_3_Bkg_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_4_Bkg_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_4_Bkg_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_4_Bkg_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_4_Bkg_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_4_Bkg_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_4_Bkg_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_5_Bkg_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_5_Bkg_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_5_Bkg_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_5_Bkg_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_5_Bkg_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_5_Bkg_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_6_Bkg_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_6_Bkg_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_6_Bkg_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_6_Bkg_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_6_Bkg_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_6_Bkg_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_7_Bkg_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_7_Bkg_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_7_Bkg_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_7_Bkg_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_7_Bkg_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_7_Bkg_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_1_Bkg_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_1_Bkg_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_1_Bkg_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_1_Bkg_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_1_Bkg_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_1_Bkg_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_2_Bkg_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_2_Bkg_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_2_Bkg_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_2_Bkg_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_2_Bkg_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_2_Bkg_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_3_Bkg_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_3_Bkg_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_3_Bkg_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_3_Bkg_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_3_Bkg_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_3_Bkg_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_4_Bkg_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_4_Bkg_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_4_Bkg_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_4_Bkg_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_4_Bkg_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_4_Bkg_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_5_Bkg_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_5_Bkg_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_5_Bkg_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_5_Bkg_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_5_Bkg_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_5_Bkg_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_6_Bkg_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_6_Bkg_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_6_Bkg_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_6_Bkg_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_6_Bkg_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_6_Bkg_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_7_Bkg_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_7_Bkg_EB",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_7_Bkg_EB_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_7_Bkg_EB",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_7_Bkg_EB_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_7_Bkg_EB",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);
  }
  meEleISO_Ntracks_MTD_4sigma_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_4sigma_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD - 4 sigma significance - "
                   "Bkg Barrel;Number of tracks;Counts",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_4sigma_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_4sigma_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic "
                   "cuts with MTD - 4 sigma significance - Bkg Barrel;p_{T} (GeV);Counts",
                   nbin_2,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_4sigma_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_4sigma_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track "
                   "after basic cuts with MTD - 4 sigma significance - Bkg Barrel;Isolation;Counts",
                   nbin_1,
                   0,
                   4);

  meEleISO_Ntracks_MTD_3sigma_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_3sigma_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD - 3 sigma significance - "
                   "Bkg Barrel;Number of tracks;Counts",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_3sigma_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_3sigma_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic "
                   "cuts with MTD - 3 sigma significance - Bkg Barrel;p_{T} (GeV);Counts",
                   nbin_2,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_3sigma_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_3sigma_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track "
                   "after basic cuts with MTD - 3 sigma significance - Bkg Barrel;Isolation;Counts",
                   nbin_1,
                   0,
                   4);

  meEleISO_Ntracks_MTD_2sigma_Bkg_EB_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_2sigma_Bkg_EB",
                   "Tracks in isolation cone around electron track after basic cuts with MTD - 2 sigma significance - "
                   "Bkg Barrel;Number of tracks;Counts",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_2sigma_Bkg_EB_ =
      ibook.book1D("Ele_chIso_sum_MTD_2sigma_Bkg_EB",
                   "Track pT sum in isolation cone around electron track after basic "
                   "cuts with MTD - 2 sigma significance - Bkg Barrel;p_{T} (GeV);Counts",
                   nbin_2,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_2sigma_Bkg_EB_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_2sigma_Bkg_EB",
                   "Track relative pT sum in isolation cone around electron track "
                   "after basic cuts with MTD - 2 sigma significance - Bkg Barrel;Isolation;Counts",
                   nbin_1,
                   0,
                   4);

  meEle_pt_tot_Bkg_EB_ =
      ibook.book1D("Ele_pT_tot_Bkg_EB", "Electron pT tot - Bkg Barrel;p_{T} (GeV);Counts", 30, 10, 100);
  meEle_pt_noMTD_Bkg_EB_ =
      ibook.book1D("Ele_pT_noMTD_Bkg_EB", "Electron pT noMTD - Bkg Barrel;p_{T} (GeV);Counts", 30, 10, 100);

  meEle_pt_sim_tot_Bkg_EB_ =
      ibook.book1D("Ele_pT_sim_tot_Bkg_EB", "Electron pT tot - Bkg Barrel;p_{T} (GeV);Counts", 30, 10, 100);

  meEle_eta_tot_Bkg_EB_ = ibook.book1D("Ele_eta_tot_Bkg_EB", "Electron eta tot - Bkg Barrel;#eta;Counts", 32, 0., 1.6);
  meEle_eta_noMTD_Bkg_EB_ =
      ibook.book1D("Ele_eta_noMTD_Bkg_EB", "Electron eta noMTD - Bkg Barrel;#eta;Counts", 32, 0., 1.6);

  meEle_phi_tot_Bkg_EB_ =
      ibook.book1D("Ele_phi_tot_Bkg_EB", "Electron phi tot - Bkg Barrel;#phi;#Counts", 64, -3.2, 3.2);
  meEle_phi_noMTD_Bkg_EB_ =
      ibook.book1D("Ele_phi_noMTD_Bkg_EB", "Electron phi noMTD - Bkg Barrel;#phi;#Counts", 64, -3.2, 3.2);

  if (optionalPlots_) {
    meEleISO_Ntracks_MTD_sim_4sigma_Bkg_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_4sigma_Bkg_EB",
                     "Tracks in isolation cone around electron track after basic "
                     "cuts with MTD SIM - 4 sigma significance - Bkg Barrel;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_4sigma_Bkg_EB_ =
        ibook.book1D("Ele_chIso_sum_MTD_sim_4sigma_Bkg_EB",
                     "Track pT sum in isolation cone around electron track after "
                     "basic cuts with MTD SIM - 4 sigma significance - Bkg Barrel;p_{T} (GeV);Counts",
                     nbin_2,
                     0,
                     20);
    meEleISO_rel_chIso_MTD_sim_4sigma_Bkg_EB_ =
        ibook.book1D("Ele_rel_chIso_sum_MTD_sim_4sigma_Bkg_EB",
                     "Track relative pT sum in isolation cone around electron track after basic cuts with MTD SIM - 4 "
                     "sigma significance - Bkg Barrel;Isolation;Counts",
                     nbin_1,
                     0,
                     4);

    meEleISO_Ntracks_MTD_sim_3sigma_Bkg_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_3sigma_Bkg_EB",
                     "Tracks in isolation cone around electron track after basic "
                     "cuts with MTD SIM - 3 sigma significance - Bkg Barrel;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_3sigma_Bkg_EB_ =
        ibook.book1D("Ele_chIso_sum_MTD_sim_3sigma_Bkg_EB",
                     "Track pT sum in isolation cone around electron track after "
                     "basic cuts with MTD SIM - 3 sigma significance - Bkg Barrel;p_{T} (GeV);Counts",
                     nbin_2,
                     0,
                     20);
    meEleISO_rel_chIso_MTD_sim_3sigma_Bkg_EB_ =
        ibook.book1D("Ele_rel_chIso_sum_MTD_sim_3sigma_Bkg_EB",
                     "Track relative pT sum in isolation cone around electron track after basic cuts with MTD SIM - 3 "
                     "sigma significance - Bkg Barrel;Isolation;Counts",
                     nbin_1,
                     0,
                     4);

    meEleISO_Ntracks_MTD_sim_2sigma_Bkg_EB_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_2sigma_Bkg_EB",
                     "Tracks in isolation cone around electron track after basic "
                     "cuts with MTD SIM - 3 sigma significance - Bkg Barrel;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_2sigma_Bkg_EB_ =
        ibook.book1D("Ele_chIso_sum_MTD_sim_2sigma_Bkg_EB",
                     "Track pT sum in isolation cone around electron track after "
                     "basic cuts with MTD SIM - 2 sigma significance - Bkg Barrel;p_{T} (GeV);Counts",
                     nbin_2,
                     0,
                     20);
    meEleISO_rel_chIso_MTD_sim_2sigma_Bkg_EB_ =
        ibook.book1D("Ele_rel_chIso_sum_MTD_sim_2sigma_Bkg_EB",
                     "Track relative pT sum in isolation cone around electron track after basic cuts with MTD SIM - 2 "
                     "sigma significance - Bkg Barrel;Isolation;Counts",
                     nbin_1,
                     0,
                     4);

    meEle_pt_MTD_1_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_1_Bkg_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_1_Bkg_EB_ = ibook.book1D("Ele_eta_MTD_1_Bkg_EB", "Electron eta MTD;#eta;Counts", 32, 0., 1.6);
    meEle_phi_MTD_1_Bkg_EB_ = ibook.book1D("Ele_phi_MTD_1_Bkg_EB", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);
    meEle_pt_gen_Bkg_EB_ =
        ibook.book1D("Ele_pT_gen_Bkg_EB", "Electron pT genInfo - Bkg Barrel;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_gen_Bkg_EB_ =
        ibook.book1D("Ele_eta_gen_Bkg_EB", "Electron eta genInfo - Bkg Barrel;#eta;Counts", 32, 0., 1.6);
    meEle_phi_gen_Bkg_EB_ =
        ibook.book1D("Ele_phi_gen_Bkg_EB", "Electron phi genInfo - Bkg Barrel;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_2_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_2_Bkg_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_2_Bkg_EB_ = ibook.book1D("Ele_eta_MTD_2_Bkg_EB", "Electron eta MTD;#eta;Counts", 32, 0., 1.6);
    meEle_phi_MTD_2_Bkg_EB_ = ibook.book1D("Ele_phi_MTD_2_Bkg_EB", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_3_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_3_Bkg_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_3_Bkg_EB_ = ibook.book1D("Ele_eta_MTD_3_Bkg_EB", "Electron eta MTD;#eta;Counts", 32, 0., 1.6);
    meEle_phi_MTD_3_Bkg_EB_ = ibook.book1D("Ele_phi_MTD_3_Bkg_EB", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_4_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_4_Bkg_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_4_Bkg_EB_ = ibook.book1D("Ele_eta_MTD_4_Bkg_EB", "Electron eta MTD;#eta;Counts", 32, 0., 1.6);
    meEle_phi_MTD_4_Bkg_EB_ = ibook.book1D("Ele_phi_MTD_4_Bkg_EB", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_5_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_5_Bkg_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_5_Bkg_EB_ = ibook.book1D("Ele_eta_MTD_5_Bkg_EB", "Electron eta MTD;#eta;Counts", 32, 0., 1.6);
    meEle_phi_MTD_5_Bkg_EB_ = ibook.book1D("Ele_phi_MTD_5_Bkg_EB", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_6_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_6_Bkg_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_6_Bkg_EB_ = ibook.book1D("Ele_eta_MTD_6_Bkg_EB", "Electron eta MTD;#eta;Counts", 32, 0., 1.6);
    meEle_phi_MTD_6_Bkg_EB_ = ibook.book1D("Ele_phi_MTD_6_Bkg_EB", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_7_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_7_Bkg_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_7_Bkg_EB_ = ibook.book1D("Ele_eta_MTD_7_Bkg_EB", "Electron eta MTD;#eta;Counts", 32, 0., 1.6);
    meEle_phi_MTD_7_Bkg_EB_ = ibook.book1D("Ele_phi_MTD_7_Bkg_EB", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_sim_MTD_1_Bkg_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_1_Bkg_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_2_Bkg_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_2_Bkg_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_3_Bkg_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_3_Bkg_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_4_Bkg_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_4_Bkg_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_5_Bkg_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_5_Bkg_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_6_Bkg_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_6_Bkg_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_7_Bkg_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_7_Bkg_EB", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
  }
  meEle_pt_MTD_4sigma_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_4sigma_Bkg_EB",
                                             "Electron pT MTD - 4 sigma compatibility - Bkg Barrel;p_{T} (GeV);Counts",
                                             30,
                                             10,
                                             100);
  meEle_eta_MTD_4sigma_Bkg_EB_ = ibook.book1D(
      "Ele_eta_MTD_4sigma_Bkg_EB", "Electron eta MTD - 4 sigma compatibility - Bkg Barrel;#eta;Counts", 32, 0., 1.6);
  meEle_phi_MTD_4sigma_Bkg_EB_ = ibook.book1D(
      "Ele_phi_MTD_4sigma_Bkg_EB", "Electron phi MTD - 4 sigma compatibility - Bkg Barrel;#phi;Counts", 64, -3.2, 3.2);

  meEle_pt_MTD_3sigma_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_3sigma_Bkg_EB",
                                             "Electron pT MTD - 3 sigma compatibility - Bkg Barrel;p_{T} (GeV);Counts",
                                             30,
                                             10,
                                             100);
  meEle_eta_MTD_3sigma_Bkg_EB_ = ibook.book1D(
      "Ele_eta_MTD_3sigma_Bkg_EB", "Electron eta MTD - 3 sigma compatibility - Bkg Barrel;#eta;Counts", 32, 0., 1.6);
  meEle_phi_MTD_3sigma_Bkg_EB_ = ibook.book1D(
      "Ele_phi_MTD_3sigma_Bkg_EB", "Electron phi MTD - 3 sigma compatibility - Bkg Barrel;#phi;Counts", 64, -3.2, 3.2);

  meEle_pt_MTD_2sigma_Bkg_EB_ = ibook.book1D("Ele_pT_MTD_2sigma_Bkg_EB",
                                             "Electron pT MTD - 2 sigma compatibility - Bkg Barrel;p_{T} (GeV);Counts",
                                             30,
                                             10,
                                             100);
  meEle_eta_MTD_2sigma_Bkg_EB_ = ibook.book1D(
      "Ele_eta_MTD_2sigma_Bkg_EB", "Electron eta MTD - 2 sigma compatibility - Bkg Barrel;#eta;Counts", 32, 0., 1.6);
  meEle_phi_MTD_2sigma_Bkg_EB_ = ibook.book1D(
      "Ele_phi_MTD_2sigma_Bkg_EB", "Electron phi MTD - 2 sigma compatibility - Bkg Barrel;#phi;Counts", 64, -3.2, 3.2);

  meEleISO_Ntracks_Bkg_EE_ = ibook.book1D(
      "Ele_Iso_Ntracks_Bkg_EE",
      "Number of tracks in isolation cone around electron track after basic cuts - Bkg Endcap;Number of tracks;Counts",
      20,
      0,
      20);
  meEleISO_chIso_Bkg_EE_ = ibook.book1D(
      "Ele_chIso_sum_Bkg_EE",
      "Track pT sum in isolation cone around electron track after basic cuts - Bkg Endcap;p_{T} (GeV);Counts",
      nbin_2,
      0,
      20);
  meEleISO_rel_chIso_Bkg_EE_ = ibook.book1D(
      "Ele_rel_chIso_sum_Bkg_EE",
      "Track relative pT sum in isolation cone around electron track after basic cuts - Bkg Endcap;Isolation;Counts",
      nbin_1,
      0,
      4);
  if (optionalPlots_) {
    meEle_pt_sim_MTD_4sigma_Bkg_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_4sigma_Bkg_EB",
                     "Electron pT MTD SIM - 4 sigma compatibility - Bkg Barrel;p_{T} (GeV);Counts",
                     30,
                     10,
                     100);
    meEle_pt_sim_MTD_3sigma_Bkg_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_3sigma_Bkg_EB",
                     "Electron pT MTD SIM - 3 sigma compatibility - Bkg Barrel;p_{T} (GeV);Counts",
                     30,
                     10,
                     100);
    meEle_pt_sim_MTD_2sigma_Bkg_EB_ =
        ibook.book1D("Ele_pT_sim_MTD_2sigma_Bkg_EB",
                     "Electron pT MTD SIM - 2 sigma compatibility - Bkg Barrel;p_{T} (GeV);Counts",
                     30,
                     10,
                     100);

    meEleISO_Ntracks_MTD_1_Bkg_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_1_Bkg_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_1_Bkg_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_1_Bkg_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_1_Bkg_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_1_Bkg_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_2_Bkg_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_2_Bkg_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_2_Bkg_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_2_Bkg_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_2_Bkg_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_2_Bkg_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);
    meEleISO_Ntracks_gen_Bkg_EE_ = ibook.book1D("Ele_Iso_Ntracks_gen_Bkg_EE",
                                                "Tracks in isolation cone around electron track after basic cuts using "
                                                "genInfo - Bkg Endcap;Number of tracks;Counts",
                                                20,
                                                0,
                                                20);
    meEleISO_chIso_gen_Bkg_EE_ = ibook.book1D("Ele_chIso_sum_gen_Bkg_EE",
                                              "Track pT sum in isolation cone around electron track after basic cuts "
                                              "using genInfo - Bkg Endcap;p_{T} (GeV);Counts",
                                              nbin_2,
                                              0,
                                              20);
    meEleISO_rel_chIso_gen_Bkg_EE_ = ibook.book1D("Ele_rel_chIso_sum_gen_Bkg_EE",
                                                  "Track relative pT sum in isolation cone around electron track after "
                                                  "basic cuts using genInfo - Bkg Endcap;Isolation;Counts",
                                                  nbin_1,
                                                  0,
                                                  4);

    meEleISO_Ntracks_MTD_3_Bkg_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_3_Bkg_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_3_Bkg_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_3_Bkg_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_3_Bkg_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_3_Bkg_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_4_Bkg_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_4_Bkg_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_4_Bkg_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_4_Bkg_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_4_Bkg_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_4_Bkg_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_5_Bkg_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_5_Bkg_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_5_Bkg_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_5_Bkg_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_5_Bkg_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_5_Bkg_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_6_Bkg_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_6_Bkg_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_6_Bkg_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_6_Bkg_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_6_Bkg_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_6_Bkg_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_7_Bkg_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_7_Bkg_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_7_Bkg_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_7_Bkg_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_7_Bkg_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_7_Bkg_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_1_Bkg_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_1_Bkg_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_1_Bkg_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_1_Bkg_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_1_Bkg_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_1_Bkg_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_2_Bkg_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_2_Bkg_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_2_Bkg_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_2_Bkg_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_2_Bkg_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_2_Bkg_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_3_Bkg_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_3_Bkg_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_3_Bkg_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_3_Bkg_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_3_Bkg_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_3_Bkg_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_4_Bkg_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_4_Bkg_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_4_Bkg_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_4_Bkg_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_4_Bkg_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_4_Bkg_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_5_Bkg_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_5_Bkg_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_5_Bkg_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_5_Bkg_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_5_Bkg_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_5_Bkg_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_6_Bkg_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_6_Bkg_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_6_Bkg_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_6_Bkg_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_6_Bkg_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_6_Bkg_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);

    meEleISO_Ntracks_MTD_sim_7_Bkg_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_7_Bkg_EE",
                     "Tracks in isolation cone around electron track after basic cuts with MTD;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_7_Bkg_EE_ = ibook.book1D(
        "Ele_chIso_sum_MTD_sim_7_Bkg_EE",
        "Track pT sum in isolation cone around electron track after basic cuts with MTD;p_{T} (GeV);Counts",
        nbin_2,
        0,
        20);
    meEleISO_rel_chIso_MTD_sim_7_Bkg_EE_ = ibook.book1D(
        "Ele_rel_chIso_sum_MTD_sim_7_Bkg_EE",
        "Track relative pT sum in isolation cone around electron track after basic cuts with MTD;Isolation;Counts",
        nbin_1,
        0,
        4);
  }
  meEleISO_Ntracks_MTD_4sigma_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_4sigma_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD - 4 sigma compatibility - "
                   "Bkg Endcap;Number of tracks;Counts",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_4sigma_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_4sigma_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic "
                   "cuts with MTD - 4 sigma compatibility - Bkg Endcap;p_{T} (GeV);Counts",
                   nbin_2,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_4sigma_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_4sigma_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track "
                   "after basic cuts with MTD - 4 sigma compatibility - Bkg Endcap;Isolation;Counts",
                   nbin_1,
                   0,
                   4);

  meEleISO_Ntracks_MTD_3sigma_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_3sigma_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD - 3 sigma compatibility - "
                   "Bkg Endcap;Number of tracks;Counts",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_3sigma_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_3sigma_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic "
                   "cuts with MTD - 3 sigma compatibility - Bkg Endcap;p_{T} (GeV);Counts",
                   nbin_2,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_3sigma_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_3sigma_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track "
                   "after basic cuts with MTD - 3 sigma compatibility - Bkg Endcap;Isolation;Counts",
                   nbin_1,
                   0,
                   4);

  meEleISO_Ntracks_MTD_2sigma_Bkg_EE_ =
      ibook.book1D("Ele_Iso_Ntracks_MTD_2sigma_Bkg_EE",
                   "Tracks in isolation cone around electron track after basic cuts with MTD - 2 sigma compatibility - "
                   "Bkg Endcap;Number of tracks;Counts",
                   20,
                   0,
                   20);
  meEleISO_chIso_MTD_2sigma_Bkg_EE_ =
      ibook.book1D("Ele_chIso_sum_MTD_2sigma_Bkg_EE",
                   "Track pT sum in isolation cone around electron track after basic "
                   "cuts with MTD - 2 sigma compatibility - Bkg Endcap;p_{T} (GeV);Counts",
                   nbin_2,
                   0,
                   20);
  meEleISO_rel_chIso_MTD_2sigma_Bkg_EE_ =
      ibook.book1D("Ele_rel_chIso_sum_MTD_2sigma_Bkg_EE",
                   "Track relative pT sum in isolation cone around electron track "
                   "after basic cuts with MTD - 2 sigma compatibility - Bkg Endcap;Isolation;Counts",
                   nbin_1,
                   0,
                   4);

  meEle_pt_tot_Bkg_EE_ =
      ibook.book1D("Ele_pT_tot_Bkg_EE", "Electron pT tot - Bkg Endcap;p_{T} (GeV);Counts", 30, 10, 100);
  meEle_pt_noMTD_Bkg_EE_ =
      ibook.book1D("Ele_pT_noMTD_Bkg_EE", "Electron pT noMTD - Bkg Endcap;p_{T} (GeV);Counts", 30, 10, 100);

  meEle_pt_sim_tot_Bkg_EE_ =
      ibook.book1D("Ele_pT_sim_tot_Bkg_EE", "Electron pT tot - Bkg Endcap;p_{T} (GeV);Counts", 30, 10, 100);

  meEle_eta_tot_Bkg_EE_ = ibook.book1D("Ele_eta_tot_Bkg_EE", "Electron eta tot - Bkg Endcap;#eta;Counts", 32, 1.6, 3.2);
  meEle_eta_noMTD_Bkg_EE_ =
      ibook.book1D("Ele_eta_noMTD_Bkg_EE", "Electron eta noMTD - Bkg Endcap;#eta;Counts", 32, 1.6, 3.2);

  meEle_phi_tot_Bkg_EE_ =
      ibook.book1D("Ele_phi_tot_Bkg_EE", "Electron phi tot - Bkg Endcap;#phi;Counts", 64, -3.2, 3.2);
  meEle_phi_noMTD_Bkg_EE_ =
      ibook.book1D("Ele_phi_noMTD_Bkg_EE", "Electron phi noMTD - Bkg Endcap;#phi;Counts", 64, -3.2, 3.2);
  if (optionalPlots_) {
    meEleISO_Ntracks_MTD_sim_4sigma_Bkg_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_4sigma_Bkg_EE",
                     "Tracks in isolation cone around electron track after basic "
                     "cuts with MTD SIM - 4 sigma compatibility - Bkg Endcap;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_4sigma_Bkg_EE_ =
        ibook.book1D("Ele_chIso_sum_MTD_sim_4sigma_Bkg_EE",
                     "Track pT sum in isolation cone around electron track after "
                     "basic cuts with MTD SIM - 4 sigma compatibility - Bkg Endcap;p_{T} (GeV);Counts",
                     nbin_2,
                     0,
                     20);
    meEleISO_rel_chIso_MTD_sim_4sigma_Bkg_EE_ =
        ibook.book1D("Ele_rel_chIso_sum_MTD_sim_4sigma_Bkg_EE",
                     "Track relative pT sum in isolation cone around electron track after basic cuts with MTD SIM - 4 "
                     "sigma compatibility - Bkg Endcap;Isolation;Counts",
                     nbin_1,
                     0,
                     4);

    meEleISO_Ntracks_MTD_sim_3sigma_Bkg_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_3sigma_Bkg_EE",
                     "Tracks in isolation cone around electron track after basic "
                     "cuts with MTD SIM - 3 sigma compatibility - Bkg Endcap;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_3sigma_Bkg_EE_ =
        ibook.book1D("Ele_chIso_sum_MTD_sim_3sigma_Bkg_EE",
                     "Track pT sum in isolation cone around electron track after "
                     "basic cuts with MTD SIM - 3 sigma compatibility - Bkg Endcap;p_{T} (GeV);Counts",
                     nbin_2,
                     0,
                     20);
    meEleISO_rel_chIso_MTD_sim_3sigma_Bkg_EE_ =
        ibook.book1D("Ele_rel_chIso_sum_MTD_sim_3sigma_Bkg_EE",
                     "Track relative pT sum in isolation cone around electron track after basic cuts with MTD SIM - 3 "
                     "sigma compatibility - Bkg Endcap;Isolation;Counts",
                     nbin_1,
                     0,
                     4);

    meEleISO_Ntracks_MTD_sim_2sigma_Bkg_EE_ =
        ibook.book1D("Ele_Iso_Ntracks_MTD_sim_2sigma_Bkg_EE",
                     "Tracks in isolation cone around electron track after basic "
                     "cuts with MTD SIM - 2 sigma compatibility - Bkg Endcap;Number of tracks;Counts",
                     20,
                     0,
                     20);
    meEleISO_chIso_MTD_sim_2sigma_Bkg_EE_ =
        ibook.book1D("Ele_chIso_sum_MTD_sim_2sigma_Bkg_EE",
                     "Track pT sum in isolation cone around electron track after "
                     "basic cuts with MTD SIM - 2 sigma compatibility - Bkg Endcap;p_{T} (GeV);Counts",
                     nbin_2,
                     0,
                     20);
    meEleISO_rel_chIso_MTD_sim_2sigma_Bkg_EE_ =
        ibook.book1D("Ele_rel_chIso_sum_MTD_sim_2sigma_Bkg_EE",
                     "Track relative pT sum in isolation cone around electron track after basic cuts with MTD SIM - 2 "
                     "sigma compatibility - Bkg Endcap;Isolation;Counts",
                     nbin_1,
                     0,
                     4);

    meEle_pt_MTD_1_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_1_Bkg_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_1_Bkg_EE_ = ibook.book1D("Ele_eta_MTD_1_Bkg_EE", "Electron eta MTD;#eta;Counts", 32, 1.6, 3.2);
    meEle_phi_MTD_1_Bkg_EE_ = ibook.book1D("Ele_phi_MTD_1_Bkg_EE", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);
    meEle_pt_gen_Bkg_EE_ =
        ibook.book1D("Ele_pT_gen_Bkg_EE", "Electron pT genInfo - Bkg Endcap;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_gen_Bkg_EE_ =
        ibook.book1D("Ele_eta_gen_Bkg_EE", "Electron eta genInfo - Bkg Endcap;#eta;Counts", 32, 1.6, 3.2);
    meEle_phi_gen_Bkg_EE_ =
        ibook.book1D("Ele_phi_gen_Bkg_EE", "Electron phi genInfo - Bkg Endcap;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_2_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_2_Bkg_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_2_Bkg_EE_ = ibook.book1D("Ele_eta_MTD_2_Bkg_EE", "Electron eta MTD;#eta;Counts", 32, 1.6, 3.2);
    meEle_phi_MTD_2_Bkg_EE_ = ibook.book1D("Ele_phi_MTD_2_Bkg_EE", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_3_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_3_Bkg_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_3_Bkg_EE_ = ibook.book1D("Ele_eta_MTD_3_Bkg_EE", "Electron eta MTD;#eta;Counts", 32, 1.6, 3.2);
    meEle_phi_MTD_3_Bkg_EE_ = ibook.book1D("Ele_phi_MTD_3_Bkg_EE", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_4_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_4_Bkg_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_4_Bkg_EE_ = ibook.book1D("Ele_eta_MTD_4_Bkg_EE", "Electron eta MTD;#eta;Counts", 32, 1.6, 3.2);
    meEle_phi_MTD_4_Bkg_EE_ = ibook.book1D("Ele_phi_MTD_4_Bkg_EE", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_5_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_5_Bkg_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_5_Bkg_EE_ = ibook.book1D("Ele_eta_MTD_5_Bkg_EE", "Electron eta MTD;#eta;Counts", 32, 1.6, 3.2);
    meEle_phi_MTD_5_Bkg_EE_ = ibook.book1D("Ele_phi_MTD_5_Bkg_EE", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_6_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_6_Bkg_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_6_Bkg_EE_ = ibook.book1D("Ele_eta_MTD_6_Bkg_EE", "Electron eta MTD;#eta;Counts", 32, 1.6, 3.2);
    meEle_phi_MTD_6_Bkg_EE_ = ibook.book1D("Ele_phi_MTD_6_Bkg_EE", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_MTD_7_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_7_Bkg_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_eta_MTD_7_Bkg_EE_ = ibook.book1D("Ele_eta_MTD_7_Bkg_EE", "Electron eta MTD;#eta;Counts", 32, 1.6, 3.2);
    meEle_phi_MTD_7_Bkg_EE_ = ibook.book1D("Ele_phi_MTD_7_Bkg_EE", "Electron phi MTD;#phi;Counts", 64, -3.2, 3.2);

    meEle_pt_sim_MTD_1_Bkg_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_1_Bkg_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_2_Bkg_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_2_Bkg_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_3_Bkg_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_3_Bkg_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_4_Bkg_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_4_Bkg_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_5_Bkg_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_5_Bkg_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_6_Bkg_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_6_Bkg_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
    meEle_pt_sim_MTD_7_Bkg_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_7_Bkg_EE", "Electron pT MTD;p_{T} (GeV);Counts", 30, 10, 100);
  }

  meEle_pt_MTD_4sigma_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_4sigma_Bkg_EE",
                                             "Electron pT MTD - 4 sigma compatibility - Bkg Endcap;p_{T} (GeV);Counts",
                                             30,
                                             10,
                                             100);
  meEle_eta_MTD_4sigma_Bkg_EE_ = ibook.book1D(
      "Ele_eta_MTD_4sigma_Bkg_EE", "Electron eta MTD - 4 sigma compatibility - Bkg Endcapi;#eta;Counts", 32, 1.6, 3.2);
  meEle_phi_MTD_4sigma_Bkg_EE_ = ibook.book1D(
      "Ele_phi_MTD_4sigma_Bkg_EE", "Electron phi MTD - 4 sigma compatibility - Bkg Endcap;#phi;Counts", 64, -3.2, 3.2);

  meEle_pt_MTD_3sigma_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_3sigma_Bkg_EE",
                                             "Electron pT MTD - 3 sigma compatibility - Bkg Endcap;p_{T} (GeV);Counts",
                                             30,
                                             10,
                                             100);
  meEle_eta_MTD_3sigma_Bkg_EE_ = ibook.book1D(
      "Ele_eta_MTD_3sigma_Bkg_EE", "Electron eta MTD - 3 sigma compatibility - Bkg Endcap;#eta;Counts", 32, 1.6, 3.2);
  meEle_phi_MTD_3sigma_Bkg_EE_ = ibook.book1D(
      "Ele_phi_MTD_3sigma_Bkg_EE", "Electron phi MTD - 3 sigma compatibility - Bkg Endcap;#phi;Counts", 64, -3.2, 3.2);

  meEle_pt_MTD_2sigma_Bkg_EE_ = ibook.book1D("Ele_pT_MTD_2sigma_Bkg_EE",
                                             "Electron pT MTD - 2 sigma compatibility - Bkg Endcap;p_{T} (GeV);Counts",
                                             30,
                                             10,
                                             100);
  meEle_eta_MTD_2sigma_Bkg_EE_ = ibook.book1D(
      "Ele_eta_MTD_2sigma_Bkg_EE", "Electron eta MTD - 2 sigma compatibility - Bkg Endcap;#eta;Counts", 32, 1.6, 3.2);
  meEle_phi_MTD_2sigma_Bkg_EE_ = ibook.book1D(
      "Ele_phi_MTD_2sigma_Bkg_EE", "Electron phi MTD - 2 sigma compatibility - Bkg Endcap;#phi;Counts", 64, -3.2, 3.2);

  if (optionalPlots_) {
    meEle_pt_sim_MTD_4sigma_Bkg_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_4sigma_Bkg_EE",
                     "Electron pT MTD SIM - 4 sigma compatibility - Bkg Endcap;p_{T} (GeV);Counts",
                     30,
                     10,
                     100);
    meEle_pt_sim_MTD_3sigma_Bkg_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_3sigma_Bkg_EE",
                     "Electron pT MTD SIM - 3 sigma compatibility - Bkg Endcap;#eta;Counts",
                     30,
                     10,
                     100);
    meEle_pt_sim_MTD_2sigma_Bkg_EE_ =
        ibook.book1D("Ele_pT_sim_MTD_2sigma_Bkg_EE",
                     "Electron pT MTD SIM - 2 sigma compatibility - Bkg Endcap;#phi;Counts",
                     30,
                     10,
                     100);
  }

  // defining vectors for more efficient hist filling
  // Promt part
  if (optionalPlots_) {
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
  }
  Ntracks_EB_list_Significance_Sig = {
      meEleISO_Ntracks_MTD_4sigma_Sig_EB_, meEleISO_Ntracks_MTD_3sigma_Sig_EB_, meEleISO_Ntracks_MTD_2sigma_Sig_EB_};
  ch_iso_EB_list_Significance_Sig = {
      meEleISO_chIso_MTD_4sigma_Sig_EB_, meEleISO_chIso_MTD_3sigma_Sig_EB_, meEleISO_chIso_MTD_2sigma_Sig_EB_};
  rel_ch_iso_EB_list_Significance_Sig = {meEleISO_rel_chIso_MTD_4sigma_Sig_EB_,
                                         meEleISO_rel_chIso_MTD_3sigma_Sig_EB_,
                                         meEleISO_rel_chIso_MTD_2sigma_Sig_EB_};

  if (optionalPlots_) {
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
  }
  Ntracks_EE_list_Significance_Sig = {
      meEleISO_Ntracks_MTD_4sigma_Sig_EE_, meEleISO_Ntracks_MTD_3sigma_Sig_EE_, meEleISO_Ntracks_MTD_2sigma_Sig_EE_};
  ch_iso_EE_list_Significance_Sig = {
      meEleISO_chIso_MTD_4sigma_Sig_EE_, meEleISO_chIso_MTD_3sigma_Sig_EE_, meEleISO_chIso_MTD_2sigma_Sig_EE_};
  rel_ch_iso_EE_list_Significance_Sig = {meEleISO_rel_chIso_MTD_4sigma_Sig_EE_,
                                         meEleISO_rel_chIso_MTD_3sigma_Sig_EE_,
                                         meEleISO_rel_chIso_MTD_2sigma_Sig_EE_};

  if (optionalPlots_) {
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
  }

  Ele_pT_MTD_EB_list_Significance_Sig = {
      meEle_pt_MTD_4sigma_Sig_EB_, meEle_pt_MTD_3sigma_Sig_EB_, meEle_pt_MTD_2sigma_Sig_EB_};
  Ele_eta_MTD_EB_list_Significance_Sig = {
      meEle_eta_MTD_4sigma_Sig_EB_, meEle_eta_MTD_3sigma_Sig_EB_, meEle_eta_MTD_2sigma_Sig_EB_};
  Ele_phi_MTD_EB_list_Significance_Sig = {
      meEle_phi_MTD_4sigma_Sig_EB_, meEle_phi_MTD_3sigma_Sig_EB_, meEle_phi_MTD_2sigma_Sig_EB_};

  if (optionalPlots_) {
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
  }
  Ele_pT_MTD_EE_list_Significance_Sig = {
      meEle_pt_MTD_4sigma_Sig_EE_, meEle_pt_MTD_3sigma_Sig_EE_, meEle_pt_MTD_2sigma_Sig_EE_};
  Ele_eta_MTD_EE_list_Significance_Sig = {
      meEle_eta_MTD_4sigma_Sig_EE_, meEle_eta_MTD_3sigma_Sig_EE_, meEle_eta_MTD_2sigma_Sig_EE_};
  Ele_phi_MTD_EE_list_Significance_Sig = {
      meEle_phi_MTD_4sigma_Sig_EE_, meEle_phi_MTD_3sigma_Sig_EE_, meEle_phi_MTD_2sigma_Sig_EE_};

  // For SIM CASE
  if (optionalPlots_) {
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
  }

  // Non-promt part
  if (optionalPlots_) {
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
  }
  Ntracks_EB_list_Significance_Bkg = {
      meEleISO_Ntracks_MTD_4sigma_Bkg_EB_, meEleISO_Ntracks_MTD_3sigma_Bkg_EB_, meEleISO_Ntracks_MTD_2sigma_Bkg_EB_};
  ch_iso_EB_list_Significance_Bkg = {
      meEleISO_chIso_MTD_4sigma_Bkg_EB_, meEleISO_chIso_MTD_3sigma_Bkg_EB_, meEleISO_chIso_MTD_2sigma_Bkg_EB_};
  rel_ch_iso_EB_list_Significance_Bkg = {meEleISO_rel_chIso_MTD_4sigma_Bkg_EB_,
                                         meEleISO_rel_chIso_MTD_3sigma_Bkg_EB_,
                                         meEleISO_rel_chIso_MTD_2sigma_Bkg_EB_};

  if (optionalPlots_) {
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
  }
  Ntracks_EE_list_Significance_Bkg = {
      meEleISO_Ntracks_MTD_4sigma_Bkg_EE_, meEleISO_Ntracks_MTD_3sigma_Bkg_EE_, meEleISO_Ntracks_MTD_2sigma_Bkg_EE_};
  ch_iso_EE_list_Significance_Bkg = {
      meEleISO_chIso_MTD_4sigma_Bkg_EE_, meEleISO_chIso_MTD_3sigma_Bkg_EE_, meEleISO_chIso_MTD_2sigma_Bkg_EE_};
  rel_ch_iso_EE_list_Significance_Bkg = {meEleISO_rel_chIso_MTD_4sigma_Bkg_EE_,
                                         meEleISO_rel_chIso_MTD_3sigma_Bkg_EE_,
                                         meEleISO_rel_chIso_MTD_2sigma_Bkg_EE_};
  if (optionalPlots_) {
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
  }
  Ele_pT_MTD_EB_list_Significance_Bkg = {
      meEle_pt_MTD_4sigma_Bkg_EB_, meEle_pt_MTD_3sigma_Bkg_EB_, meEle_pt_MTD_2sigma_Bkg_EB_};
  Ele_eta_MTD_EB_list_Significance_Bkg = {
      meEle_eta_MTD_4sigma_Bkg_EB_, meEle_eta_MTD_3sigma_Bkg_EB_, meEle_eta_MTD_2sigma_Bkg_EB_};
  Ele_phi_MTD_EB_list_Significance_Bkg = {
      meEle_phi_MTD_4sigma_Bkg_EB_, meEle_phi_MTD_3sigma_Bkg_EB_, meEle_phi_MTD_2sigma_Bkg_EB_};

  if (optionalPlots_) {
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
  }
  Ele_pT_MTD_EE_list_Significance_Bkg = {
      meEle_pt_MTD_4sigma_Bkg_EE_, meEle_pt_MTD_3sigma_Bkg_EE_, meEle_pt_MTD_2sigma_Bkg_EE_};
  Ele_eta_MTD_EE_list_Significance_Bkg = {
      meEle_eta_MTD_4sigma_Bkg_EE_, meEle_eta_MTD_3sigma_Bkg_EE_, meEle_eta_MTD_2sigma_Bkg_EE_};
  Ele_phi_MTD_EE_list_Significance_Bkg = {
      meEle_phi_MTD_4sigma_Bkg_EE_, meEle_phi_MTD_3sigma_Bkg_EE_, meEle_phi_MTD_2sigma_Bkg_EE_};

  // SIM CASE
  if (optionalPlots_) {
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
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------

void MtdEleIsoValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/ElectronIso");
  desc.add<edm::InputTag>("inputTagG", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("inputTag_vtx", edm::InputTag("offlinePrimaryVertices4D"));
  desc.add<edm::InputTag>("inputEle_EB", edm::InputTag("gedGsfElectrons"));
  desc.add<edm::InputTag>("inputEle_EE", edm::InputTag("ecalDrivenGsfElectronsHGC"));
  desc.add<edm::InputTag>("TPtoRecoTrackAssoc", edm::InputTag("trackingParticleRecoTrackAsssociation"));
  desc.add<edm::InputTag>("t0PID", edm::InputTag("tofPID:t0"));
  desc.add<edm::InputTag>("sigmat0PID", edm::InputTag("tofPID:sigmat0"));
  desc.add<edm::InputTag>("trackMVAQual", edm::InputTag("mtdTrackQualityMVA:mtdQualMVA"));
  desc.add<double>("trackMinimumPt", 1.0);  // [GeV]
  desc.add<double>("trackMinimumEta", 1.5);
  desc.add<double>("trackMaximumEta", 3.2);
  desc.add<double>("rel_iso_cut", 0.08);
  desc.add<bool>("optionTrackMatchToPV", false);
  desc.add<bool>("option_dtToTrack", true);  // default is dt with track, if false will do dt to vertex
  desc.add<bool>("option_plots", false);
  desc.add<double>("min_dR_cut", 0.01);
  desc.add<double>("max_dR_cut", 0.3);
  desc.add<double>("min_pt_cut_EB", 0.7);
  desc.add<double>("min_pt_cut_EE", 0.4);
  desc.add<double>("max_dz_cut_EB", 0.5);  // PARAM
  desc.add<double>("max_dz_cut_EE", 0.5);  // PARAM
  desc.add<double>("max_dz_vtx_cut", 0.5);
  desc.add<double>("max_dxy_vtx_cut", 0.2);
  desc.add<double>("min_strip_cut", 0.01);
  desc.add<double>("min_track_mtd_mva_cut", 0.5);

  descriptions.add("mtdEleIsoValid", desc);
}

DEFINE_FWK_MODULE(MtdEleIsoValidation);

//*/

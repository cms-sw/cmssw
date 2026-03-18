// Based on RecoNtuple/HGCalAnalysis with modifications for PF
// Used for MLPF training
// Author and maintainer: Joosep Pata (KBFI, Tallinn, Estonia)
// cms-mlpf@cern.ch

#include <map>
#include <set>
#include <string>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "RecoParticleFlow/PFProducer/interface/MLPFModel.h"
#include "SimDataFormats/Associations/interface/TrackAssociation.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "RecoParticleFlow/PFProducer/interface/MLPFModel.h"
#include <TTree.h>

using namespace std;

class ElementWithIndex {
public:
  const reco::PFBlockElement& orig;
  size_t idx_block;
  size_t idx_elem;
  ElementWithIndex(const reco::PFBlockElement& _orig, size_t _idx_block, size_t _idx_elem)
      : orig(_orig), idx_block(_idx_block), idx_elem(_idx_elem) {}
};

vector<int> find_element_ref(const vector<ElementWithIndex>& vec, const edm::RefToBase<reco::Track>& r) {
  vector<int> ret;
  for (unsigned int i = 0; i < vec.size(); i++) {
    const auto& elem = vec.at(i);
    if (elem.orig.type() == reco::PFBlockElement::TRACK) {
      const auto& ref = elem.orig.trackRef();
      if (ref.isNonnull() && ref->extra().isNonnull()) {
        if (ref.key() == r.key()) {
          ret.push_back(i);
        }
      }
    } else if (elem.orig.type() == reco::PFBlockElement::GSF) {
      const auto& ref = ((const reco::PFBlockElementGsfTrack*)&elem.orig)->GsftrackRef();
      if (ref.isNonnull()) {
        if (ref.key() == r.key()) {
          ret.push_back(i);
        }
      }
    } else if (elem.orig.type() == reco::PFBlockElement::BREM) {
      const auto& ref = ((const reco::PFBlockElementBrem*)&elem.orig)->GsftrackRef();
      if (ref.isNonnull()) {
        if (ref.key() == r.key()) {
          ret.push_back(i);
        }
      }
    }
  }
  return ret;
}

class PFAnalysis : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  PFAnalysis();
  explicit PFAnalysis(const edm::ParameterSet&);
  ~PFAnalysis() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void processTrackingParticles(const edm::View<TrackingParticle>& trackingParticles,
                                edm::Handle<edm::View<TrackingParticle>>& trackingParticlesHandle);

  pair<vector<ElementWithIndex>, vector<tuple<int, int, float>>> processBlocks(
      const std::vector<reco::PFBlock>& pfBlocks);

  void clearVariables();

  // ----------member data ---------------------------

  edm::EDGetTokenT<std::vector<reco::GenParticle>> genParticles_;
  edm::EDGetTokenT<edm::View<TrackingParticle>> trackingParticles_;
  edm::EDGetTokenT<edm::View<CaloParticle>> caloParticles_;
  edm::EDGetTokenT<edm::View<reco::Track>> tracks_;
  edm::EDGetTokenT<edm::View<reco::Track>> gsftracks_;
  edm::EDGetTokenT<std::vector<reco::PFBlock>> pfBlocks_;
  edm::EDGetTokenT<std::vector<reco::PFCandidate>> pfCandidates_;
  edm::EDGetTokenT<reco::RecoToSimCollection> tracks_recotosim_;
  edm::EDGetTokenT<reco::RecoToSimCollection> gsf_recotosim_;
  edm::EDGetTokenT<edm::View<reco::GsfElectron>> gsfElectrons_;
  edm::EDGetTokenT<edm::View<reco::GenJet>> genJets_;
  edm::EDGetTokenT<edm::View<reco::GenMET>> genMETs_;

  TTree* t_;

  edm::RunNumber_t ev_run_;
  edm::LuminosityBlockNumber_t ev_lumi_;
  edm::EventNumber_t ev_event_;

  vector<float> trackingparticle_eta_;
  vector<float> trackingparticle_phi_;
  vector<float> trackingparticle_pt_;
  vector<float> trackingparticle_px_;
  vector<float> trackingparticle_py_;
  vector<float> trackingparticle_pz_;
  vector<float> trackingparticle_energy_;
  vector<float> trackingparticle_dvx_;
  vector<float> trackingparticle_dvy_;
  vector<float> trackingparticle_dvz_;
  vector<int> trackingparticle_bx_;
  vector<int> trackingparticle_ev_;
  vector<float> trackingparticle_ovx_;
  vector<float> trackingparticle_ovy_;
  vector<float> trackingparticle_ovz_;
  vector<float> trackingparticle_exx_;
  vector<float> trackingparticle_exy_;
  vector<int> trackingparticle_mother_;
  vector<int> trackingparticle_pid_;
  vector<int> trackingparticle_charge_;

  vector<float> simcluster_eta_;
  vector<float> simcluster_phi_;
  vector<float> simcluster_pt_;
  vector<float> simcluster_energy_;
  vector<float> simcluster_px_;
  vector<float> simcluster_py_;
  vector<float> simcluster_pz_;
  vector<int> simcluster_bx_;
  vector<int> simcluster_ev_;
  vector<int> simcluster_pid_;
  vector<int> simcluster_charge_;
  vector<int> simcluster_idx_trackingparticle_;
  vector<int> simcluster_idx_caloparticle_;
  vector<int> simcluster_nhits_;
  vector<std::map<uint64_t, double>> simcluster_detids_;

  vector<float> caloparticle_eta_;
  vector<float> caloparticle_phi_;
  vector<float> caloparticle_pt_;
  vector<float> caloparticle_energy_;
  vector<float> caloparticle_simenergy_;
  vector<float> caloparticle_bx_;
  vector<float> caloparticle_ev_;
  vector<int> caloparticle_pid_;
  vector<int> caloparticle_charge_;
  vector<int> caloparticle_idx_trackingparticle_;

  vector<float> simhit_frac_;
  vector<float> simhit_x_;
  vector<float> simhit_y_;
  vector<float> simhit_z_;
  vector<float> simhit_eta_;
  vector<float> simhit_phi_;
  vector<int> simhit_det_;
  vector<int> simhit_subdet_;
  vector<int> simhit_idx_simcluster_;
  vector<uint64_t> simhit_detid_;

  vector<float> rechit_e_;
  vector<float> rechit_x_;
  vector<float> rechit_y_;
  vector<float> rechit_z_;
  vector<float> rechit_det_;
  vector<float> rechit_subdet_;
  vector<float> rechit_eta_;
  vector<float> rechit_phi_;
  vector<int> rechit_idx_element_;
  vector<uint64_t> rechit_detid_;

  vector<float> gen_eta_;
  vector<float> gen_phi_;
  vector<float> gen_pt_;
  vector<float> gen_px_;
  vector<float> gen_py_;
  vector<float> gen_pz_;
  vector<float> gen_energy_;
  vector<int> gen_charge_;
  vector<int> gen_pdgid_;
  vector<int> gen_status_;
  vector<vector<int>> gen_daughters_;

  vector<float> genjet_pt_;
  vector<float> genjet_eta_;
  vector<float> genjet_phi_;
  vector<float> genjet_energy_;

  vector<float> genmet_pt_;
  vector<float> genmet_phi_;

  vector<float> element_pt_;
  vector<float> element_pterror_;
  vector<float> element_px_;
  vector<float> element_py_;
  vector<float> element_pz_;
  vector<float> element_sigma_x_;
  vector<float> element_sigma_y_;
  vector<float> element_sigma_z_;
  vector<float> element_deltap_;
  vector<float> element_sigmadeltap_;
  vector<float> element_eta_;
  vector<float> element_etaerror_;
  vector<float> element_phi_;
  vector<float> element_phierror_;
  vector<float> element_energy_;
  vector<float> element_corr_energy_;
  vector<float> element_corr_energy_err_;
  vector<float> element_eta_ecal_;
  vector<float> element_phi_ecal_;
  vector<float> element_eta_hcal_;
  vector<float> element_phi_hcal_;
  vector<int> element_charge_;
  vector<int> element_type_;
  vector<int> element_layer_;
  vector<float> element_depth_;
  vector<float> element_trajpoint_;
  vector<float> element_muon_dt_hits_;
  vector<float> element_muon_csc_hits_;
  vector<float> element_muon_type_;
  vector<float> element_cluster_flags_;
  vector<float> element_gsf_electronseed_trkorecal_;
  vector<float> element_gsf_electronseed_dnn1_;
  vector<float> element_gsf_electronseed_dnn2_;
  vector<float> element_gsf_electronseed_dnn3_;
  vector<float> element_gsf_electronseed_dnn4_;
  vector<float> element_gsf_electronseed_dnn5_;
  vector<float> element_num_hits_;
  vector<float> element_lambda_;
  vector<float> element_lambdaerror_;
  vector<float> element_theta_;
  vector<float> element_thetaerror_;
  vector<float> element_vx_;
  vector<float> element_vy_;
  vector<float> element_vz_;
  vector<float> element_time_;
  vector<float> element_timeerror_;
  vector<float> element_etaerror1_;
  vector<float> element_etaerror2_;
  vector<float> element_etaerror3_;
  vector<float> element_etaerror4_;
  vector<float> element_phierror1_;
  vector<float> element_phierror2_;
  vector<float> element_phierror3_;
  vector<float> element_phierror4_;

  vector<int> element_distance_i_;
  vector<int> element_distance_j_;
  vector<float> element_distance_d_;

  vector<float> pfcandidate_eta_;
  vector<float> pfcandidate_phi_;
  vector<float> pfcandidate_pt_;
  vector<float> pfcandidate_px_;
  vector<float> pfcandidate_py_;
  vector<float> pfcandidate_pz_;
  vector<float> pfcandidate_energy_;
  vector<int> pfcandidate_pdgid_;

  vector<pair<int, int>> trackingparticle_to_element;
  vector<float> trackingparticle_to_element_cmp;
  vector<pair<int, int>> caloparticle_to_element;
  vector<float> caloparticle_to_element_cmp;
  vector<pair<int, int>> simcluster_to_element;
  vector<float> simcluster_to_element_cmp;
  vector<pair<int, int>> element_to_candidate;
  vector<pair<int, int>> caloparticle_to_simcluster;

  bool saveHits;
};

PFAnalysis::PFAnalysis() { ; }

PFAnalysis::PFAnalysis(const edm::ParameterSet& iConfig) {
  tracks_recotosim_ = consumes<reco::RecoToSimCollection>(edm::InputTag("trackingParticleRecoTrackAsssociation"));
  gsf_recotosim_ = consumes<reco::RecoToSimCollection>(edm::InputTag("trackingParticleGsfTrackAssociation"));
  trackingParticles_ = consumes<edm::View<TrackingParticle>>(edm::InputTag("mix", "MergedTrackTruth"));
  caloParticles_ = consumes<edm::View<CaloParticle>>(edm::InputTag("mix", "MergedCaloTruth"));
  genParticles_ = consumes<std::vector<reco::GenParticle>>(edm::InputTag("genParticles"));
  pfBlocks_ = consumes<std::vector<reco::PFBlock>>(edm::InputTag("particleFlowBlock"));
  pfCandidates_ = consumes<std::vector<reco::PFCandidate>>(edm::InputTag("particleFlow"));
  tracks_ = consumes<edm::View<reco::Track>>(edm::InputTag("generalTracks"));
  gsftracks_ = consumes<edm::View<reco::Track>>(edm::InputTag("electronGsfTracks"));
  saveHits = iConfig.getUntrackedParameter<bool>("saveHits", false);
  gsfElectrons_ = consumes<edm::View<reco::GsfElectron>>(edm::InputTag("gedGsfElectrons"));
  genJets_ = consumes<edm::View<reco::GenJet>>(edm::InputTag("ak4GenJets"));
  genMETs_ = consumes<edm::View<reco::GenMET>>(edm::InputTag("genMetTrue"));

  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;

  t_ = fs->make<TTree>("pftree", "pftree");

  // event info
  t_->Branch("event", &ev_event_);
  t_->Branch("lumi", &ev_lumi_);
  t_->Branch("run", &ev_run_);

  t_->Branch("trackingparticle_eta", &trackingparticle_eta_);
  t_->Branch("trackingparticle_phi", &trackingparticle_phi_);
  t_->Branch("trackingparticle_pt", &trackingparticle_pt_);
  t_->Branch("trackingparticle_px", &trackingparticle_px_);
  t_->Branch("trackingparticle_py", &trackingparticle_py_);
  t_->Branch("trackingparticle_pz", &trackingparticle_pz_);
  t_->Branch("trackingparticle_energy", &trackingparticle_energy_);
  t_->Branch("trackingparticle_dvx", &trackingparticle_dvx_);
  t_->Branch("trackingparticle_dvy", &trackingparticle_dvy_);
  t_->Branch("trackingparticle_dvz", &trackingparticle_dvz_);
  t_->Branch("trackingparticle_bx", &trackingparticle_bx_);
  t_->Branch("trackingparticle_ev", &trackingparticle_ev_);
  t_->Branch("trackingparticle_pid", &trackingparticle_pid_);
  t_->Branch("trackingparticle_charge", &trackingparticle_charge_);

  t_->Branch("simcluster_eta", &simcluster_eta_);
  t_->Branch("simcluster_phi", &simcluster_phi_);
  t_->Branch("simcluster_pt", &simcluster_pt_);
  t_->Branch("simcluster_energy", &simcluster_energy_);
  t_->Branch("simcluster_bx", &simcluster_bx_);
  t_->Branch("simcluster_ev", &simcluster_ev_);
  t_->Branch("simcluster_pid", &simcluster_pid_);
  t_->Branch("simcluster_charge", &simcluster_charge_);
  t_->Branch("simcluster_idx_trackingparticle", &simcluster_idx_trackingparticle_);
  t_->Branch("simcluster_idx_caloparticle", &simcluster_idx_caloparticle_);

  t_->Branch("caloparticle_eta", &caloparticle_eta_);
  t_->Branch("caloparticle_phi", &caloparticle_phi_);
  t_->Branch("caloparticle_pt", &caloparticle_pt_);
  t_->Branch("caloparticle_energy", &caloparticle_energy_);
  t_->Branch("caloparticle_simenergy", &caloparticle_simenergy_);
  t_->Branch("caloparticle_bx", &caloparticle_bx_);
  t_->Branch("caloparticle_ev", &caloparticle_ev_);
  t_->Branch("caloparticle_pid", &caloparticle_pid_);
  t_->Branch("caloparticle_charge", &caloparticle_charge_);
  t_->Branch("caloparticle_idx_trackingparticle", &caloparticle_idx_trackingparticle_);

  t_->Branch("gen_eta", &gen_eta_);
  t_->Branch("gen_phi", &gen_phi_);
  t_->Branch("gen_pt", &gen_pt_);
  t_->Branch("gen_px", &gen_px_);
  t_->Branch("gen_py", &gen_py_);
  t_->Branch("gen_pz", &gen_pz_);
  t_->Branch("gen_energy", &gen_energy_);
  t_->Branch("gen_charge", &gen_charge_);
  t_->Branch("gen_pdgid", &gen_pdgid_);
  t_->Branch("gen_status", &gen_status_);
  t_->Branch("gen_daughters", &gen_daughters_);

  t_->Branch("genjet_pt", &genjet_pt_);
  t_->Branch("genjet_eta", &genjet_eta_);
  t_->Branch("genjet_phi", &genjet_phi_);
  t_->Branch("genjet_energy", &genjet_energy_);

  t_->Branch("genmet_pt", &genmet_pt_);
  t_->Branch("genmet_phi", &genmet_phi_);

  //PF Elements
  t_->Branch("element_pt", &element_pt_);
  t_->Branch("element_pterror", &element_pterror_);
  t_->Branch("element_px", &element_px_);
  t_->Branch("element_py", &element_py_);
  t_->Branch("element_pz", &element_pz_);
  t_->Branch("element_sigma_x", &element_sigma_x_);
  t_->Branch("element_sigma_y", &element_sigma_y_);
  t_->Branch("element_sigma_z", &element_sigma_z_);
  t_->Branch("element_deltap", &element_deltap_);
  t_->Branch("element_sigmadeltap", &element_sigmadeltap_);
  t_->Branch("element_eta", &element_eta_);
  t_->Branch("element_etaerror", &element_etaerror_);
  t_->Branch("element_phi", &element_phi_);
  t_->Branch("element_phierror", &element_phierror_);
  t_->Branch("element_energy", &element_energy_);
  t_->Branch("element_corr_energy", &element_corr_energy_);
  t_->Branch("element_corr_energy_err", &element_corr_energy_err_);
  t_->Branch("element_eta_ecal", &element_eta_ecal_);
  t_->Branch("element_phi_ecal", &element_phi_ecal_);
  t_->Branch("element_eta_hcal", &element_eta_hcal_);
  t_->Branch("element_phi_hcal", &element_phi_hcal_);
  t_->Branch("element_charge", &element_charge_);
  t_->Branch("element_type", &element_type_);
  t_->Branch("element_layer", &element_layer_);
  t_->Branch("element_depth", &element_depth_);
  t_->Branch("element_trajpoint", &element_trajpoint_);
  t_->Branch("element_muon_dt_hits", &element_muon_dt_hits_);
  t_->Branch("element_muon_csc_hits", &element_muon_csc_hits_);
  t_->Branch("element_muon_type", &element_muon_type_);
  t_->Branch("element_cluster_flags", &element_cluster_flags_);
  t_->Branch("element_gsf_electronseed_trkorecal", &element_gsf_electronseed_trkorecal_);
  t_->Branch("element_gsf_electronseed_dnn1", &element_gsf_electronseed_dnn1_);
  t_->Branch("element_gsf_electronseed_dnn2", &element_gsf_electronseed_dnn2_);
  t_->Branch("element_gsf_electronseed_dnn3", &element_gsf_electronseed_dnn3_);
  t_->Branch("element_gsf_electronseed_dnn4", &element_gsf_electronseed_dnn4_);
  t_->Branch("element_gsf_electronseed_dnn5", &element_gsf_electronseed_dnn5_);
  t_->Branch("element_num_hits", &element_num_hits_);
  t_->Branch("element_lambda", &element_lambda_);
  t_->Branch("element_lambdaerror", &element_lambdaerror_);
  t_->Branch("element_theta", &element_theta_);
  t_->Branch("element_thetaerror", &element_thetaerror_);
  t_->Branch("element_vx", &element_vx_);
  t_->Branch("element_vy", &element_vy_);
  t_->Branch("element_vz", &element_vz_);
  t_->Branch("element_time", &element_time_);
  t_->Branch("element_timeerror", &element_timeerror_);
  t_->Branch("element_etaerror1", &element_etaerror1_);
  t_->Branch("element_etaerror2", &element_etaerror2_);
  t_->Branch("element_etaerror3", &element_etaerror3_);
  t_->Branch("element_etaerror4", &element_etaerror4_);
  t_->Branch("element_phierror1", &element_phierror1_);
  t_->Branch("element_phierror2", &element_phierror2_);
  t_->Branch("element_phierror3", &element_phierror3_);
  t_->Branch("element_phierror4", &element_phierror4_);

  //Distance matrix between PF elements
  t_->Branch("element_distance_i", &element_distance_i_);
  t_->Branch("element_distance_j", &element_distance_j_);
  t_->Branch("element_distance_d", &element_distance_d_);

  t_->Branch("pfcandidate_eta", &pfcandidate_eta_);
  t_->Branch("pfcandidate_phi", &pfcandidate_phi_);
  t_->Branch("pfcandidate_pt", &pfcandidate_pt_);
  t_->Branch("pfcandidate_px", &pfcandidate_px_);
  t_->Branch("pfcandidate_py", &pfcandidate_py_);
  t_->Branch("pfcandidate_pz", &pfcandidate_pz_);
  t_->Branch("pfcandidate_energy", &pfcandidate_energy_);
  t_->Branch("pfcandidate_pdgid", &pfcandidate_pdgid_);

  //Links between reco, gen and PFCandidate objects
  t_->Branch("trackingparticle_to_element", &trackingparticle_to_element);
  t_->Branch("trackingparticle_to_element_cmp", &trackingparticle_to_element_cmp);
  t_->Branch("caloparticle_to_element", &caloparticle_to_element);
  t_->Branch("caloparticle_to_element_cmp", &caloparticle_to_element_cmp);
  t_->Branch("simcluster_to_element", &simcluster_to_element);
  t_->Branch("simcluster_to_element_cmp", &simcluster_to_element_cmp);
  t_->Branch("element_to_candidate", &element_to_candidate);
  t_->Branch("caloparticle_to_simcluster", &caloparticle_to_simcluster);
}  // constructor

PFAnalysis::~PFAnalysis() {}

void PFAnalysis::clearVariables() {
  ev_run_ = 0;
  ev_lumi_ = 0;
  ev_event_ = 0;

  trackingparticle_to_element.clear();
  trackingparticle_to_element_cmp.clear();
  caloparticle_to_element.clear();
  caloparticle_to_element_cmp.clear();
  simcluster_to_element.clear();
  simcluster_to_element_cmp.clear();
  element_to_candidate.clear();
  caloparticle_to_simcluster.clear();

  trackingparticle_eta_.clear();
  trackingparticle_phi_.clear();
  trackingparticle_pt_.clear();
  trackingparticle_px_.clear();
  trackingparticle_py_.clear();
  trackingparticle_pz_.clear();
  trackingparticle_energy_.clear();
  trackingparticle_dvx_.clear();
  trackingparticle_dvy_.clear();
  trackingparticle_dvz_.clear();
  trackingparticle_bx_.clear();
  trackingparticle_ev_.clear();
  trackingparticle_ovx_.clear();
  trackingparticle_ovy_.clear();
  trackingparticle_ovz_.clear();
  trackingparticle_exx_.clear();
  trackingparticle_exy_.clear();
  trackingparticle_mother_.clear();
  trackingparticle_pid_.clear();
  trackingparticle_charge_.clear();

  simcluster_eta_.clear();
  simcluster_phi_.clear();
  simcluster_pt_.clear();
  simcluster_energy_.clear();
  simcluster_pid_.clear();
  simcluster_charge_.clear();
  simcluster_bx_.clear();
  simcluster_ev_.clear();
  simcluster_idx_trackingparticle_.clear();
  simcluster_idx_caloparticle_.clear();

  caloparticle_pt_.clear();
  caloparticle_eta_.clear();
  caloparticle_phi_.clear();
  caloparticle_energy_.clear();
  caloparticle_simenergy_.clear();
  caloparticle_bx_.clear();
  caloparticle_ev_.clear();
  caloparticle_pid_.clear();
  caloparticle_charge_.clear();
  caloparticle_idx_trackingparticle_.clear();

  gen_eta_.clear();
  gen_phi_.clear();
  gen_pt_.clear();
  gen_px_.clear();
  gen_py_.clear();
  gen_pz_.clear();
  gen_energy_.clear();
  gen_charge_.clear();
  gen_pdgid_.clear();
  gen_status_.clear();
  gen_daughters_.clear();

  genjet_pt_.clear();
  genjet_eta_.clear();
  genjet_phi_.clear();
  genjet_energy_.clear();

  genmet_pt_.clear();
  genmet_phi_.clear();

  element_pt_.clear();
  element_pterror_.clear();
  element_px_.clear();
  element_py_.clear();
  element_pz_.clear();
  element_sigma_x_.clear();
  element_sigma_y_.clear();
  element_sigma_z_.clear();
  element_deltap_.clear();
  element_sigmadeltap_.clear();
  element_eta_.clear();
  element_etaerror_.clear();
  element_phi_.clear();
  element_phierror_.clear();
  element_energy_.clear();
  element_corr_energy_.clear();
  element_corr_energy_err_.clear();
  element_eta_ecal_.clear();
  element_phi_ecal_.clear();
  element_eta_hcal_.clear();
  element_phi_hcal_.clear();
  element_charge_.clear();
  element_type_.clear();
  element_layer_.clear();
  element_depth_.clear();
  element_trajpoint_.clear();
  element_muon_dt_hits_.clear();
  element_muon_csc_hits_.clear();
  element_muon_type_.clear();
  element_cluster_flags_.clear();
  element_gsf_electronseed_trkorecal_.clear();
  element_gsf_electronseed_dnn1_.clear();
  element_gsf_electronseed_dnn2_.clear();
  element_gsf_electronseed_dnn3_.clear();
  element_gsf_electronseed_dnn4_.clear();
  element_gsf_electronseed_dnn5_.clear();
  element_num_hits_.clear();
  element_lambda_.clear();
  element_lambdaerror_.clear();
  element_theta_.clear();
  element_thetaerror_.clear();
  element_vx_.clear();
  element_vy_.clear();
  element_vz_.clear();
  element_time_.clear();
  element_timeerror_.clear();
  element_etaerror1_.clear();
  element_etaerror2_.clear();
  element_etaerror3_.clear();
  element_etaerror4_.clear();
  element_phierror1_.clear();
  element_phierror2_.clear();
  element_phierror3_.clear();
  element_phierror4_.clear();

  element_distance_i_.clear();
  element_distance_j_.clear();
  element_distance_d_.clear();

  pfcandidate_eta_.clear();
  pfcandidate_phi_.clear();
  pfcandidate_pt_.clear();
  pfcandidate_px_.clear();
  pfcandidate_py_.clear();
  pfcandidate_pz_.clear();
  pfcandidate_energy_.clear();
  pfcandidate_pdgid_.clear();

}  //clearVariables

void PFAnalysis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  clearVariables();

  //Simulated tracks, cleaned up by TrackingTruthAccumulator
  edm::Handle<edm::View<TrackingParticle>> trackingParticlesHandle;
  iEvent.getByToken(trackingParticles_, trackingParticlesHandle);
  const edm::View<TrackingParticle>& trackingParticles = *trackingParticlesHandle;

  edm::Handle<edm::View<CaloParticle>> caloParticlesHandle;
  iEvent.getByToken(caloParticles_, caloParticlesHandle);
  const edm::View<CaloParticle>& caloParticles = *caloParticlesHandle;

  //Matches reco tracks to sim tracks (TrackingParticle)
  edm::Handle<reco::RecoToSimCollection> recotosimCollection;
  iEvent.getByToken(tracks_recotosim_, recotosimCollection);
  const auto recotosim = *recotosimCollection;

  edm::Handle<reco::RecoToSimCollection> gsfrecotosimCollection;
  iEvent.getByToken(gsf_recotosim_, gsfrecotosimCollection);
  const auto gsfrecotosim = *gsfrecotosimCollection;

  edm::Handle<edm::View<reco::Track>> trackHandle;
  iEvent.getByToken(tracks_, trackHandle);
  const edm::View<reco::Track>& tracks = *trackHandle;

  edm::Handle<edm::View<reco::Track>> gsftrackHandle;
  iEvent.getByToken(gsftracks_, gsftrackHandle);
  const edm::View<reco::Track>& gsftracks = *gsftrackHandle;

  edm::Handle<edm::View<reco::GsfElectron>> gsfElectronHandle;
  iEvent.getByToken(gsfElectrons_, gsfElectronHandle);
  const edm::View<reco::GsfElectron>& gsfElectrons = *gsfElectronHandle;

  edm::Handle<std::vector<reco::GenParticle>> genParticlesHandle;
  iEvent.getByToken(genParticles_, genParticlesHandle);
  for (std::vector<reco::GenParticle>::const_iterator it_p = genParticlesHandle->begin();
       it_p != genParticlesHandle->end();
       ++it_p) {
    gen_eta_.push_back(it_p->eta());
    gen_phi_.push_back(it_p->phi());
    gen_pt_.push_back(it_p->pt());
    gen_px_.push_back(it_p->px());
    gen_py_.push_back(it_p->py());
    gen_pz_.push_back(it_p->pz());
    gen_energy_.push_back(it_p->energy());
    gen_charge_.push_back(it_p->charge());
    gen_pdgid_.push_back(it_p->pdgId());
    gen_status_.push_back(it_p->status());
    std::vector<int> daughters(it_p->daughterRefVector().size(), 0);
    for (unsigned j = 0; j < it_p->daughterRefVector().size(); ++j) {
      daughters[j] = static_cast<int>(it_p->daughterRefVector().at(j).key());
    }
    gen_daughters_.push_back(daughters);
    LogTrace("PFAnalysisNtuplizer") << "gp pt=" << it_p->pt() << " pid=" << it_p->pdgId() << " dau=" << daughters.size()
                                    << std::endl;
  }

  edm::Handle<edm::View<reco::GenJet>> genJetsHandle;
  iEvent.getByToken(genJets_, genJetsHandle);
  const edm::View<reco::GenJet>& genJets = *genJetsHandle;
  for (const auto& genjet : genJets) {
    genjet_pt_.push_back(genjet.pt());
    genjet_eta_.push_back(genjet.eta());
    genjet_phi_.push_back(genjet.phi());
    genjet_energy_.push_back(genjet.energy());
  }

  edm::Handle<edm::View<reco::GenMET>> genMETsHandle;
  iEvent.getByToken(genMETs_, genMETsHandle);
  const edm::View<reco::GenMET>& genMETs = *genMETsHandle;
  for (const auto& genmet : genMETs) {
    genmet_pt_.push_back(genmet.pt());
    genmet_phi_.push_back(genmet.phi());
  }

  edm::Handle<std::vector<reco::PFCandidate>> pfCandidatesHandle;
  iEvent.getByToken(pfCandidates_, pfCandidatesHandle);
  std::vector<reco::PFCandidate> pfCandidates = *pfCandidatesHandle;

  edm::Handle<std::vector<reco::PFBlock>> pfBlocksHandle;
  iEvent.getByToken(pfBlocks_, pfBlocksHandle);
  std::vector<reco::PFBlock> pfBlocks = *pfBlocksHandle;

  //Collect all clusters, tracks and superclusters
  const auto& all_elements_distances = processBlocks(pfBlocks);
  const auto& all_elements = all_elements_distances.first;
  const auto& all_distances = all_elements_distances.second;
  assert(!all_elements.empty());
  //assert(all_distances.size() > 0);
  for (const auto& d : all_distances) {
    element_distance_i_.push_back(get<0>(d));
    element_distance_j_.push_back(get<1>(d));
    element_distance_d_.push_back(get<2>(d));
  }

#ifdef EDM_ML_DEBUG
  LogTrace("PFAnalysisNtuplizer") << "RecoToSim";
  for (const auto& x : recotosim) {
    LogTrace("PFAnalysisNtuplizer") << "reco=" << x.key.key();
    for (const auto& tp : x.val) {
      LogTrace("PFAnalysisNtuplizer") << "tp=" << tp.first.key();
    }
  }

  LogTrace("PFAnalysisNtuplizer") << "GsfRecoToSim" << std::endl;
  for (const auto& x : gsfrecotosim) {
    LogTrace("PFAnalysisNtuplizer") << "reco=" << x.key.key();
    for (const auto& tp : x.val) {
      LogTrace("PFAnalysisNtuplizer") << "tp=" << tp.first.key();
    }
  }
#endif

  LogTrace("PFAnalysisNtuplizer") << "Gsf";
  for (unsigned long ntrack = 0; ntrack < gsftracks.size(); ntrack++) {
    edm::RefToBase<reco::Track> trackref(gsftrackHandle, ntrack);
    LogTrace("PFAnalysisNtuplizer") << "trackref=" << trackref.key();
    const auto vec_idx_in_all_elements = find_element_ref(all_elements, trackref);

    //track was not used by PF, we skip as well
    if (vec_idx_in_all_elements.empty()) {
      continue;
    }

    if (gsfrecotosim.find(trackref) != gsfrecotosim.end()) {
      const auto& tps = gsfrecotosim[trackref];
      for (const auto& tp : tps) {
        edm::Ref<std::vector<TrackingParticle>> tpr = tp.first;
        for (auto idx_in_all_elements : vec_idx_in_all_elements) {
          LogTrace("PFAnalysisNtuplizer") << "GSF assoc " << ntrack << " " << idx_in_all_elements << " " << tpr.key();
          trackingparticle_to_element.emplace_back(tpr.key(), idx_in_all_elements);
          trackingparticle_to_element_cmp.emplace_back(tp.second);
        }
      }
    }
  }

  LogTrace("PFAnalysisNtuplizer") << "Track";
  //We need to use the original reco::Track collection for track association
  for (unsigned long ntrack = 0; ntrack < tracks.size(); ntrack++) {
    edm::RefToBase<reco::Track> trackref(trackHandle, ntrack);
    LogTrace("PFAnalysisNtuplizer") << "trackref=" << trackref.key() << std::endl;
    const auto vec_idx_in_all_elements = find_element_ref(all_elements, trackref);

    //track was not used by PF, we skip as well
    if (vec_idx_in_all_elements.empty()) {
      continue;
    }

    if (recotosim.find(trackref) != recotosim.end()) {
      const auto& tps = recotosim[trackref];
      for (const auto& tp : tps) {
        edm::Ref<std::vector<TrackingParticle>> tpr = tp.first;
        for (auto idx_in_all_elements : vec_idx_in_all_elements) {
          LogTrace("PFAnalysisNtuplizer") << "track assoc " << ntrack << " " << idx_in_all_elements << " " << tpr.key();
          trackingparticle_to_element.emplace_back(tpr.key(), idx_in_all_elements);
          trackingparticle_to_element_cmp.emplace_back(tp.second);
        }
      }
    }
  }

  processTrackingParticles(trackingParticles, trackingParticlesHandle);

  map<pair<int, int>, float> caloparticle_to_pfcluster;
  map<pair<int, int>, float> simcluster_to_pfcluster;
  map<uint64_t, vector<pair<int, float>>> simhit_to_caloparticle;
  map<uint64_t, vector<pair<int, float>>> simhit_to_simcluster;
  int nsimcluster = 0;
  for (unsigned int ncaloparticle = 0; ncaloparticle < caloParticles.size(); ncaloparticle++) {
    const auto& cp = caloParticles.at(ncaloparticle);
    edm::RefToBase<CaloParticle> cpref(caloParticlesHandle, ncaloparticle);

    caloparticle_pt_.push_back(cp.p4().pt());
    caloparticle_eta_.push_back(cp.p4().eta());
    caloparticle_phi_.push_back(cp.p4().phi());
    caloparticle_energy_.push_back(cp.p4().energy());
    caloparticle_simenergy_.push_back(cp.simEnergy());

    caloparticle_ev_.push_back(cp.eventId().event());
    caloparticle_bx_.push_back(cp.eventId().bunchCrossing());
    caloparticle_pid_.push_back(cp.pdgId());
    caloparticle_charge_.push_back(cp.charge());

    LogTrace("PFAnalysisNtuplizer") << "cp=" << ncaloparticle << " pt=" << cp.p4().pt() << " typ=" << cp.pdgId();

    const auto& simtrack = cp.g4Tracks().at(0);

    int caloparticle_to_trackingparticle = -1;
    for (size_t itp = 0; itp < trackingParticles.size(); itp++) {
      const auto& simtrack2 = trackingParticles.at(itp).g4Tracks().at(0);
      //compare the two tracks, taking into account that both eventId and trackId need to be compared due to pileup
      if (simtrack.eventId() == simtrack2.eventId() && simtrack.trackId() == simtrack2.trackId()) {
        caloparticle_to_trackingparticle = itp;
        //we are satisfied with the first match, in practice there should not be more
        break;
      }
    }  //trackingParticles
    caloparticle_idx_trackingparticle_.push_back(caloparticle_to_trackingparticle);

    for (const auto& simcluster_ref : cp.simClusters()) {
      const auto& simcluster = *simcluster_ref;
      simcluster_eta_.push_back(simcluster.p4().eta());
      simcluster_phi_.push_back(simcluster.p4().phi());
      simcluster_pt_.push_back(simcluster.p4().pt());
      simcluster_energy_.push_back(simcluster.p4().energy());
      simcluster_pid_.push_back(simcluster.pdgId());
      simcluster_charge_.push_back(simcluster.charge());
      simcluster_bx_.push_back(simcluster.eventId().bunchCrossing());
      simcluster_ev_.push_back(simcluster.eventId().event());
      simcluster_idx_caloparticle_.push_back(ncaloparticle);
      LogTrace("PFAnalysisNtuplizer") << "  sc pt=" << simcluster.p4().pt() << " typ=" << simcluster.pdgId()
                                      << " gen=" << simcluster.genParticles().size()
                                      << " tid=" << simcluster.g4Tracks().at(0).trackId() << std::endl;

      int simcluster_to_trackingparticle = -1;
      for (size_t itp = 0; itp < trackingParticles.size(); itp++) {
        const auto& simtrack2 = trackingParticles.at(itp).g4Tracks().at(0);
        //compare the two tracks, taking into account that both eventId and trackId need to be compared due to pileup
        if (simcluster.g4Tracks().at(0).eventId() == simtrack2.eventId() &&
            simcluster.g4Tracks().at(0).trackId() == simtrack2.trackId()) {
          simcluster_to_trackingparticle = itp;
          //we are satisfied with the first match, in practice there should not be more
          break;
        }
      }  //trackingParticles
      simcluster_idx_trackingparticle_.push_back(simcluster_to_trackingparticle);

      for (const auto& hf : simcluster.hits_and_fractions()) {
        LogTrace("PFAnalysisNtuplizer") << "  cp=" << ncaloparticle << " sc=" << nsimcluster << " detid=" << hf.first
                                        << " " << hf.second;
        simhit_to_caloparticle[hf.first].push_back({ncaloparticle, hf.second});
        simhit_to_simcluster[hf.first].push_back({nsimcluster, hf.second});
      }
      caloparticle_to_simcluster.push_back({ncaloparticle, nsimcluster});
      nsimcluster++;
    }  //simclusters
  }  //caloParticles

  //fill pfcluster to rechit
  map<int, vector<pair<uint64_t, float>>> pfcluster_to_rechit;
  for (size_t ielem = 0; ielem < all_elements.size(); ielem++) {
    const auto& elem = all_elements[ielem];
    const auto& type = elem.orig.type();
    LogTrace("PFAnalysisNtuplizer") << "elem=" << ielem << " typ=" << type;
    if (type == reco::PFBlockElement::ECAL || type == reco::PFBlockElement::HCAL || type == reco::PFBlockElement::PS1 ||
        type == reco::PFBlockElement::PS2 || type == reco::PFBlockElement::HO || type == reco::PFBlockElement::HFHAD ||
        type == reco::PFBlockElement::HFEM) {
      const auto& clref = elem.orig.clusterRef();
      assert(clref.isNonnull());
      const auto& cluster = *clref;

      //all rechits and the energy fractions in this cluster
      const vector<reco::PFRecHitFraction>& rechit_fracs = cluster.recHitFractions();
      for (const auto& rh : rechit_fracs) {
        const reco::PFRecHit pfrh = *rh.recHitRef();
        LogTrace("PFAnalysisNtuplizer") << "  elem=" << ielem << " detid=" << pfrh.detId() << " " << pfrh.energy()
                                        << " " << rh.fraction();
        pfcluster_to_rechit[ielem].push_back({pfrh.detId(), pfrh.energy() * rh.fraction()});
      }  //rechit_fracs
    } else if (type == reco::PFBlockElement::SC) {
      const auto& clref = ((const reco::PFBlockElementSuperCluster*)&(elem.orig))->superClusterRef();
      assert(clref.isNonnull());
      const auto& cluster = *clref;

      //all rechits and the energy fractions in this cluster
      const auto& rechit_fracs = cluster.hitsAndFractions();
      for (const auto& rh : rechit_fracs) {
        LogTrace("PFAnalysisNtuplizer") << "  elem=" << ielem << " detid=" << rh.first.rawId() << " " << rh.second;
        pfcluster_to_rechit[ielem].push_back({rh.first.rawId(), rh.second});
      }  //rechit_fracs
    }
  }  //all_elements

  //fill elements
  for (size_t ielem = 0; ielem < all_elements.size(); ielem++) {
    const auto& elem = all_elements.at(ielem);
    const auto& orig = elem.orig;

    const auto& found = pfcluster_to_rechit.find(ielem);
    if (found != pfcluster_to_rechit.end()) {
      for (const auto& rechit_frac : (*found).second) {
        const auto& found_cp = simhit_to_caloparticle.find(rechit_frac.first);
        if (found_cp != simhit_to_caloparticle.end()) {
          for (const auto& simhit_frac : (*found_cp).second) {
            //(icalo, ielem) += rechit_energy*rechit_fraction*simhit_fraction
            const pair<size_t, size_t> key{simhit_frac.first, ielem};
            if (caloparticle_to_pfcluster.find(key) == caloparticle_to_pfcluster.end()) {
              caloparticle_to_pfcluster[key] = 0.0;
            }
            LogTrace("PFAnalysisNtuplizer") << "cp match " << key.first << " " << key.second << " "
                                            << rechit_frac.second << " " << simhit_frac.second;
            caloparticle_to_pfcluster[key] += rechit_frac.second * simhit_frac.second;
          }
        }

        const auto& found_sc = simhit_to_simcluster.find(rechit_frac.first);
        if (found_sc != simhit_to_simcluster.end()) {
          for (const auto& simhit_frac : (*found_sc).second) {
            const pair<size_t, size_t> key{simhit_frac.first, ielem};
            if (simcluster_to_pfcluster.find(key) == simcluster_to_pfcluster.end()) {
              simcluster_to_pfcluster[key] = 0.0;
            }
            LogTrace("PFAnalysisNtuplizer") << "sc match " << key.first << " " << key.second << " "
                                            << rechit_frac.second << " " << simhit_frac.second;
            simcluster_to_pfcluster[key] += rechit_frac.second * simhit_frac.second;
          }
        }
      }
    }

    const auto& props = reco::mlpf::getElementProperties(orig, gsfElectrons);

    element_pt_.push_back(props.pt);
    element_pterror_.push_back(props.pterror);
    element_px_.push_back(props.px);
    element_py_.push_back(props.py);
    element_pz_.push_back(props.pz);
    element_sigma_x_.push_back(props.sigma_x);
    element_sigma_y_.push_back(props.sigma_y);
    element_sigma_z_.push_back(props.sigma_z);
    element_deltap_.push_back(props.deltap);
    element_sigmadeltap_.push_back(props.sigmadeltap);
    element_eta_.push_back(props.eta);
    element_etaerror_.push_back(props.etaerror);
    element_phi_.push_back(props.phi);
    element_phierror_.push_back(props.phierror);
    element_energy_.push_back(props.energy);
    element_corr_energy_.push_back(props.corr_energy);
    element_corr_energy_err_.push_back(props.corr_energy_err);
    element_eta_ecal_.push_back(props.eta_ecal);
    element_phi_ecal_.push_back(props.phi_ecal);
    element_eta_hcal_.push_back(props.eta_hcal);
    element_phi_hcal_.push_back(props.phi_hcal);
    element_charge_.push_back(props.charge);
    element_type_.push_back(props.type);
    element_layer_.push_back(props.layer);
    element_depth_.push_back(props.depth);
    element_trajpoint_.push_back(props.trajpoint);
    element_muon_dt_hits_.push_back(props.muon_dt_hits);
    element_muon_csc_hits_.push_back(props.muon_csc_hits);
    element_muon_type_.push_back(props.muon_type);
    element_cluster_flags_.push_back(props.cluster_flags);
    element_gsf_electronseed_trkorecal_.push_back(props.gsf_electronseed_trkorecal);
    element_gsf_electronseed_dnn1_.push_back(props.gsf_electronseed_dnn1);
    element_gsf_electronseed_dnn2_.push_back(props.gsf_electronseed_dnn2);
    element_gsf_electronseed_dnn3_.push_back(props.gsf_electronseed_dnn3);
    element_gsf_electronseed_dnn4_.push_back(props.gsf_electronseed_dnn4);
    element_gsf_electronseed_dnn5_.push_back(props.gsf_electronseed_dnn5);
    element_num_hits_.push_back(props.num_hits);
    element_lambda_.push_back(props.lambda);
    element_lambdaerror_.push_back(props.lambdaerror);
    element_theta_.push_back(props.theta);
    element_thetaerror_.push_back(props.thetaerror);
    element_vx_.push_back(props.vx);
    element_vy_.push_back(props.vy);
    element_vz_.push_back(props.vz);
    element_time_.push_back(props.time);
    element_timeerror_.push_back(props.timeerror);
    element_etaerror1_.push_back(props.etaerror1);
    element_etaerror2_.push_back(props.etaerror2);
    element_etaerror3_.push_back(props.etaerror3);
    element_etaerror4_.push_back(props.etaerror4);
    element_phierror1_.push_back(props.phierror1);
    element_phierror2_.push_back(props.phierror2);
    element_phierror3_.push_back(props.phierror3);
    element_phierror4_.push_back(props.phierror4);
  }  //all_elements

  //fill caloparticle_to_element
  for (const auto& cp_to_pf : caloparticle_to_pfcluster) {
    const auto& cp_pf = cp_to_pf.first;
    const auto energy = cp_to_pf.second;
    caloparticle_to_element.push_back(cp_pf);
    caloparticle_to_element_cmp.push_back(energy);
    LogTrace("PFAnalysisNtuplizer") << "cp_to_elem=" << cp_pf.first << "," << cp_pf.second << " e=" << energy;
  }

  for (const auto& sc_to_pf : simcluster_to_pfcluster) {
    const auto& sc_pf = sc_to_pf.first;
    const auto energy = sc_to_pf.second;
    simcluster_to_element.push_back(sc_pf);
    simcluster_to_element_cmp.push_back(energy);
    LogTrace("PFAnalysisNtuplizer") << "sc_to_elem=" << sc_pf.first << "," << sc_pf.second << " e=" << energy;
  }

  //associate candidates to elements
  int icandidate = 0;
  for (const auto& cand : pfCandidates) {
    pfcandidate_eta_.push_back(cand.eta());
    pfcandidate_phi_.push_back(cand.phi());
    pfcandidate_pt_.push_back(cand.pt());
    pfcandidate_px_.push_back(cand.px());
    pfcandidate_py_.push_back(cand.py());
    pfcandidate_pz_.push_back(cand.pz());
    pfcandidate_energy_.push_back(cand.energy());
    pfcandidate_pdgid_.push_back(cand.pdgId());

    for (const auto& el : cand.elementsInBlocks()) {
      const auto idx_block = el.first.index();
      unsigned idx_element_in_block = el.second;

      int ielem = -1;
      for (const auto& elem_with_index : all_elements) {
        ielem += 1;
        if (elem_with_index.idx_block == idx_block && elem_with_index.idx_elem == idx_element_in_block) {
          break;
        }
      }
      assert(ielem != -1);
      element_to_candidate.push_back(make_pair(ielem, icandidate));
    }  //elements

    icandidate += 1;
  }  //pfCandidates

  ev_event_ = iEvent.id().event();
  ev_lumi_ = iEvent.id().luminosityBlock();
  ev_run_ = iEvent.id().run();

  t_->Fill();
}  //analyze

void PFAnalysis::processTrackingParticles(const edm::View<TrackingParticle>& trackingParticles,
                                          edm::Handle<edm::View<TrackingParticle>>& trackingParticlesHandle) {
  for (unsigned long ntrackingparticle = 0; ntrackingparticle < trackingParticles.size(); ntrackingparticle++) {
    const auto& tp = trackingParticles.at(ntrackingparticle);
    edm::RefToBase<TrackingParticle> tpref(trackingParticlesHandle, ntrackingparticle);

    math::XYZTLorentzVectorD vtx(0, 0, 0, 0);

    if (!tp.decayVertices().empty()) {
      vtx = tp.decayVertices().at(0)->position();
    }
    auto orig_vtx = tp.vertex();

    // fill branches
    trackingparticle_eta_.push_back(tp.p4().eta());
    trackingparticle_phi_.push_back(tp.p4().phi());
    trackingparticle_pt_.push_back(tp.p4().pt());
    trackingparticle_px_.push_back(tp.p4().px());
    trackingparticle_py_.push_back(tp.p4().py());
    trackingparticle_pz_.push_back(tp.p4().pz());
    trackingparticle_energy_.push_back(tp.p4().energy());
    trackingparticle_dvx_.push_back(vtx.x());
    trackingparticle_dvy_.push_back(vtx.y());
    trackingparticle_dvz_.push_back(vtx.z());
    trackingparticle_bx_.push_back(tp.eventId().bunchCrossing());
    trackingparticle_ev_.push_back(tp.eventId().event());

    trackingparticle_ovx_.push_back(orig_vtx.x());
    trackingparticle_ovy_.push_back(orig_vtx.y());
    trackingparticle_ovz_.push_back(orig_vtx.z());

    trackingparticle_pid_.push_back(tp.pdgId());
    trackingparticle_charge_.push_back(tp.charge());
    LogTrace("PFAnalysisNtuplizer") << "tp=" << ntrackingparticle << " pt=" << tp.p4().pt() << " typ=" << tp.pdgId()
                                    << " gen=" << tp.genParticles().size() << " tid=" << tp.g4Tracks().at(0).trackId()
                                    << std::endl;
  }
}

//https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix/27088560
int get_index_triu_vector(int i, int j, int n) {
  int k = (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1;
  return k;
}

pair<int, int> get_triu_vector_index(int k, int n) {
  int i = n - 2 - floor(sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5);
  int j = k + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2;
  return make_pair(i, j);
}

pair<vector<ElementWithIndex>, vector<tuple<int, int, float>>> PFAnalysis::processBlocks(
    const std::vector<reco::PFBlock>& pfBlocks) {
  vector<ElementWithIndex> ret;
  vector<tuple<int, int, float>> distances;

  //Collect all the elements
  int iblock = 0;
  for (const auto& block : pfBlocks) {
    int ielem = 0;
    const auto& linkdata = block.linkData();

    //create a list of global element indices with distances
    for (const auto& link : linkdata) {
      const auto vecidx = link.first;
      const auto dist = link.second.distance;
      const auto& ij = get_triu_vector_index(vecidx, block.elements().size());
      auto globalindex_i = ij.first + ret.size();
      auto globalindex_j = ij.second + ret.size();
      distances.push_back(make_tuple(globalindex_i, globalindex_j, dist));
    }

    for (const auto& elem : block.elements()) {
      ElementWithIndex elem_index(elem, iblock, ielem);
      ret.push_back(elem_index);
      ielem += 1;
    }  //elements
    iblock += 1;
  }  //blocks
  return make_pair(ret, distances);

}  //processBlocks

void PFAnalysis::beginRun(edm::Run const& iEvent, edm::EventSetup const& es) {}

void PFAnalysis::endRun(edm::Run const& iEvent, edm::EventSetup const&) {}

void PFAnalysis::beginJob() { ; }

void PFAnalysis::endJob() {}

void PFAnalysis::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(PFAnalysis);

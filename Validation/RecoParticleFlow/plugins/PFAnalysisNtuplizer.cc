// Based on RecoNtuple/HGCalAnalysis with modifications for PF
//
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Associations/interface/TrackAssociation.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "Math/Transform3D.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/VolumeGeometry/interface/MagVolumeOutsideValidity.h"
#include "RecoParticleFlow/PFProducer/interface/MLPFModel.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "TH1F.h"
#include "TVector2.h"
#include "TTree.h"

#include <map>
#include <set>
#include <string>
#include <vector>
#include <set>

using namespace std;

class ElementWithIndex {
public:
  const reco::PFBlockElement& orig;
  size_t idx_block;
  size_t idx_elem;
  ElementWithIndex(const reco::PFBlockElement& _orig, size_t _idx_block, size_t _idx_elem)
      : orig(_orig), idx_block(_idx_block), idx_elem(_idx_elem){};
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
    }
  }
  return ret;
}

double detid_compare(const map<uint64_t, double>& rechits, const map<uint64_t, double>& simhits) {
  double ret = 0.0;

  for (const auto& rh : rechits) {
    for (const auto& sh : simhits) {
      if (rh.first == sh.first) {
        //rechit energy times simhit fraction
        ret += rh.second * sh.second;
        break;
      }
    }
  }
  return ret;
}

class PFAnalysis : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  typedef ROOT::Math::Transform3D::Point Point;

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

  void associateClusterToSimCluster(const vector<ElementWithIndex>& all_elements);

  void clearVariables();

  GlobalPoint getHitPosition(const DetId& id);
  // ----------member data ---------------------------

  edm::EDGetTokenT<std::vector<reco::GenParticle>> genParticles_;
  edm::EDGetTokenT<edm::View<TrackingParticle>> trackingParticles_;
  edm::EDGetTokenT<edm::View<CaloParticle>> caloParticles_;
  edm::EDGetTokenT<edm::View<reco::Track>> tracks_;
  edm::EDGetTokenT<std::vector<reco::PFBlock>> pfBlocks_;
  edm::EDGetTokenT<std::vector<reco::PFCandidate>> pfCandidates_;
  edm::EDGetTokenT<reco::RecoToSimCollection> tracks_recotosim_;

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
  vector<int> simcluster_idx_trackingparticle_;
  vector<int> simcluster_nhits_;
  vector<std::map<uint64_t, double>> simcluster_detids_;

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

  vector<float> simtrack_x_;
  vector<float> simtrack_y_;
  vector<float> simtrack_z_;
  vector<int> simtrack_idx_simcluster_;
  vector<int> simtrack_pid_;

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

  vector<float> element_pt_;
  vector<float> element_px_;
  vector<float> element_py_;
  vector<float> element_pz_;
  vector<float> element_deltap_;
  vector<float> element_sigmadeltap_;
  vector<float> element_eta_;
  vector<float> element_phi_;
  vector<float> element_energy_;
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
  vector<pair<int, int>> simcluster_to_element;
  vector<float> simcluster_to_element_cmp;
  vector<pair<int, int>> element_to_candidate;

  // and also the magnetic field
  MagneticField const* aField_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> topologyToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> hcalDDDrecToken_;

  CaloGeometry* geom;
  HcalTopology* hcal_topo;
  const HcalDDDRecConstants* hcons;

  bool saveHits;
};

PFAnalysis::PFAnalysis() { ; }

PFAnalysis::PFAnalysis(const edm::ParameterSet& iConfig) {
  tracks_recotosim_ = consumes<reco::RecoToSimCollection>(edm::InputTag("trackingParticleRecoTrackAsssociation"));
  trackingParticles_ = consumes<edm::View<TrackingParticle>>(edm::InputTag("mix", "MergedTrackTruth"));
  caloParticles_ = consumes<edm::View<CaloParticle>>(edm::InputTag("mix", "MergedCaloTruth"));
  genParticles_ = consumes<std::vector<reco::GenParticle>>(edm::InputTag("genParticles"));
  pfBlocks_ = consumes<std::vector<reco::PFBlock>>(edm::InputTag("particleFlowBlock"));
  pfCandidates_ = consumes<std::vector<reco::PFCandidate>>(edm::InputTag("particleFlow"));
  tracks_ = consumes<edm::View<reco::Track>>(edm::InputTag("generalTracks"));
  saveHits = iConfig.getUntrackedParameter<bool>("saveHits", false);

  geometryToken_ = esConsumes<CaloGeometry, CaloGeometryRecord>(edm::ESInputTag{});
  topologyToken_ = esConsumes<HcalTopology, HcalRecNumberingRecord>(edm::ESInputTag{});
  magFieldToken_ = esConsumes<edm::Transition::BeginRun>();
  hcalDDDrecToken_ = esConsumes<edm::Transition::BeginRun>();

  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  fs->make<TH1F>("total", "total", 100, 0, 5.);

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

  t_->Branch("simcluster_eta", &simcluster_eta_);
  t_->Branch("simcluster_phi", &simcluster_phi_);
  t_->Branch("simcluster_pt", &simcluster_pt_);
  t_->Branch("simcluster_px", &simcluster_px_);
  t_->Branch("simcluster_py", &simcluster_py_);
  t_->Branch("simcluster_pz", &simcluster_pz_);
  t_->Branch("simcluster_energy", &simcluster_energy_);
  t_->Branch("simcluster_bx", &simcluster_bx_);
  t_->Branch("simcluster_ev", &simcluster_ev_);
  t_->Branch("simcluster_pid", &simcluster_pid_);
  t_->Branch("simcluster_idx_trackingparticle", &simcluster_idx_trackingparticle_);
  t_->Branch("simcluster_nhits", &simcluster_nhits_);

  if (saveHits) {
    t_->Branch("simhit_frac", &simhit_frac_);
    t_->Branch("simhit_x", &simhit_x_);
    t_->Branch("simhit_y", &simhit_y_);
    t_->Branch("simhit_z", &simhit_z_);
    t_->Branch("simhit_det", &simhit_det_);
    t_->Branch("simhit_subdet", &simhit_subdet_);
    t_->Branch("simhit_eta", &simhit_eta_);
    t_->Branch("simhit_phi", &simhit_phi_);
    t_->Branch("simhit_idx_simcluster", &simhit_idx_simcluster_);
    t_->Branch("simhit_detid", &simhit_detid_);

    t_->Branch("rechit_e", &rechit_e_);
    t_->Branch("rechit_x", &rechit_x_);
    t_->Branch("rechit_y", &rechit_y_);
    t_->Branch("rechit_z", &rechit_z_);
    t_->Branch("rechit_det", &rechit_det_);
    t_->Branch("rechit_subdet", &rechit_subdet_);
    t_->Branch("rechit_eta", &rechit_eta_);
    t_->Branch("rechit_phi", &rechit_phi_);
    t_->Branch("rechit_idx_element", &rechit_idx_element_);
    t_->Branch("rechit_detid", &rechit_detid_);
  }

  t_->Branch("simtrack_x", &simtrack_x_);
  t_->Branch("simtrack_y", &simtrack_y_);
  t_->Branch("simtrack_z", &simtrack_z_);
  t_->Branch("simtrack_idx_simcluster_", &simtrack_idx_simcluster_);
  t_->Branch("simtrack_pid", &simtrack_pid_);

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

  //PF Elements
  t_->Branch("element_pt", &element_pt_);
  t_->Branch("element_px", &element_px_);
  t_->Branch("element_py", &element_py_);
  t_->Branch("element_pz", &element_pz_);
  t_->Branch("element_deltap", &element_deltap_);
  t_->Branch("element_sigmadeltap", &element_sigmadeltap_);
  t_->Branch("element_eta", &element_eta_);
  t_->Branch("element_phi", &element_phi_);
  t_->Branch("element_energy", &element_energy_);
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
  t_->Branch("simcluster_to_element", &simcluster_to_element);
  t_->Branch("simcluster_to_element_cmp", &simcluster_to_element_cmp);
  t_->Branch("element_to_candidate", &element_to_candidate);
}  // constructor

PFAnalysis::~PFAnalysis() {}

void PFAnalysis::clearVariables() {
  ev_run_ = 0;
  ev_lumi_ = 0;
  ev_event_ = 0;

  trackingparticle_to_element.clear();
  simcluster_to_element.clear();
  simcluster_to_element_cmp.clear();
  element_to_candidate.clear();

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

  simcluster_eta_.clear();
  simcluster_phi_.clear();
  simcluster_pt_.clear();
  simcluster_energy_.clear();
  simcluster_pid_.clear();
  simcluster_detids_.clear();
  simcluster_bx_.clear();
  simcluster_ev_.clear();
  simcluster_px_.clear();
  simcluster_py_.clear();
  simcluster_pz_.clear();
  simcluster_idx_trackingparticle_.clear();
  simcluster_nhits_.clear();

  if (saveHits) {
    simhit_frac_.clear();
    simhit_x_.clear();
    simhit_y_.clear();
    simhit_z_.clear();
    simhit_det_.clear();
    simhit_subdet_.clear();
    simhit_eta_.clear();
    simhit_phi_.clear();
    simhit_idx_simcluster_.clear();
    simhit_detid_.clear();

    rechit_e_.clear();
    rechit_x_.clear();
    rechit_y_.clear();
    rechit_z_.clear();
    rechit_det_.clear();
    rechit_subdet_.clear();
    rechit_eta_.clear();
    rechit_phi_.clear();
    rechit_idx_element_.clear();
    rechit_detid_.clear();
  }

  simtrack_x_.clear();
  simtrack_y_.clear();
  simtrack_z_.clear();
  simtrack_idx_simcluster_.clear();
  simtrack_pid_.clear();

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

  element_pt_.clear();
  element_px_.clear();
  element_py_.clear();
  element_pz_.clear();
  element_deltap_.clear();
  element_sigmadeltap_.clear();
  element_eta_.clear();
  element_phi_.clear();
  element_energy_.clear();
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

GlobalPoint PFAnalysis::getHitPosition(const DetId& id) {
  GlobalPoint ret;

  bool present = false;
  if (((id.det() == DetId::Ecal &&
        (id.subdetId() == EcalBarrel || id.subdetId() == EcalEndcap || id.subdetId() == EcalPreshower)) ||
       (id.det() == DetId::Hcal && (id.subdetId() == HcalBarrel || id.subdetId() == HcalEndcap ||
                                    id.subdetId() == HcalForward || id.subdetId() == HcalOuter)))) {
    const CaloSubdetectorGeometry* geom_sd(geom->getSubdetectorGeometry(id.det(), id.subdetId()));
    present = geom_sd->present(id);
    if (present) {
      const auto& cell = geom_sd->getGeometry(id);
      ret = GlobalPoint(cell->getPosition());
    }
  }
  return ret;
}

void PFAnalysis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  clearVariables();

  auto& pG = iSetup.getData(geometryToken_);
  geom = (CaloGeometry*)&pG;
  auto& pT = iSetup.getData(topologyToken_);
  hcal_topo = (HcalTopology*)&pT;

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

  edm::Handle<edm::View<reco::Track>> trackHandle;
  iEvent.getByToken(tracks_, trackHandle);
  const edm::View<reco::Track>& tracks = *trackHandle;

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

  //We need to use the original reco::Track collection for track association
  for (unsigned long ntrack = 0; ntrack < tracks.size(); ntrack++) {
    edm::RefToBase<reco::Track> trackref(trackHandle, ntrack);
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
          trackingparticle_to_element.emplace_back(tpr.key(), idx_in_all_elements);
        }
      }
    }
  }

  processTrackingParticles(trackingParticles, trackingParticlesHandle);

  int idx_simcluster = 0;
  //Fill genparticles from calorimeter hits
  for (unsigned long ncaloparticle = 0; ncaloparticle < caloParticles.size(); ncaloparticle++) {
    const auto& cp = caloParticles.at(ncaloparticle);
    edm::RefToBase<CaloParticle> cpref(caloParticlesHandle, ncaloparticle);

    int nhits = 0;
    for (const auto& simcluster : cp.simClusters()) {
      //create a map of detId->energy of all the rechits in all the clusters of this SimCluster
      map<uint64_t, double> detid_energy;

      simcluster_nhits_.push_back(nhits);
      simcluster_eta_.push_back(simcluster->p4().eta());
      simcluster_phi_.push_back(simcluster->p4().phi());
      simcluster_pt_.push_back(simcluster->p4().pt());
      simcluster_energy_.push_back(simcluster->energy());
      simcluster_pid_.push_back(simcluster->pdgId());
      simcluster_bx_.push_back(simcluster->eventId().bunchCrossing());
      simcluster_ev_.push_back(simcluster->eventId().event());

      simcluster_px_.push_back(simcluster->p4().x());
      simcluster_py_.push_back(simcluster->p4().y());
      simcluster_pz_.push_back(simcluster->p4().z());

      for (const auto& hf : simcluster->hits_and_fractions()) {
        DetId id(hf.first);

        if (id.det() == DetId::Hcal || id.det() == DetId::Ecal) {
          const auto& pos = getHitPosition(id);
          nhits += 1;

          const float x = pos.x();
          const float y = pos.y();
          const float z = pos.z();
          const float eta = pos.eta();
          const float phi = pos.phi();

          simhit_frac_.push_back(hf.second);
          simhit_x_.push_back(x);
          simhit_y_.push_back(y);
          simhit_z_.push_back(z);
          simhit_det_.push_back(id.det());
          simhit_subdet_.push_back(id.subdetId());
          simhit_eta_.push_back(eta);
          simhit_phi_.push_back(phi);
          simhit_idx_simcluster_.push_back(idx_simcluster);
          simhit_detid_.push_back(id.rawId());
          detid_energy[id.rawId()] += hf.second;
        }
      }

      int simcluster_to_trackingparticle = -1;
      for (const auto& simtrack : simcluster->g4Tracks()) {
        simtrack_x_.push_back(simtrack.trackerSurfacePosition().x());
        simtrack_y_.push_back(simtrack.trackerSurfacePosition().y());
        simtrack_z_.push_back(simtrack.trackerSurfacePosition().z());
        simtrack_idx_simcluster_.push_back(idx_simcluster);
        simtrack_pid_.push_back(simtrack.type());

        for (unsigned int itp = 0; itp < trackingParticles.size(); itp++) {
          const auto& simtrack2 = trackingParticles.at(itp).g4Tracks().at(0);
          //compare the two tracks, taking into account that both eventId and trackId need to be compared due to pileup
          if (simtrack.eventId() == simtrack2.eventId() && simtrack.trackId() == simtrack2.trackId()) {
            simcluster_to_trackingparticle = itp;
            //we are satisfied with the first match, in practice there should not be more
            break;
          }
        }  //trackingParticles
      }    //simcluster tracks

      simcluster_detids_.push_back(detid_energy);
      simcluster_idx_trackingparticle_.push_back(simcluster_to_trackingparticle);

      idx_simcluster += 1;
    }  //simclusters
  }    //caloParticles

  associateClusterToSimCluster(all_elements);

  //fill elements
  for (unsigned int ielem = 0; ielem < all_elements.size(); ielem++) {
    const auto& elem = all_elements.at(ielem);
    const auto& orig = elem.orig;
    reco::PFBlockElement::Type type = orig.type();

    float pt = 0.0;
    float deltap = 0.0;
    float sigmadeltap = 0.0;
    float px = 0.0;
    float py = 0.0;
    float pz = 0.0;
    float eta = 0.0;
    float phi = 0.0;
    float energy = 0.0;
    float trajpoint = 0.0;
    float eta_ecal = 0.0;
    float phi_ecal = 0.0;
    float eta_hcal = 0.0;
    float phi_hcal = 0.0;
    int charge = 0;
    int layer = 0;
    float depth = 0;
    float muon_dt_hits = 0.0;
    float muon_csc_hits = 0.0;

    if (type == reco::PFBlockElement::TRACK) {
      const auto& matched_pftrack = orig.trackRefPF();
      if (matched_pftrack.isNonnull()) {
        const auto& atECAL = matched_pftrack->extrapolatedPoint(reco::PFTrajectoryPoint::ECALShowerMax);
        const auto& atHCAL = matched_pftrack->extrapolatedPoint(reco::PFTrajectoryPoint::HCALEntrance);
        if (atHCAL.isValid()) {
          eta_hcal = atHCAL.positionREP().eta();
          phi_hcal = atHCAL.positionREP().phi();
        }
        if (atECAL.isValid()) {
          eta_ecal = atECAL.positionREP().eta();
          phi_ecal = atECAL.positionREP().phi();
        }
      }
      const auto& ref = ((const reco::PFBlockElementTrack*)&orig)->trackRef();
      pt = ref->pt();
      px = ref->px();
      py = ref->py();
      pz = ref->pz();
      eta = ref->eta();
      phi = ref->phi();
      energy = ref->p();
      charge = ref->charge();

      reco::MuonRef muonRef = orig.muonRef();
      if (muonRef.isNonnull()) {
        reco::TrackRef standAloneMu = muonRef->standAloneMuon();
        if (standAloneMu.isNonnull()) {
          muon_dt_hits = standAloneMu->hitPattern().numberOfValidMuonDTHits();
          muon_csc_hits = standAloneMu->hitPattern().numberOfValidMuonCSCHits();
        }
      }

    } else if (type == reco::PFBlockElement::BREM) {
      const auto* orig2 = (const reco::PFBlockElementBrem*)&orig;
      const auto& ref = orig2->GsftrackRef();
      if (ref.isNonnull()) {
        deltap = orig2->DeltaP();
        sigmadeltap = orig2->SigmaDeltaP();
        pt = ref->pt();
        px = ref->px();
        py = ref->py();
        pz = ref->pz();
        eta = ref->eta();
        phi = ref->phi();
        energy = ref->p();
        trajpoint = orig2->indTrajPoint();
        charge = ref->charge();
      }
    } else if (type == reco::PFBlockElement::GSF) {
      //requires to keep GsfPFRecTracks
      const auto* orig2 = (const reco::PFBlockElementGsfTrack*)&orig;
      const auto& vec = orig2->Pin();
      pt = vec.pt();
      px = vec.px();
      py = vec.py();
      pz = vec.pz();
      eta = vec.eta();
      phi = vec.phi();
      energy = vec.energy();
      if (!orig2->GsftrackRefPF().isNull()) {
        charge = orig2->GsftrackRefPF()->charge();
      }
    } else if (type == reco::PFBlockElement::ECAL || type == reco::PFBlockElement::PS1 ||
               type == reco::PFBlockElement::PS2 || type == reco::PFBlockElement::HCAL ||
               type == reco::PFBlockElement::HO || type == reco::PFBlockElement::HFHAD ||
               type == reco::PFBlockElement::HFEM) {
      const auto& ref = ((const reco::PFBlockElementCluster*)&orig)->clusterRef();
      if (ref.isNonnull()) {
        eta = ref->eta();
        phi = ref->phi();
        px = ref->position().x();
        py = ref->position().y();
        pz = ref->position().z();
        energy = ref->energy();
        layer = ref->layer();
        depth = ref->depth();
      }
    } else if (type == reco::PFBlockElement::SC) {
      const auto& clref = ((const reco::PFBlockElementSuperCluster*)&orig)->superClusterRef();
      if (clref.isNonnull()) {
        eta = clref->eta();
        phi = clref->phi();
        px = clref->position().x();
        py = clref->position().y();
        pz = clref->position().z();
        energy = clref->energy();
      }
    }
    vector<int> tps;
    for (const auto& t : trackingparticle_to_element) {
      if (t.second == (int)ielem) {
        tps.push_back(t.first);
      }
    }
    vector<int> scs;
    for (const auto& t : simcluster_to_element) {
      if (t.second == (int)ielem) {
        scs.push_back(t.first);
      }
    }

    element_pt_.push_back(pt);
    element_px_.push_back(px);
    element_py_.push_back(py);
    element_pz_.push_back(pz);
    element_deltap_.push_back(deltap);
    element_sigmadeltap_.push_back(sigmadeltap);
    element_eta_.push_back(eta);
    element_phi_.push_back(phi);
    element_energy_.push_back(energy);
    element_eta_ecal_.push_back(eta_ecal);
    element_phi_ecal_.push_back(phi_ecal);
    element_eta_hcal_.push_back(eta_hcal);
    element_phi_hcal_.push_back(phi_hcal);
    element_charge_.push_back(charge);
    element_type_.push_back(type);
    element_layer_.push_back(layer);
    element_depth_.push_back(depth);
    element_trajpoint_.push_back(trajpoint);
    element_muon_dt_hits_.push_back(muon_dt_hits);
    element_muon_csc_hits_.push_back(muon_csc_hits);
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

void PFAnalysis::associateClusterToSimCluster(const vector<ElementWithIndex>& all_elements) {
  vector<map<uint64_t, double>> detids_elements;
  map<uint64_t, double> rechits_energy_all;

  int idx_element = 0;
  for (const auto& elem : all_elements) {
    map<uint64_t, double> detids;
    const auto& type = elem.orig.type();

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
        if (detids.find(pfrh.detId()) != detids.end()) {
          continue;
        }
        detids[pfrh.detId()] += pfrh.energy() * rh.fraction();
        const auto id = DetId(pfrh.detId());
        float x = 0;
        float y = 0;
        float z = 0;
        float eta = 0;
        float phi = 0;

        const auto& pos = getHitPosition(id);
        x = pos.x();
        y = pos.y();
        z = pos.z();
        eta = pos.eta();
        phi = pos.phi();

        rechit_x_.push_back(x);
        rechit_y_.push_back(y);
        rechit_z_.push_back(z);
        rechit_det_.push_back(id.det());
        rechit_subdet_.push_back(id.subdetId());
        rechit_eta_.push_back(eta);
        rechit_phi_.push_back(phi);
        rechit_e_.push_back(pfrh.energy() * rh.fraction());
        rechit_idx_element_.push_back(idx_element);
        rechit_detid_.push_back(id.rawId());
        rechits_energy_all[id.rawId()] += pfrh.energy() * rh.fraction();
      }  //rechit_fracs
    } else if (type == reco::PFBlockElement::SC) {
      const auto& clref = ((const reco::PFBlockElementSuperCluster*)&(elem.orig))->superClusterRef();
      assert(clref.isNonnull());
      const auto& cluster = *clref;

      //all rechits and the energy fractions in this cluster
      const auto& rechit_fracs = cluster.hitsAndFractions();
      for (const auto& rh : rechit_fracs) {
        if (detids.find(rh.first.rawId()) != detids.end()) {
          continue;
        }
        detids[rh.first.rawId()] += cluster.energy() * rh.second;
        const auto id = rh.first;
        float x = 0;
        float y = 0;
        float z = 0;
        float eta = 0;
        float phi = 0;

        const auto& pos = getHitPosition(id);
        x = pos.x();
        y = pos.y();
        z = pos.z();
        eta = pos.eta();
        phi = pos.phi();

        rechit_x_.push_back(x);
        rechit_y_.push_back(y);
        rechit_z_.push_back(z);
        rechit_det_.push_back(id.det());
        rechit_subdet_.push_back(id.subdetId());
        rechit_eta_.push_back(eta);
        rechit_phi_.push_back(phi);
        rechit_e_.push_back(rh.second);
        rechit_idx_element_.push_back(idx_element);
        rechit_detid_.push_back(id.rawId());
        rechits_energy_all[id.rawId()] += cluster.energy() * rh.second;
      }  //rechit_fracs
    }
    detids_elements.push_back(detids);
    idx_element += 1;
  }  //all_elements

  //associate elements (reco clusters) to simclusters
  int ielement = 0;
  for (const auto& detids : detids_elements) {
    int isimcluster = 0;
    if (!detids.empty()) {
      double sum_e_tot = 0.0;
      for (const auto& c : detids) {
        sum_e_tot += c.second;
      }

      for (const auto& simcluster_detids : simcluster_detids_) {
        double sum_e_tot_sc = 0.0;
        for (const auto& c : simcluster_detids) {
          sum_e_tot_sc += c.second;
        }

        //get the energy of the simcluster hits that matches detids of the rechits
        double cmp = detid_compare(detids, simcluster_detids);
        if (cmp > 0) {
          simcluster_to_element.push_back(make_pair(isimcluster, ielement));
          simcluster_to_element_cmp.push_back((float)cmp);
        }
        isimcluster += 1;
      }
    }  //element had rechits
    ielement += 1;
  }  //rechit clusters
}

void PFAnalysis::beginRun(edm::Run const& iEvent, edm::EventSetup const& es) {
  hcons = &es.getData(hcalDDDrecToken_);
  aField_ = &es.getData(magFieldToken_);
}

void PFAnalysis::endRun(edm::Run const& iEvent, edm::EventSetup const&) {}

void PFAnalysis::beginJob() { ; }

void PFAnalysis::endJob() {}

void PFAnalysis::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(PFAnalysis);

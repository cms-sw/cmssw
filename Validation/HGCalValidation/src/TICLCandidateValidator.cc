#include <numeric>
#include <iomanip>
#include <sstream>

#include "Validation/HGCalValidation/interface/TICLCandidateValidator.h"
#include "DataFormats/HGCalReco/interface/Common.h"

TICLCandidateValidator::TICLCandidateValidator(edm::EDGetTokenT<std::vector<TICLCandidate>> ticlCandidates,
                                               edm::EDGetTokenT<std::vector<TICLCandidate>> simTICLCandidatesToken,
                                               edm::EDGetTokenT<std::vector<reco::Track>> recoTracksToken,
                                               edm::EDGetTokenT<std::vector<ticl::Trackster>> trackstersToken,
                                               edm::EDGetTokenT<ticl::TracksterToTracksterMap> associatorMapRtSToken,
                                               edm::EDGetTokenT<ticl::TracksterToTracksterMap> associatorMapStRToken,
                                               bool isTICLv5)
    : TICLCandidatesToken_(ticlCandidates),
      simTICLCandidatesToken_(simTICLCandidatesToken),
      recoTracksToken_(recoTracksToken),
      trackstersToken_(trackstersToken),
      associatorMapRtSToken_(associatorMapRtSToken),
      associatorMapStRToken_(associatorMapStRToken),
      isTICLv5_(isTICLv5) {}

TICLCandidateValidator::~TICLCandidateValidator() {}

void TICLCandidateValidator::bookCandidatesHistos(DQMStore::IBooker& ibook,
                                                  Histograms& histograms,
                                                  std::string baseDir) const {
  // book CAND histos
  histograms.h_tracksters_in_candidate =
      ibook.book1D("N of tracksters in candidate", "N of tracksters in candidate", 100, 0, 99);
  histograms.h_candidate_raw_energy =
      ibook.book1D("Candidates raw energy", "Candidates raw energy;E (GeV)", 250, 0, 250);
  histograms.h_candidate_regressed_energy =
      ibook.book1D("Candidates regressed energy", "Candidates regressed energy;E (GeV)", 250, 0, 250);
  histograms.h_candidate_pT = ibook.book1D("Candidates pT", "Candidates pT;p_{T}", 250, 0, 250);
  histograms.h_candidate_charge = ibook.book1D("Candidates charge", "Candidates charge;Charge", 3, -1.5, 1.5);
  histograms.h_candidate_pdgId = ibook.book1D("Candidates PDG Id", "Candidates PDG ID", 100, -220, 220);
  histograms.h_candidate_partType = ibook.book1D("Candidates type", "Candidates type", 9, -0.5, 8.5);

  // neutral: photon, pion, hadron
  const std::vector<std::string> neutrals{"photons", "neutral_pions", "neutral_hadrons"};
  for (long unsigned int i = 0; i < neutrals.size(); i++) {
    ibook.setCurrentFolder(baseDir + "/" + neutrals[i]);

    histograms.h_neut_tracksters_in_candidate.push_back(ibook.book1D("N of tracksters in candidate for " + neutrals[i],
                                                                     "N of tracksters in candidate for " + neutrals[i],
                                                                     100,
                                                                     0,
                                                                     99));
    histograms.h_neut_candidate_regressed_energy.push_back(ibook.book1D(
        neutrals[i] + "candidates regressed energy", neutrals[i] + " candidates regressed energy;E (GeV)", 250, 0, 250));
    histograms.h_neut_candidate_charge.push_back(
        ibook.book1D(neutrals[i] + " candidates charge", neutrals[i] + " candidates charge;Charge", 3, -1.5, 1.5));
    histograms.h_neut_candidate_pdgId.push_back(
        ibook.book1D(neutrals[i] + " candidates PDG Id", neutrals[i] + " candidates PDG ID", 100, -220, 220));
    histograms.h_neut_candidate_partType.push_back(
        ibook.book1D(neutrals[i] + " candidates type", neutrals[i] + " candidates type", 9, -0.5, 8.5));

    histograms.h_den_fake_neut_energy_candidate.push_back(
        ibook.book1D("den_fake_cand_vs_energy_" + neutrals[i], neutrals[i] + " candidates energy;E (GeV)", 50, 0, 250));
    histograms.h_num_fake_neut_energy_candidate_pdgId.push_back(ibook.book1D(
        "num_fake_pid_cand_vs_energy_" + neutrals[i], neutrals[i] + " PID fake vs energy;E (GeV)", 50, 0, 250));
    histograms.h_num_fake_neut_energy_candidate_energy.push_back(
        ibook.book1D("num_fake_energy_cand_vs_energy_" + neutrals[i],
                     neutrals[i] + " PID and energy fake vs energy;E (GeV)",
                     50,
                     0,
                     250));
    histograms.h_den_fake_neut_pt_candidate.push_back(
        ibook.book1D("den_fake_cand_vs_pt_" + neutrals[i], neutrals[i] + " candidates pT;p_{T} (GeV)", 50, 0, 250));
    histograms.h_num_fake_neut_pt_candidate_pdgId.push_back(ibook.book1D(
        "num_fake_pid_cand_vs_pt_" + neutrals[i], neutrals[i] + " PID fake vs pT;p_{T} (GeV)", 50, 0, 250));
    histograms.h_num_fake_neut_pt_candidate_energy.push_back(
        ibook.book1D("num_fake_energy_cand_vs_pt_" + neutrals[i],
                     neutrals[i] + " PID and energy fake vs pT;p_{T} (GeV)",
                     50,
                     0,
                     250));
    histograms.h_den_fake_neut_eta_candidate.push_back(
        ibook.book1D("den_fake_cand_vs_eta_" + neutrals[i], neutrals[i] + " candidates eta;#eta (GeV)", 50, -3, 3));
    histograms.h_num_fake_neut_eta_candidate_pdgId.push_back(ibook.book1D(
        "num_fake_pid_cand_vs_eta_" + neutrals[i], neutrals[i] + " PID fake vs eta;#eta (GeV)", 50, -3, 3));
    histograms.h_num_fake_neut_eta_candidate_energy.push_back(
        ibook.book1D("num_fake_energy_cand_vs_eta_" + neutrals[i],
                     neutrals[i] + " PID and energy fake vs eta;#eta (GeV)",
                     50,
                     -3,
                     3));
    histograms.h_den_fake_neut_phi_candidate.push_back(ibook.book1D(
        "den_fake_cand_vs_phi_" + neutrals[i], neutrals[i] + " candidates phi;#phi (GeV)", 50, -3.14159, 3.14159));
    histograms.h_num_fake_neut_phi_candidate_pdgId.push_back(ibook.book1D(
        "num_fake_pid_cand_vs_phi_" + neutrals[i], neutrals[i] + " PID fake vs phi;#phi (GeV)", 50, -3.14159, 3.14159));
    histograms.h_num_fake_neut_phi_candidate_energy.push_back(
        ibook.book1D("num_fake_energy_cand_vs_phi_" + neutrals[i],
                     neutrals[i] + " PID and energy fake vs phi;#phi (GeV)",
                     50,
                     -3.14159,
                     3.14159));

    histograms.h_den_neut_energy_candidate.push_back(
        ibook.book1D("den_cand_vs_energy_" + neutrals[i], neutrals[i] + " simCandidates energy;E (GeV)", 50, 0, 250));
    histograms.h_num_neut_energy_candidate_pdgId.push_back(
        ibook.book1D("num_pid_cand_vs_energy_" + neutrals[i],
                     neutrals[i] + " track and PID efficiency vs energy;E (GeV)",
                     50,
                     0,
                     250));
    histograms.h_num_neut_energy_candidate_energy.push_back(
        ibook.book1D("num_energy_cand_vs_energy_" + neutrals[i],
                     neutrals[i] + " track, PID and energy efficiency vs energy;E (GeV)",
                     50,
                     0,
                     250));
    histograms.h_den_neut_pt_candidate.push_back(
        ibook.book1D("den_cand_vs_pt_" + neutrals[i], neutrals[i] + " simCandidates pT;p_{T} (GeV)", 50, 0, 250));
    histograms.h_num_neut_pt_candidate_pdgId.push_back(ibook.book1D(
        "num_pid_cand_vs_pt_" + neutrals[i], neutrals[i] + " track and PID efficiency vs pT;p_{T} (GeV)", 50, 0, 250));
    histograms.h_num_neut_pt_candidate_energy.push_back(
        ibook.book1D("num_energy_cand_vs_pt_" + neutrals[i],
                     neutrals[i] + " track, PID and energy efficiency vs pT;p_{T} (GeV)",
                     50,
                     0,
                     250));
    histograms.h_den_neut_eta_candidate.push_back(
        ibook.book1D("den_cand_vs_eta_" + neutrals[i], neutrals[i] + " simCandidates eta;#eta (GeV)", 50, -3, 3));
    histograms.h_num_neut_eta_candidate_pdgId.push_back(ibook.book1D(
        "num_pid_cand_vs_eta_" + neutrals[i], neutrals[i] + " track and PID efficiency vs eta;#eta (GeV)", 50, -3, 3));
    histograms.h_num_neut_eta_candidate_energy.push_back(
        ibook.book1D("num_energy_cand_vs_eta_" + neutrals[i],
                     neutrals[i] + " track, PID and energy efficiency vs eta;#eta (GeV)",
                     50,
                     -3,
                     3));
    histograms.h_den_neut_phi_candidate.push_back(ibook.book1D(
        "den_cand_vs_phi_" + neutrals[i], neutrals[i] + " simCandidates phi;#phi (GeV)", 50, -3.14159, 3.14159));
    histograms.h_num_neut_phi_candidate_pdgId.push_back(
        ibook.book1D("num_pid_cand_vs_phi_" + neutrals[i],
                     neutrals[i] + " track and PID efficiency vs phi;#phi (GeV)",
                     50,
                     -3.14159,
                     3.14159));
    histograms.h_num_neut_phi_candidate_energy.push_back(
        ibook.book1D("num_energy_cand_vs_phi_" + neutrals[i],
                     neutrals[i] + " track, PID and energy efficiency vs phi;#phi (GeV)",
                     50,
                     -3.14159,
                     3.14159));
  }
  // charged: electron, muon, hadron
  const std::vector<std::string> charged{"electrons", "muons", "charged_hadrons"};
  for (long unsigned int i = 0; i < charged.size(); i++) {
    ibook.setCurrentFolder(baseDir + "/" + charged[i]);

    histograms.h_chg_tracksters_in_candidate.push_back(ibook.book1D(
        "N of tracksters in candidate for " + charged[i], "N of tracksters in candidate for " + charged[i], 100, 0, 99));
    histograms.h_chg_candidate_regressed_energy.push_back(ibook.book1D(
        charged[i] + "candidates regressed energy", charged[i] + " candidates regressed energy;E (GeV)", 250, 0, 250));
    histograms.h_chg_candidate_charge.push_back(
        ibook.book1D(charged[i] + " candidates charge", charged[i] + " candidates charge;Charge", 3, -1.5, 1.5));
    histograms.h_chg_candidate_pdgId.push_back(
        ibook.book1D(charged[i] + " candidates PDG Id", charged[i] + " candidates PDG ID", 100, -220, 220));
    histograms.h_chg_candidate_partType.push_back(
        ibook.book1D(charged[i] + " candidates type", charged[i] + " candidates type", 9, -0.5, 8.5));

    histograms.h_den_fake_chg_energy_candidate.push_back(
        ibook.book1D("den_fake_cand_vs_energy_" + charged[i], charged[i] + " candidates energy;E (GeV)", 50, 0, 250));
    histograms.h_num_fake_chg_energy_candidate_track.push_back(ibook.book1D(
        "num_fake_track_cand_vs_energy_" + charged[i], charged[i] + " track fake vs energy;E (GeV)", 50, 0, 250));
    histograms.h_num_fake_chg_energy_candidate_pdgId.push_back(ibook.book1D(
        "num_fake_pid_cand_vs_energy_" + charged[i], charged[i] + " track and PID fake vs energy;E (GeV)", 50, 0, 250));
    histograms.h_num_fake_chg_energy_candidate_energy.push_back(
        ibook.book1D("num_fake_energy_cand_vs_energy_" + charged[i],
                     charged[i] + " track, PID and energy fake vs energy;E (GeV)",
                     50,
                     0,
                     250));
    histograms.h_den_fake_chg_pt_candidate.push_back(
        ibook.book1D("den_fake_cand_vs_pt_" + charged[i], charged[i] + " candidates pT;p_{T} (GeV)", 50, 0, 250));
    histograms.h_num_fake_chg_pt_candidate_track.push_back(ibook.book1D(
        "num_fake_track_cand_vs_pt_" + charged[i], charged[i] + " track fake vs pT;p_{T} (GeV)", 50, 0, 250));
    histograms.h_num_fake_chg_pt_candidate_pdgId.push_back(ibook.book1D(
        "num_fake_pid_cand_vs_pt_" + charged[i], charged[i] + " track and PID fake vs pT;p_{T} (GeV)", 50, 0, 250));
    histograms.h_num_fake_chg_pt_candidate_energy.push_back(
        ibook.book1D("num_fake_energy_cand_vs_pt_" + charged[i],
                     charged[i] + " track, PID and energy fake vs pT;p_{T} (GeV)",
                     50,
                     0,
                     250));
    histograms.h_den_fake_chg_eta_candidate.push_back(
        ibook.book1D("den_fake_cand_vs_eta_" + charged[i], charged[i] + " candidates eta;#eta (GeV)", 50, -3, 3));
    histograms.h_num_fake_chg_eta_candidate_track.push_back(ibook.book1D(
        "num_fake_track_cand_vs_eta_" + charged[i], charged[i] + " track fake vs eta;#eta (GeV)", 50, -3, 3));
    histograms.h_num_fake_chg_eta_candidate_pdgId.push_back(ibook.book1D(
        "num_fake_pid_cand_vs_eta_" + charged[i], charged[i] + " track and PID fake vs eta;#eta (GeV)", 50, -3, 3));
    histograms.h_num_fake_chg_eta_candidate_energy.push_back(
        ibook.book1D("num_fake_energy_cand_vs_eta_" + charged[i],
                     charged[i] + " track, PID and energy fake vs eta;#eta (GeV)",
                     50,
                     -3,
                     3));
    histograms.h_den_fake_chg_phi_candidate.push_back(ibook.book1D(
        "den_fake_cand_vs_phi_" + charged[i], charged[i] + " candidates phi;#phi (GeV)", 50, -3.14159, 3.14159));
    histograms.h_num_fake_chg_phi_candidate_track.push_back(ibook.book1D("num_fake_track_cand_vs_phi_" + charged[i],
                                                                         charged[i] + " track fake vs phi;#phi (GeV)",
                                                                         50,
                                                                         -3.14159,
                                                                         3.14159));
    histograms.h_num_fake_chg_phi_candidate_pdgId.push_back(
        ibook.book1D("num_fake_pid_cand_vs_phi_" + charged[i],
                     charged[i] + " track and PID fake vs phi;#phi (GeV)",
                     50,
                     -3.14159,
                     3.14159));
    histograms.h_num_fake_chg_phi_candidate_energy.push_back(
        ibook.book1D("num_fake_energy_cand_vs_phi_" + charged[i],
                     charged[i] + " track, PID and energy fake vs phi;#phi (GeV)",
                     50,
                     -3.14159,
                     3.14159));

    histograms.h_den_chg_energy_candidate.push_back(
        ibook.book1D("den_cand_vs_energy_" + charged[i], charged[i] + " simCandidates energy;E (GeV)", 50, 0, 250));
    histograms.h_num_chg_energy_candidate_track.push_back(ibook.book1D(
        "num_track_cand_vs_energy_" + charged[i], charged[i] + " track efficiency vs energy;E (GeV)", 50, 0, 250));
    histograms.h_num_chg_energy_candidate_pdgId.push_back(ibook.book1D(
        "num_pid_cand_vs_energy_" + charged[i], charged[i] + " track and PID efficiency vs energy;E (GeV)", 50, 0, 250));
    histograms.h_num_chg_energy_candidate_energy.push_back(
        ibook.book1D("num_energy_cand_vs_energy_" + charged[i],
                     charged[i] + " track, PID and energy efficiency vs energy;E (GeV)",
                     50,
                     0,
                     250));
    histograms.h_den_chg_pt_candidate.push_back(
        ibook.book1D("den_cand_vs_pt_" + charged[i], charged[i] + " simCandidates pT;p_{T} (GeV)", 50, 0, 250));
    histograms.h_num_chg_pt_candidate_track.push_back(ibook.book1D(
        "num_track_cand_vs_pt_" + charged[i], charged[i] + " track efficiency vs pT;p_{T} (GeV)", 50, 0, 250));
    histograms.h_num_chg_pt_candidate_pdgId.push_back(ibook.book1D(
        "num_pid_cand_vs_pt_" + charged[i], charged[i] + " track and PID efficiency vs pT;p_{T} (GeV)", 50, 0, 250));
    histograms.h_num_chg_pt_candidate_energy.push_back(
        ibook.book1D("num_energy_cand_vs_pt_" + charged[i],
                     charged[i] + " track, PID and energy efficiency vs pT;p_{T} (GeV)",
                     50,
                     0,
                     250));
    histograms.h_den_chg_eta_candidate.push_back(
        ibook.book1D("den_cand_vs_eta_" + charged[i], charged[i] + " simCandidates eta;#eta (GeV)", 50, -3, 3));
    histograms.h_num_chg_eta_candidate_track.push_back(ibook.book1D(
        "num_track_cand_vs_eta_" + charged[i], charged[i] + " track efficiency vs eta;#eta (GeV)", 50, -3, 3));
    histograms.h_num_chg_eta_candidate_pdgId.push_back(ibook.book1D(
        "num_pid_cand_vs_eta_" + charged[i], charged[i] + " track and PID efficiency vs eta;#eta (GeV)", 50, -3, 3));
    histograms.h_num_chg_eta_candidate_energy.push_back(
        ibook.book1D("num_energy_cand_vs_eta_" + charged[i],
                     charged[i] + " track, PID and energy efficiency vs eta;#eta (GeV)",
                     50,
                     -3,
                     3));
    histograms.h_den_chg_phi_candidate.push_back(ibook.book1D(
        "den_cand_vs_phi_" + charged[i], charged[i] + " simCandidates phi;#phi (GeV)", 50, -3.14159, 3.14159));
    histograms.h_num_chg_phi_candidate_track.push_back(ibook.book1D("num_track_cand_vs_phi_" + charged[i],
                                                                    charged[i] + " track efficiency vs phi;#phi (GeV)",
                                                                    50,
                                                                    -3.14159,
                                                                    3.14159));
    histograms.h_num_chg_phi_candidate_pdgId.push_back(
        ibook.book1D("num_pid_cand_vs_phi_" + charged[i],
                     charged[i] + " track and PID efficiency vs phi;#phi (GeV)",
                     50,
                     -3.14159,
                     3.14159));
    histograms.h_num_chg_phi_candidate_energy.push_back(
        ibook.book1D("num_energy_cand_vs_phi_" + charged[i],
                     charged[i] + " track, PID and energy efficiency vs phi;#phi (GeV)",
                     50,
                     -3.14159,
                     3.14159));
  }
}

void TICLCandidateValidator::fillCandidateHistos(const edm::Event& event,
                                                 const Histograms& histograms,
                                                 edm::Handle<ticl::TracksterCollection> simTrackstersCP_h) const {
  auto TICLCandidates = event.get(TICLCandidatesToken_);

  edm::Handle<std::vector<TICLCandidate>> simTICLCandidates_h;
  event.getByToken(simTICLCandidatesToken_, simTICLCandidates_h);
  auto simTICLCandidates = *simTICLCandidates_h;

  edm::Handle<std::vector<reco::Track>> recoTracks_h;
  event.getByToken(recoTracksToken_, recoTracks_h);
  auto recoTracks = *recoTracks_h;

  edm::Handle<std::vector<ticl::Trackster>> Tracksters_h;
  event.getByToken(trackstersToken_, Tracksters_h);
  auto trackstersMerged = *Tracksters_h;

  edm::Handle<ticl::TracksterToTracksterMap> mergeTsRecoToSim_h;
  event.getByToken(associatorMapRtSToken_, mergeTsRecoToSim_h);
  auto const& mergeTsRecoToSimMap = *mergeTsRecoToSim_h;

  edm::Handle<ticl::TracksterToTracksterMap> mergeTsSimToReco_h;
  event.getByToken(associatorMapStRToken_, mergeTsSimToReco_h);
  auto const& mergeTsSimToRecoMap = *mergeTsSimToReco_h;

  // candidates plots
  for (const auto& cand : TICLCandidates) {
    histograms.h_tracksters_in_candidate->Fill(cand.tracksters().size());
    histograms.h_candidate_raw_energy->Fill(cand.rawEnergy());
    histograms.h_candidate_regressed_energy->Fill(cand.energy());
    histograms.h_candidate_pT->Fill(cand.pt());
    histograms.h_candidate_charge->Fill(cand.charge());
    histograms.h_candidate_pdgId->Fill(cand.pdgId());
    const auto& arr = cand.idProbabilities();
    histograms.h_candidate_partType->Fill(std::max_element(arr.begin(), arr.end()) - arr.begin());
  }

  std::vector<int> chargedCandidates;
  std::vector<int> neutralCandidates;
  chargedCandidates.reserve(simTICLCandidates.size());
  neutralCandidates.reserve(simTICLCandidates.size());

  for (size_t i = 0; i < simTICLCandidates.size(); ++i) {
    const auto& simCand = simTICLCandidates[i];
    const auto particleType = ticl::tracksterParticleTypeFromPdgId(simCand.pdgId(), simCand.charge());
    if (particleType == ticl::Trackster::ParticleType::electron or
        particleType == ticl::Trackster::ParticleType::muon or
        particleType == ticl::Trackster::ParticleType::charged_hadron)
      chargedCandidates.emplace_back(i);
    else if (particleType == ticl::Trackster::ParticleType::photon or
             particleType == ticl::Trackster::ParticleType::neutral_pion or
             particleType == ticl::Trackster::ParticleType::neutral_hadron)
      neutralCandidates.emplace_back(i);
    // should consider also unknown ?
  }

  chargedCandidates.shrink_to_fit();
  neutralCandidates.shrink_to_fit();

  for (const auto i : chargedCandidates) {
    const auto& simCand = simTICLCandidates[i];
    auto index = std::log2(int(ticl::tracksterParticleTypeFromPdgId(simCand.pdgId(), 1)));
    /* 11 (type 1) becomes 0
     * 13 (type 2) becomes 1
     * 211 (type 4) becomes 2
     */
    int32_t simCandTrackIdx = -1;
    if (simCand.trackPtr().get() != nullptr)
      simCandTrackIdx = simCand.trackPtr().get() - edm::Ptr<reco::Track>(recoTracks_h, 0).get();
    else {
      // no reco track, but simCand is charged
      continue;
    }
    if (simCand.trackPtr().get()->pt() < 1 or simCand.trackPtr().get()->missingOuterHits() > 5 or
        not simCand.trackPtr().get()->quality(reco::TrackBase::highPurity))
      continue;

    // +1 to all denominators
    histograms.h_den_chg_energy_candidate[index]->Fill(simCand.rawEnergy());
    histograms.h_den_chg_pt_candidate[index]->Fill(simCand.pt());
    histograms.h_den_chg_eta_candidate[index]->Fill(simCand.eta());
    histograms.h_den_chg_phi_candidate[index]->Fill(simCand.phi());

    int32_t cand_idx = -1;
    float shared_energy = 0.;
    const auto ts_vec = mergeTsSimToRecoMap[i];
    if (!ts_vec.empty()) {
      auto min_elem =
          std::min_element(ts_vec.begin(), ts_vec.end(), [](auto const& ts1_id_pair, auto const& ts2_id_pair) {
            return ts1_id_pair.second.second < ts2_id_pair.second.second;
          });
      shared_energy = min_elem->second.first;
      cand_idx = min_elem->first;
    }
    // no reco associated to sim
    if (cand_idx == -1)
      continue;

    auto& recoCand = TICLCandidates[cand_idx];
    if (isTICLv5_) {
      // cand_idx is the tsMerge index, find the ts in the candidates collection
      auto const tsPtr = edm::Ptr<ticl::Trackster>(Tracksters_h, cand_idx);
      auto cand_it = std::find_if(TICLCandidates.begin(), TICLCandidates.end(), [tsPtr](TICLCandidate const& cand) {
        if (!cand.tracksters().empty())
          return cand.tracksters()[0] == tsPtr;
        else
          return false;
      });
      if (cand_it != TICLCandidates.end())
        recoCand = *cand_it;
      else
        continue;
    }

    if (recoCand.trackPtr().get() != nullptr) {
      const auto candTrackIdx = recoCand.trackPtr().get() - edm::Ptr<reco::Track>(recoTracks_h, 0).get();
      if (simCandTrackIdx == candTrackIdx) {
        // +1 to track num
        histograms.h_num_chg_energy_candidate_track[index]->Fill(simCand.rawEnergy());
        histograms.h_num_chg_pt_candidate_track[index]->Fill(simCand.pt());
        histograms.h_num_chg_eta_candidate_track[index]->Fill(simCand.eta());
        histograms.h_num_chg_phi_candidate_track[index]->Fill(simCand.phi());
      } else {
        continue;
      }
    } else {
      continue;
    }

    //step 2: PID
    if (simCand.pdgId() == recoCand.pdgId()) {
      // +1 to num pdg id
      histograms.h_num_chg_energy_candidate_pdgId[index]->Fill(simCand.rawEnergy());
      histograms.h_num_chg_pt_candidate_pdgId[index]->Fill(simCand.pt());
      histograms.h_num_chg_eta_candidate_pdgId[index]->Fill(simCand.eta());
      histograms.h_num_chg_phi_candidate_pdgId[index]->Fill(simCand.phi());

      //step 3: energy
      if (shared_energy / simCand.rawEnergy() > 0.5) {
        // +1 to ene num
        histograms.h_num_chg_energy_candidate_energy[index]->Fill(simCand.rawEnergy());
        histograms.h_num_chg_pt_candidate_energy[index]->Fill(simCand.pt());
        histograms.h_num_chg_eta_candidate_energy[index]->Fill(simCand.eta());
        histograms.h_num_chg_phi_candidate_energy[index]->Fill(simCand.phi());
      }
    }
  }

  for (const auto i : neutralCandidates) {
    const auto& simCand = simTICLCandidates[i];
    auto index = int(ticl::tracksterParticleTypeFromPdgId(simCand.pdgId(), 0)) / 2;
    /* 22 (type 0) becomes 0
     * 111 (type 3) becomes 1
     * 130 (type 5) becomes 2
     */
    histograms.h_den_neut_energy_candidate[index]->Fill(simCand.rawEnergy());
    histograms.h_den_neut_pt_candidate[index]->Fill(simCand.pt());
    histograms.h_den_neut_eta_candidate[index]->Fill(simCand.eta());
    histograms.h_den_neut_phi_candidate[index]->Fill(simCand.phi());

    int32_t cand_idx = -1;
    float shared_energy = 0.;
    const auto ts_vec = mergeTsSimToRecoMap[i];
    if (!ts_vec.empty()) {
      auto min_elem =
          std::min_element(ts_vec.begin(), ts_vec.end(), [](auto const& ts1_id_pair, auto const& ts2_id_pair) {
            return ts1_id_pair.second.second < ts2_id_pair.second.second;
          });
      shared_energy = min_elem->second.first;
      cand_idx = min_elem->first;
    }

    // no reco associated to sim
    if (cand_idx == -1)
      continue;

    auto& recoCand = TICLCandidates[cand_idx];
    if (isTICLv5_) {
      // cand_idx is the tsMerge index, find the ts in the candidates collection
      auto const tsPtr = edm::Ptr<ticl::Trackster>(Tracksters_h, cand_idx);
      auto cand_it = std::find_if(TICLCandidates.begin(), TICLCandidates.end(), [tsPtr](TICLCandidate const& cand) {
        if (!cand.tracksters().empty())
          return cand.tracksters()[0] == tsPtr;
        else
          return false;
      });
      if (cand_it != TICLCandidates.end())
        recoCand = *cand_it;
      else
        continue;
    }

    if (recoCand.trackPtr().get() != nullptr)
      continue;

    //step 2: PID
    if (simCand.pdgId() == recoCand.pdgId()) {
      // +1 to num pdg id
      histograms.h_num_neut_energy_candidate_pdgId[index]->Fill(simCand.rawEnergy());
      histograms.h_num_neut_pt_candidate_pdgId[index]->Fill(simCand.pt());
      histograms.h_num_neut_eta_candidate_pdgId[index]->Fill(simCand.eta());
      histograms.h_num_neut_phi_candidate_pdgId[index]->Fill(simCand.phi());

      //step 3: energy
      if (shared_energy / simCand.rawEnergy() > 0.5) {
        // +1 to ene num
        histograms.h_num_neut_energy_candidate_energy[index]->Fill(simCand.rawEnergy());
        histograms.h_num_neut_pt_candidate_energy[index]->Fill(simCand.pt());
        histograms.h_num_neut_eta_candidate_energy[index]->Fill(simCand.eta());
        histograms.h_num_neut_phi_candidate_energy[index]->Fill(simCand.phi());
      }
    }
  }

  // FAKE rate
  chargedCandidates.clear();
  neutralCandidates.clear();
  chargedCandidates.reserve(TICLCandidates.size());
  neutralCandidates.reserve(TICLCandidates.size());

  auto isCharged = [](int pdgId) {
    pdgId = std::abs(pdgId);
    return (pdgId == 11 or pdgId == 211 or pdgId == 13);
  };

  for (size_t i = 0; i < TICLCandidates.size(); ++i) {
    const auto& cand = TICLCandidates[i];
    const auto& charged = isCharged(cand.pdgId());
    if (charged)
      chargedCandidates.emplace_back(i);
    else
      neutralCandidates.emplace_back(i);

    // should consider also unknown ?
  }

  chargedCandidates.shrink_to_fit();
  neutralCandidates.shrink_to_fit();

  // loop on charged
  for (const auto i : chargedCandidates) {
    const auto& cand = TICLCandidates[i];
    auto index = std::log2(int(ticl::tracksterParticleTypeFromPdgId(cand.pdgId(), 1)));
    /* 11 (type 1) becomes 0
     * 13 (type 2) becomes 1
     * 211 (type 4) becomes 2
     */
    int32_t candTrackIdx = -1;
    candTrackIdx = cand.trackPtr().get() - edm::Ptr<reco::Track>(recoTracks_h, 0).get();

    if (cand.tracksters().empty())
      continue;

    // i is the candidate idx == ts idx only in v4, find ts_idx in v5
    auto mergeTs_id = i;
    if (isTICLv5_) {
      mergeTs_id = cand.tracksters()[0].get() - edm::Ptr<ticl::Trackster>(Tracksters_h, 0).get();
    }

    // +1 to all denominators
    histograms.h_den_fake_chg_energy_candidate[index]->Fill(cand.rawEnergy());
    histograms.h_den_fake_chg_pt_candidate[index]->Fill(cand.pt());
    histograms.h_den_fake_chg_eta_candidate[index]->Fill(cand.eta());
    histograms.h_den_fake_chg_phi_candidate[index]->Fill(cand.phi());

    histograms.h_chg_tracksters_in_candidate[index]->Fill(cand.tracksters().size());
    histograms.h_chg_candidate_regressed_energy[index]->Fill(cand.energy());
    histograms.h_chg_candidate_charge[index]->Fill(cand.charge());
    histograms.h_chg_candidate_pdgId[index]->Fill(cand.pdgId());
    const auto& arr = cand.idProbabilities();
    histograms.h_chg_candidate_partType[index]->Fill(std::max_element(arr.begin(), arr.end()) - arr.begin());

    int32_t simCand_idx = -1;
    const auto sts_vec = mergeTsRecoToSimMap[mergeTs_id];
    float shared_energy = 0.;
    // search for reco cand associated
    if (!sts_vec.empty()) {
      auto min_elem =
          std::min_element(sts_vec.begin(), sts_vec.end(), [](auto const& sts1_id_pair, auto const& sts2_id_pair) {
            return sts1_id_pair.second.second < sts2_id_pair.second.second;
          });
      shared_energy = min_elem->second.first;
      simCand_idx = min_elem->first;
    }

    if (simCand_idx == -1)
      continue;

    const auto& simCand = simTICLCandidates[simCand_idx];
    if (simCand.trackPtr().get() != nullptr) {
      const auto simCandTrackIdx = simCand.trackPtr().get() - edm::Ptr<reco::Track>(recoTracks_h, 0).get();
      if (simCandTrackIdx != candTrackIdx) {
        // fake += 1
        histograms.h_num_fake_chg_energy_candidate_track[index]->Fill(cand.rawEnergy());
        histograms.h_num_fake_chg_pt_candidate_track[index]->Fill(cand.pt());
        histograms.h_num_fake_chg_eta_candidate_track[index]->Fill(cand.eta());
        histograms.h_num_fake_chg_phi_candidate_track[index]->Fill(cand.phi());
        continue;
      }
    } else {
      // fake += 1
      histograms.h_num_fake_chg_energy_candidate_track[index]->Fill(cand.rawEnergy());
      histograms.h_num_fake_chg_pt_candidate_track[index]->Fill(cand.pt());
      histograms.h_num_fake_chg_eta_candidate_track[index]->Fill(cand.eta());
      histograms.h_num_fake_chg_phi_candidate_track[index]->Fill(cand.phi());
      continue;
    }

    //step 2: PID
    if (simCand.pdgId() != cand.pdgId()) {
      // +1 to num fake pdg id
      histograms.h_num_fake_chg_energy_candidate_pdgId[index]->Fill(cand.rawEnergy());
      histograms.h_num_fake_chg_pt_candidate_pdgId[index]->Fill(cand.pt());
      histograms.h_num_fake_chg_eta_candidate_pdgId[index]->Fill(cand.eta());
      histograms.h_num_fake_chg_phi_candidate_pdgId[index]->Fill(cand.phi());
      continue;
    }

    //step 3: energy
    if (shared_energy / simCand.rawEnergy() < 0.5) {
      // +1 to ene num
      histograms.h_num_fake_chg_energy_candidate_energy[index]->Fill(cand.rawEnergy());
      histograms.h_num_fake_chg_pt_candidate_energy[index]->Fill(cand.pt());
      histograms.h_num_fake_chg_eta_candidate_energy[index]->Fill(cand.eta());
      histograms.h_num_fake_chg_phi_candidate_energy[index]->Fill(cand.phi());
    }
  }
  // loop on neutrals
  for (const auto i : neutralCandidates) {
    const auto& cand = TICLCandidates[i];
    auto index = int(ticl::tracksterParticleTypeFromPdgId(cand.pdgId(), 0)) / 2;
    /* 22 (type 0) becomes 0
     * 111 (type 3) becomes 1
     * 130 (type 5) becomes 2
     */

    if (cand.tracksters().empty())
      continue;

    // i is the candidate idx == ts idx only in v4, find ts_idx in v5
    auto mergeTs_id = i;
    if (isTICLv5_) {
      mergeTs_id = cand.tracksters()[0].get() - edm::Ptr<ticl::Trackster>(Tracksters_h, 0).get();
    }

    // +1 to all denominators
    histograms.h_den_fake_neut_energy_candidate[index]->Fill(cand.rawEnergy());
    histograms.h_den_fake_neut_pt_candidate[index]->Fill(cand.pt());
    histograms.h_den_fake_neut_eta_candidate[index]->Fill(cand.eta());
    histograms.h_den_fake_neut_phi_candidate[index]->Fill(cand.phi());

    histograms.h_neut_tracksters_in_candidate[index]->Fill(cand.tracksters().size());
    histograms.h_neut_candidate_regressed_energy[index]->Fill(cand.energy());
    histograms.h_neut_candidate_charge[index]->Fill(cand.charge());
    histograms.h_neut_candidate_pdgId[index]->Fill(cand.pdgId());
    const auto& arr = cand.idProbabilities();
    histograms.h_neut_candidate_partType[index]->Fill(std::max_element(arr.begin(), arr.end()) - arr.begin());

    int32_t simCand_idx = -1;
    const auto sts_vec = mergeTsRecoToSimMap[mergeTs_id];
    float shared_energy = 0.;
    // search for reco cand associated
    if (!sts_vec.empty()) {
      auto min_elem =
          std::min_element(sts_vec.begin(), sts_vec.end(), [](auto const& sts1_id_pair, auto const& sts2_id_pair) {
            return sts1_id_pair.second.second < sts2_id_pair.second.second;
          });
      shared_energy = min_elem->second.first;
      simCand_idx = min_elem->first;
    }

    if (simCand_idx == -1)
      continue;

    const auto& simCand = simTICLCandidates[simCand_idx];

    //step 2: PID
    if (simCand.pdgId() != cand.pdgId()) {
      // +1 to num fake pdg id
      histograms.h_num_fake_neut_energy_candidate_pdgId[index]->Fill(cand.rawEnergy());
      histograms.h_num_fake_neut_pt_candidate_pdgId[index]->Fill(cand.pt());
      histograms.h_num_fake_neut_eta_candidate_pdgId[index]->Fill(cand.eta());
      histograms.h_num_fake_neut_phi_candidate_pdgId[index]->Fill(cand.phi());
      continue;
    }

    //step 3: energy
    if (shared_energy / simCand.rawEnergy() < 0.5) {
      // +1 to ene num
      histograms.h_num_fake_neut_energy_candidate_energy[index]->Fill(cand.rawEnergy());
      histograms.h_num_fake_neut_pt_candidate_energy[index]->Fill(cand.pt());
      histograms.h_num_fake_neut_eta_candidate_energy[index]->Fill(cand.eta());
      histograms.h_num_fake_neut_phi_candidate_energy[index]->Fill(cand.phi());
    }
  }
}

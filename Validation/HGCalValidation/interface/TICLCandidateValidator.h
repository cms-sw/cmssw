#ifndef Validation_HGCalValidation_TICLCandidateValidator_h
#define Validation_HGCalValidation_TICLCandidateValidator_h

#include <iostream>
#include <vector>
#include <unordered_map>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"

#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

#include "DQMServices/Core/interface/DQMStore.h"

namespace ticl {
  using TracksterToTracksterMap =
      ticl::AssociationMap<ticl::mapWithFractionAndScore, std::vector<ticl::Trackster>, std::vector<ticl::Trackster>>;
}

struct TICLCandidateValidatorHistograms {
  dqm::reco::MonitorElement* h_tracksters_in_candidate;
  dqm::reco::MonitorElement* h_candidate_raw_energy;
  dqm::reco::MonitorElement* h_candidate_regressed_energy;
  dqm::reco::MonitorElement* h_candidate_pT;
  dqm::reco::MonitorElement* h_candidate_charge;
  dqm::reco::MonitorElement* h_candidate_pdgId;
  dqm::reco::MonitorElement* h_candidate_partType;

  std::vector<dqm::reco::MonitorElement*> h_den_chg_energy_candidate;
  std::vector<dqm::reco::MonitorElement*> h_num_chg_energy_candidate_track;
  std::vector<dqm::reco::MonitorElement*> h_num_chg_energy_candidate_pdgId;
  std::vector<dqm::reco::MonitorElement*> h_num_chg_energy_candidate_energy;
  std::vector<dqm::reco::MonitorElement*> h_den_chg_pt_candidate;
  std::vector<dqm::reco::MonitorElement*> h_num_chg_pt_candidate_track;
  std::vector<dqm::reco::MonitorElement*> h_num_chg_pt_candidate_pdgId;
  std::vector<dqm::reco::MonitorElement*> h_num_chg_pt_candidate_energy;
  std::vector<dqm::reco::MonitorElement*> h_den_chg_eta_candidate;
  std::vector<dqm::reco::MonitorElement*> h_num_chg_eta_candidate_track;
  std::vector<dqm::reco::MonitorElement*> h_num_chg_eta_candidate_pdgId;
  std::vector<dqm::reco::MonitorElement*> h_num_chg_eta_candidate_energy;
  std::vector<dqm::reco::MonitorElement*> h_den_chg_phi_candidate;
  std::vector<dqm::reco::MonitorElement*> h_num_chg_phi_candidate_track;
  std::vector<dqm::reco::MonitorElement*> h_num_chg_phi_candidate_pdgId;
  std::vector<dqm::reco::MonitorElement*> h_num_chg_phi_candidate_energy;

  std::vector<dqm::reco::MonitorElement*> h_den_neut_energy_candidate;
  std::vector<dqm::reco::MonitorElement*> h_num_neut_energy_candidate_pdgId;
  std::vector<dqm::reco::MonitorElement*> h_num_neut_energy_candidate_energy;
  std::vector<dqm::reco::MonitorElement*> h_den_neut_pt_candidate;
  std::vector<dqm::reco::MonitorElement*> h_num_neut_pt_candidate_pdgId;
  std::vector<dqm::reco::MonitorElement*> h_num_neut_pt_candidate_energy;
  std::vector<dqm::reco::MonitorElement*> h_den_neut_eta_candidate;
  std::vector<dqm::reco::MonitorElement*> h_num_neut_eta_candidate_pdgId;
  std::vector<dqm::reco::MonitorElement*> h_num_neut_eta_candidate_energy;
  std::vector<dqm::reco::MonitorElement*> h_den_neut_phi_candidate;
  std::vector<dqm::reco::MonitorElement*> h_num_neut_phi_candidate_pdgId;
  std::vector<dqm::reco::MonitorElement*> h_num_neut_phi_candidate_energy;

  std::vector<dqm::reco::MonitorElement*> h_den_fake_chg_energy_candidate;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_chg_energy_candidate_track;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_chg_energy_candidate_pdgId;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_chg_energy_candidate_energy;
  std::vector<dqm::reco::MonitorElement*> h_den_fake_chg_pt_candidate;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_chg_pt_candidate_track;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_chg_pt_candidate_pdgId;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_chg_pt_candidate_energy;
  std::vector<dqm::reco::MonitorElement*> h_den_fake_chg_eta_candidate;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_chg_eta_candidate_track;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_chg_eta_candidate_pdgId;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_chg_eta_candidate_energy;
  std::vector<dqm::reco::MonitorElement*> h_den_fake_chg_phi_candidate;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_chg_phi_candidate_track;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_chg_phi_candidate_pdgId;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_chg_phi_candidate_energy;

  std::vector<dqm::reco::MonitorElement*> h_den_fake_neut_energy_candidate;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_neut_energy_candidate_pdgId;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_neut_energy_candidate_energy;
  std::vector<dqm::reco::MonitorElement*> h_den_fake_neut_pt_candidate;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_neut_pt_candidate_pdgId;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_neut_pt_candidate_energy;
  std::vector<dqm::reco::MonitorElement*> h_den_fake_neut_eta_candidate;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_neut_eta_candidate_pdgId;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_neut_eta_candidate_energy;
  std::vector<dqm::reco::MonitorElement*> h_den_fake_neut_phi_candidate;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_neut_phi_candidate_pdgId;
  std::vector<dqm::reco::MonitorElement*> h_num_fake_neut_phi_candidate_energy;

  std::vector<dqm::reco::MonitorElement*> h_chg_tracksters_in_candidate;
  std::vector<dqm::reco::MonitorElement*> h_chg_candidate_regressed_energy;
  std::vector<dqm::reco::MonitorElement*> h_chg_candidate_charge;
  std::vector<dqm::reco::MonitorElement*> h_chg_candidate_pdgId;
  std::vector<dqm::reco::MonitorElement*> h_chg_candidate_partType;

  std::vector<dqm::reco::MonitorElement*> h_neut_tracksters_in_candidate;
  std::vector<dqm::reco::MonitorElement*> h_neut_candidate_regressed_energy;
  std::vector<dqm::reco::MonitorElement*> h_neut_candidate_charge;
  std::vector<dqm::reco::MonitorElement*> h_neut_candidate_pdgId;
  std::vector<dqm::reco::MonitorElement*> h_neut_candidate_partType;
};

class TICLCandidateValidator {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  TICLCandidateValidator(){};
  TICLCandidateValidator(edm::EDGetTokenT<std::vector<TICLCandidate>> TICLCandidates,
                         edm::EDGetTokenT<std::vector<TICLCandidate>> simTICLCandidatesToken,
                         edm::EDGetTokenT<std::vector<reco::Track>> recoTracksToken,
                         edm::EDGetTokenT<std::vector<ticl::Trackster>> trackstersToken,
                         edm::EDGetTokenT<ticl::TracksterToTracksterMap> associatorMapRtSToken,
                         edm::EDGetTokenT<ticl::TracksterToTracksterMap> associatorMapStRToken,
                         bool isTICLv5);
  ~TICLCandidateValidator();

  using Histograms = TICLCandidateValidatorHistograms;

  void bookCandidatesHistos(DQMStore::IBooker& ibook, Histograms& histograms, std::string baseDir) const;

  void fillCandidateHistos(const edm::Event& event,
                           const Histograms& histograms,
                           edm::Handle<ticl::TracksterCollection> simTrackstersCP_h) const;

private:
  edm::EDGetTokenT<std::vector<TICLCandidate>> TICLCandidatesToken_;
  edm::EDGetTokenT<std::vector<TICLCandidate>> simTICLCandidatesToken_;
  edm::EDGetTokenT<std::vector<reco::Track>> recoTracksToken_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>> trackstersToken_;
  edm::EDGetTokenT<ticl::TracksterToTracksterMap> associatorMapRtSToken_;
  edm::EDGetTokenT<ticl::TracksterToTracksterMap> associatorMapStRToken_;
  bool isTICLv5_ = false;
};

#endif

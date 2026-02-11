
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
// including GenParticles
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

// icnluding GenMET
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"

#include "Validation/HLTrigger/interface/HLTGenValObject.h"

#include <vector>

class HLTGenValObjectMgr {
public:
  HLTGenValObjectMgr(const edm::ParameterSet& iConfig, edm::ConsumesCollector cc);

  std::vector<HLTGenValObject> getGenValObjects(const edm::Event& iEvent, const std::string& objType);
  std::vector<HLTGenValObject> getGenParticles(const edm::Event& iEvent, const std::string& objType);
  bool passGenJetID(const reco::GenJet& jet);
  static reco::GenParticle getLastCopy(reco::GenParticle part);
  static reco::GenParticle getLastCopyPreFSR(reco::GenParticle part);

  static edm::ParameterSetDescription makePSetDescription();

private:
  const edm::EDGetTokenT<reco::GenParticleCollection> genParticleToken_;
  const edm::EDGetTokenT<reco::GenMETCollection> genMETToken_;
  const edm::EDGetTokenT<reco::GenJetCollection> ak4GenJetToken_;
  const edm::EDGetTokenT<reco::GenJetCollection> ak8GenJetToken_;
  const edm::EDGetTokenT<reco::GenJetCollection> tauGenJetToken_;

  //some jet id
  //max fraction of pt a prompt particles can contribute to jets
  //basically we would prefer not to lump high pt prompt muons reconstructed as jets
  //in the the category of hadronic jets
  float maxPromptGenJetFrac_;

  float minPtForGenHT_;
  float maxAbsEtaForGenHT_;
};
#include "Validation/HLTrigger/interface/HLTGenValObjectMgr.h"
#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"

HLTGenValObjectMgr::HLTGenValObjectMgr(const edm::ParameterSet& iConfig, edm::ConsumesCollector cc)
    : genParticleToken_(cc.consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("genParticles"))),
      genMETToken_(cc.consumes<reco::GenMETCollection>(iConfig.getParameter<edm::InputTag>("genMET"))),
      ak4GenJetToken_(cc.consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("ak4GenJets"))),
      ak8GenJetToken_(cc.consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("ak8GenJets"))),
      tauGenJetToken_(cc.consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("tauGenJets"))),
      maxPromptGenJetFrac_(iConfig.getParameter<double>("maxPromptGenJetFrac")),
      minPtForGenHT_(iConfig.getParameter<double>("minPtForGenHT")),
      maxAbsEtaForGenHT_(iConfig.getParameter<double>("maxAbsEtaForGenHT")) {}

edm::ParameterSetDescription HLTGenValObjectMgr::makePSetDescription() {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("genParticles", edm::InputTag("genParticles"));
  desc.add<edm::InputTag>("genMET", edm::InputTag("genMetTrue"));
  desc.add<edm::InputTag>("ak4GenJets", edm::InputTag("ak4GenJetsNoNu"));
  desc.add<edm::InputTag>("ak8GenJets", edm::InputTag("ak8GenJetsNoNu"));
  desc.add<edm::InputTag>("tauGenJets", edm::InputTag("tauGenJets"));
  desc.add<double>("maxPromptGenJetFrac", 0.1);
  desc.add<double>("minPtForGenHT", 30);
  desc.add<double>("maxAbsEtaForGenHT", 2.5);
  return desc;
}

// this method handles the different object types and collections that can be used for efficiency calculation
std::vector<HLTGenValObject> HLTGenValObjectMgr::getGenValObjects(const edm::Event& iEvent,
                                                                  const std::string& objType) {
  std::vector<HLTGenValObject> objects;  // the vector of objects to be filled

  // handle object type
  std::vector<std::string> implementedGenParticles = {"ele", "pho", "mu", "tau"};
  if (std::find(implementedGenParticles.begin(), implementedGenParticles.end(), objType) !=
      implementedGenParticles.end()) {
    objects = getGenParticles(iEvent, objType);
  } else if (objType == "AK4jet") {  // ak4 jets, using the ak4GenJets collection
    const auto& genJets = iEvent.getHandle(ak4GenJetToken_);
    for (size_t i = 0; i < genJets->size(); i++) {
      const reco::GenJet p = (*genJets)[i];
      if (passGenJetID(p)) {
        objects.emplace_back(p);
      }
    }
  } else if (objType == "AK8jet") {  // ak8 jets, using the ak8GenJets collection
    const auto& genJets = iEvent.getHandle(ak8GenJetToken_);
    for (size_t i = 0; i < genJets->size(); i++) {
      const reco::GenJet p = (*genJets)[i];
      if (passGenJetID(p)) {
        objects.emplace_back(p);
      }
    }
  } else if (objType == "AK4HT") {  // ak4-based HT, using the ak4GenJets collection
    const auto& genJets = iEvent.getHandle(ak4GenJetToken_);
    if (!genJets->empty()) {
      double HTsum = 0.;
      for (const auto& genJet : *genJets) {
        if (genJet.pt() > minPtForGenHT_ && std::abs(genJet.eta()) < maxAbsEtaForGenHT_ && passGenJetID(genJet)) {
          HTsum += genJet.pt();
        }
      }
      if (HTsum > 0)
        objects.emplace_back(reco::Candidate::PolarLorentzVector(HTsum, 0, 0, 0));
    }
  } else if (objType == "AK8HT") {  // ak8-based HT, using the ak8GenJets collection
    const auto& genJets = iEvent.getHandle(ak8GenJetToken_);
    if (!genJets->empty()) {
      double HTsum = 0.;
      for (const auto& genJet : *genJets) {
        if (genJet.pt() > minPtForGenHT_ && std::abs(genJet.eta()) < maxAbsEtaForGenHT_ && passGenJetID(genJet)) {
          HTsum += genJet.pt();
        }
      }
      if (HTsum > 0)
        objects.emplace_back(reco::Candidate::PolarLorentzVector(HTsum, 0, 0, 0));
    }
  } else if (objType == "MET") {  // MET, using genMET
    const auto& genMET = iEvent.getHandle(genMETToken_);
    if (!genMET->empty()) {
      auto genMETpt = (*genMET)[0].pt();
      objects.emplace_back(reco::Candidate::PolarLorentzVector(genMETpt, 0, 0, 0));
    }
  } else if (objType == "tauHAD") {
    const auto& tauJets = iEvent.getHandle(tauGenJetToken_);
    for (const auto& tauJet : *tauJets) {
      const std::string& decayMode = JetMCTagUtils::genTauDecayMode(tauJet);
      if (decayMode != "electron" && decayMode != "muon") {
        objects.emplace_back(tauJet);
      }
    }
  } else
    throw cms::Exception("InputError") << "Generator-level validation is not available for type " << objType << ".\n"
                                       << "Please check for a potential spelling error.\n";

  return objects;
}

// in case of GenParticles, a subset of the entire collection needs to be chosen
std::vector<HLTGenValObject> HLTGenValObjectMgr::getGenParticles(const edm::Event& iEvent, const std::string& objType) {
  std::vector<HLTGenValObject> objects;  // vector to be filled

  const auto& genParticles = iEvent.getHandle(genParticleToken_);  // getting all GenParticles

  // we need to ge the ID corresponding to the desired GenParticle type
  int pdgID = -1;  // setting to -1 should not be needed, but prevents the compiler warning :)
  if (objType == "ele")
    pdgID = 11;
  else if (objType == "pho")
    pdgID = 22;
  else if (objType == "mu")
    pdgID = 13;
  else if (objType == "tau")
    pdgID = 15;

  // main loop over GenParticles
  for (size_t i = 0; i < genParticles->size(); ++i) {
    const reco::GenParticle p = (*genParticles)[i];

    // only GenParticles with correct ID
    if (std::abs(p.pdgId()) != pdgID)
      continue;

    // checking if particle comes from "hard process"
    if (p.isHardProcess()) {
      // depending on the particle type, last particle before or after FSR is chosen
      if ((objType == "ele") || (objType == "pho"))
        objects.emplace_back(getLastCopyPreFSR(p));
      else if ((objType == "mu") || (objType == "tau"))
        objects.emplace_back(getLastCopy(p));
    }
  }

  return objects;
}

// function returning the last GenParticle in a decay chain before FSR
reco::GenParticle HLTGenValObjectMgr::getLastCopyPreFSR(reco::GenParticle part) {
  const auto& daughters = part.daughterRefVector();
  if (daughters.size() == 1 && daughters.at(0)->pdgId() == part.pdgId())
    return getLastCopyPreFSR(*daughters.at(0).get());  // recursion, whooo
  else
    return part;
}

// function returning the last GenParticle in a decay chain
reco::GenParticle HLTGenValObjectMgr::getLastCopy(reco::GenParticle part) {
  for (const auto& daughter : part.daughterRefVector()) {
    if (daughter->pdgId() == part.pdgId())
      return getLastCopy(*daughter.get());
  }
  return part;
}

bool HLTGenValObjectMgr::passGenJetID(const reco::GenJet& jet) {
  float promptPt = 0;
  for (const auto& genPart : jet.getGenConstituents()) {
    if (genPart->fromHardProcessFinalState()) {
      promptPt += genPart->pt();
    }
  }
  return promptPt < jet.pt() * maxPromptGenJetFrac_;
}
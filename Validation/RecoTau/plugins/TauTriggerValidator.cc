//********************************************************************************
//
//  Description:
//    Standalone DQM validator for tau cross triggers with event level, multi leg
//    gen-level denominators.
//
//    Unlike HLTGenValSource this module selects N required legs per event. Two
//    things are then asked of that event, and they cut across each other:
//
//      * how far down the path it got - the Gen / Gen+L1 / Gen+L1+HLT stages,
//        taken from the path decision and the module index it stopped at;
//      * whether each leg is actually the object that fired the trigger - every
//        leg is deltaR matched against the trigger objects of the filter it
//        names, the way HLTGenValHistCollFilter does it, and "_matched" requires
//        all of them to match.
//
//    Events that match only some of their legs are kept in their own histograms
//    rather than disappearing into the denominator, so a loss can be attributed
//    to a leg. The gap between "_hlt" and "_matched" is how often the path fired
//    on something other than the legs that were selected.
//
//  Author: Youngwan Kim, Chihwan An, Eunsu Kim Sept. 2026
//
//********************************************************************************

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include <algorithm>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

namespace {

  struct LegObject {
    double pt;
    double eta;
    double phi;
  };

  // a gen/trigger object pair inside the matching cone. slotPos and keyPos are
  // positions within the group and within the filter's key list, never the
  // global object index or the key itself.
  struct MatchPair {
    double dR2;
    size_t slotPos;
    size_t keyPos;
  };

  enum Stage { kGen = 0, kL1, kHLT, kMatched, kNStages };
  constexpr const char* kStageSuffix[kNStages] = {"", "_l1", "_hlt", "_matched"};
  constexpr const char* kStageTitle[kNStages] = {"Gen", "Gen+L1", "Gen+L1+HLT", "Gen+object matched"};

  // filters that are ignored are listed with a leading '-' or '!' in the path but
  // not in the trigger summary
  std::string stripFlags(const std::string& s) {
    const auto pos = s.find_first_not_of("-!");
    return pos == std::string::npos ? std::string() : s.substr(pos);
  }
}  // namespace

class TauTriggerValidator : public DQMEDAnalyzer {
public:
  explicit TauTriggerValidator(const edm::ParameterSet&);
  ~TauTriggerValidator() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

  std::vector<LegObject> getLegCandidates(const edm::Event&, size_t legIdx) const;

  // -> one entry per object slot: the matched trigger object key, or -1
  std::vector<int> matchToTriggerObjects(const trigger::TriggerEvent&, const std::vector<LegObject>&) const;

  struct LegConfig {
    std::string objType;  // "tau", "ele", "mu"
    unsigned multiplicity;
    double ptMin;
    double etaMax;
    // extra cuts applied only when filling a histogram that doesn't itself draw
    // the other variable: plateauEtaCut gates pt-only histograms, plateauPtCut
    // gates eta-only histograms. Histograms that draw both (pt_eta) get neither.
    double plateauEtaCut;
    double plateauPtCut;
    std::string filterName;                   // HLT filter whose trigger objects this leg is matched to
    double dR2limit;                          // squared, as in HLTGenValSource
    int pdgId;                                // only used for "ele"/"mu"
    bool requireDirectPromptTauDecayProduct;  // only used for "ele"/"mu"
    int tauTokenIdx = -1;                     // index into genTauTokens_
    int leptonTokenIdx = -1;                  // index into genLeptonTokens_
  };

  std::vector<LegConfig> legs_;
  unsigned nObjs_ = 0;

  // an object slot is one required object: a leg with multiplicity 2 owns two of
  // them. The histogram index leg1/leg2 counts slots, not legs.
  std::vector<std::string> slotObjType_;
  std::vector<unsigned> slotLeg_;

  // slots grouped by the filter they are matched against. Grouping by filter and
  // not by leg is what stops one trigger object from being claimed twice when two
  // slots share a filter, which is the ditau case: one filter with MinN = 2.
  std::vector<std::pair<std::string, std::vector<unsigned>>> matchGroups_;

  std::vector<edm::EDGetTokenT<reco::GenJetCollection>> genTauTokens_;
  std::vector<edm::EDGetTokenT<reco::GenParticleCollection>> genLeptonTokens_;

  const edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  const edm::EDGetTokenT<trigger::TriggerEvent> trigEventToken_;
  const std::string hltPathToCheck_;
  const std::string hltProcessName_;
  const std::string label_;
  const std::string outFolder_;
  const std::string l1SeedModule_;
  const double ptMax_;

  HLTConfigProvider hltConfig_;
  bool hltConfigInitialised_ = false;
  bool pathFound_ = false;
  unsigned pathIndex_ = 0;

  bool l1Found_ = false;
  unsigned l1ModuleIndex_ = 0;

  struct StageHists {
    std::vector<MonitorElement*> pt, eta, phi, ptEta, ptPhi, etaPhi;
    MonitorElement* pt1Pt2 = nullptr;
  };
  StageHists hists_[kNStages];

  // diagnostics for the events that are not in "_matched": which single leg did
  // match, or none at all. Booked with pt and eta only - they are there to point
  // at the leg that lost the event, not to be measured precisely.
  StageHists noMatch_;
  std::vector<StageHists> onlyMatched_;  // indexed by the slot that matched

  void bookStage(DQMStore::IBooker&, StageHists&, const std::string& suffix, const std::string& title, bool full);
  void fill(StageHists&, const std::vector<LegObject>& objs);
  void fillStage(Stage stage, const std::vector<LegObject>& objs) { fill(hists_[stage], objs); }
};

TauTriggerValidator::TauTriggerValidator(const edm::ParameterSet& iConfig)
    : triggerResultsToken_(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("triggerResults"))),
      trigEventToken_(consumes<trigger::TriggerEvent>(iConfig.getParameter<edm::InputTag>("trigEvent"))),
      hltPathToCheck_(iConfig.getParameter<std::string>("hltPath")),
      hltProcessName_(iConfig.getParameter<std::string>("hltProcessName")),
      label_(iConfig.getParameter<std::string>("label")),
      outFolder_(iConfig.getParameter<std::string>("outFolder")),
      l1SeedModule_(iConfig.getParameter<std::string>("l1SeedModule")),
      ptMax_(iConfig.getParameter<double>("ptMax")) {
  const auto legPSets = iConfig.getParameter<std::vector<edm::ParameterSet>>("legs");
  if (legPSets.empty())
    throw cms::Exception("Configuration") << "TauTriggerValidator '" << label_ << "': 'legs' is empty";

  for (auto const& legPSet : legPSets) {
    LegConfig leg;
    leg.objType = legPSet.getParameter<std::string>("objType");
    leg.multiplicity = legPSet.getParameter<unsigned int>("multiplicity");
    leg.ptMin = legPSet.getParameter<double>("ptMin");
    leg.etaMax = legPSet.getParameter<double>("etaMax");
    leg.plateauEtaCut = legPSet.getParameter<double>("plateauEtaCut");
    leg.plateauPtCut = legPSet.getParameter<double>("plateauPtCut");
    leg.filterName = legPSet.getParameter<std::string>("filterName");
    leg.dR2limit = legPSet.getParameter<double>("dR2limit");

    if (leg.filterName.empty()) {
      // an empty filter would only show up as a flat zero efficiency, which is
      // much harder to notice than a configuration error
      throw cms::Exception("Configuration")
          << "TauTriggerValidator '" << label_ << "': leg of type '" << leg.objType
          << "' has no filterName. Give the HLT filter whose trigger objects this leg should be matched to; "
          << "the module lists the saveTags filters of " << hltPathToCheck_ << " at the start of each run.";
    }

    if (leg.objType == "tau") {
      leg.tauTokenIdx = static_cast<int>(genTauTokens_.size());
      genTauTokens_.push_back(consumes<reco::GenJetCollection>(legPSet.getParameter<edm::InputTag>("genCollection")));
    } else if (leg.objType == "ele" || leg.objType == "mu") {
      leg.pdgId = (leg.objType == "ele") ? 11 : 13;
      leg.requireDirectPromptTauDecayProduct = legPSet.getParameter<bool>("fromTauDecay");
      leg.leptonTokenIdx = static_cast<int>(genLeptonTokens_.size());
      genLeptonTokens_.push_back(
          consumes<reco::GenParticleCollection>(legPSet.getParameter<edm::InputTag>("genParticleCollection")));
    } else {
      throw cms::Exception("Configuration")
          << "TauTriggerValidator: unknown leg objType '" << leg.objType << "', must be one of 'tau', 'ele', 'mu'";
    }

    // expand the leg into its object slots, then file those slots under the
    // filter they will be matched against
    const unsigned legIdx = legs_.size();
    const std::string filterName = stripFlags(leg.filterName);
    auto group = std::find_if(
        matchGroups_.begin(), matchGroups_.end(), [&filterName](const auto& g) { return g.first == filterName; });
    if (group == matchGroups_.end()) {
      matchGroups_.emplace_back(filterName, std::vector<unsigned>());
      group = std::prev(matchGroups_.end());
    }
    for (unsigned n = 0; n < leg.multiplicity; ++n) {
      group->second.push_back(nObjs_ + n);
      slotObjType_.push_back(leg.objType);
      slotLeg_.push_back(legIdx);
    }

    nObjs_ += leg.multiplicity;
    legs_.push_back(leg);
  }
}

std::vector<LegObject> TauTriggerValidator::getLegCandidates(const edm::Event& iEvent, size_t legIdx) const {
  std::vector<LegObject> cands;
  const auto& leg = legs_[legIdx];

  if (leg.objType == "tau") {
    auto genTaus = iEvent.getHandle(genTauTokens_[leg.tauTokenIdx]);
    if (!genTaus.isValid())
      return cands;
    for (auto const& jet : *genTaus) {
      if (jet.pt() > leg.ptMin && std::abs(jet.eta()) < leg.etaMax)
        cands.push_back(LegObject{jet.pt(), jet.eta(), jet.phi()});
    }
  } else {
    auto genParts = iEvent.getHandle(genLeptonTokens_[leg.leptonTokenIdx]);
    if (!genParts.isValid())
      return cands;
    for (auto const& p : *genParts) {
      if (p.status() != 1 || std::abs(p.pdgId()) != leg.pdgId)
        continue;
      const bool passFlag = leg.requireDirectPromptTauDecayProduct ? p.statusFlags().isDirectPromptTauDecayProduct()
                                                                   : p.statusFlags().isPrompt();
      if (!passFlag)
        continue;
      if (p.pt() > leg.ptMin && std::abs(p.eta()) < leg.etaMax)
        cands.push_back(LegObject{p.pt(), p.eta(), p.phi()});
    }
  }

  std::sort(cands.begin(), cands.end(), [](const LegObject& a, const LegObject& b) { return a.pt > b.pt; });
  if (cands.size() > leg.multiplicity)
    cands.resize(leg.multiplicity);
  return cands;
}

std::vector<int> TauTriggerValidator::matchToTriggerObjects(const trigger::TriggerEvent& trigEvent,
                                                            const std::vector<LegObject>& genObjs) const {
  std::vector<int> matchedKey(nObjs_, -1);
  const trigger::TriggerObjectCollection& allObjs = trigEvent.getObjects();

  for (auto const& group : matchGroups_) {
    const std::string& filterName = group.first;
    const std::vector<unsigned>& slots = group.second;

    trigger::size_type filterIndex = trigEvent.filterIndex(edm::InputTag(filterName, "", hltProcessName_));
    if (filterIndex >= trigEvent.sizeFilters()) {
      // filterIndex matches on the encoded label:instance:process, so a process
      // name that does not match the one that wrote the summary would lose every
      // match without a word. Retry on the label alone before giving up.
      for (trigger::size_type i = 0; i < trigEvent.sizeFilters(); ++i) {
        if (trigEvent.filterLabel(i) == filterName) {
          filterIndex = i;
          edm::LogWarning("TauTriggerValidator")
              << label_ << ": filter '" << filterName << "' was only found by label, not with process name '"
              << hltProcessName_ << "'. The summary was written by '" << trigEvent.usedProcessName() << "'.";
          break;
        }
      }
    }
    // a filter is absent from the summary in every event where the path stopped
    // before reaching it, which is most of them. That is an unmatched leg, not an
    // error.
    if (filterIndex >= trigEvent.sizeFilters())
      continue;

    const trigger::Keys& keys = trigEvent.filterKeys(filterIndex);
    const size_t nSlots = slots.size();
    const size_t nKeys = keys.size();
    if (nKeys == 0)
      continue;

    // all pairs inside the cone, plus the same thing indexed by slot so the
    // augmenting pass can walk a slot's candidates nearest first
    std::vector<MatchPair> pairs;
    std::vector<std::vector<std::pair<double, size_t>>> adj(nSlots);
    for (size_t slotPos = 0; slotPos < nSlots; ++slotPos) {
      const LegObject& gen = genObjs[slots[slotPos]];
      const double dR2limit = legs_[slotLeg_[slots[slotPos]]].dR2limit;
      for (size_t keyPos = 0; keyPos < nKeys; ++keyPos) {
        const trigger::TriggerObject& trigObj = allObjs[keys[keyPos]];
        const double dR2 = reco::deltaR2(gen.eta, gen.phi, trigObj.eta(), trigObj.phi());
        if (dR2 < dR2limit) {
          pairs.push_back({dR2, slotPos, keyPos});
          adj[slotPos].emplace_back(dR2, keyPos);
        }
      }
      std::sort(adj[slotPos].begin(), adj[slotPos].end());
    }
    std::sort(pairs.begin(), pairs.end(), [](const MatchPair& a, const MatchPair& b) {
      return std::tie(a.dR2, a.slotPos, a.keyPos) < std::tie(b.dR2, b.slotPos, b.keyPos);
    });

    std::vector<int> slotMatch(nSlots, -1);
    std::vector<int> keyOwner(nKeys, -1);

    // closest pair first, which is the assignment that makes physical sense
    for (auto const& p : pairs) {
      if (slotMatch[p.slotPos] == -1 && keyOwner[p.keyPos] == -1) {
        slotMatch[p.slotPos] = static_cast<int>(p.keyPos);
        keyOwner[p.keyPos] = static_cast<int>(p.slotPos);
      }
    }

    // greedy alone is not a maximum matching: the closest pair can take the only
    // trigger object another slot could have used. Since "_matched" wants every
    // slot matched, that would bias the efficiency low. Augmenting paths only add
    // matches, so the greedy assignment survives wherever it was already optimal.
    std::function<bool(size_t, std::vector<bool>&)> augment = [&](size_t slotPos, std::vector<bool>& seen) -> bool {
      for (auto const& cand : adj[slotPos]) {
        const size_t keyPos = cand.second;
        if (seen[keyPos])
          continue;
        seen[keyPos] = true;
        if (keyOwner[keyPos] == -1 || augment(keyOwner[keyPos], seen)) {
          keyOwner[keyPos] = static_cast<int>(slotPos);
          slotMatch[slotPos] = static_cast<int>(keyPos);
          return true;
        }
      }
      return false;
    };
    for (size_t slotPos = 0; slotPos < nSlots; ++slotPos) {
      if (slotMatch[slotPos] == -1) {
        std::vector<bool> seen(nKeys, false);
        augment(slotPos, seen);
      }
    }

    for (size_t slotPos = 0; slotPos < nSlots; ++slotPos) {
      if (slotMatch[slotPos] != -1)
        matchedKey[slots[slotPos]] = static_cast<int>(keys[slotMatch[slotPos]]);
    }
  }

  return matchedKey;
}

void TauTriggerValidator::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  bool changedConfig = false;
  if (!hltConfig_.init(iRun, iSetup, hltProcessName_, changedConfig)) {
    edm::LogError("TauTriggerValidator") << "Initialization of HLTConfigProvider failed!";
    hltConfigInitialised_ = false;
    pathFound_ = false;
    return;
  }
  hltConfigInitialised_ = true;

  pathFound_ = false;
  for (auto const& pathFromConfig : hltConfig_.triggerNames()) {
    if (pathFromConfig.find(hltPathToCheck_) != std::string::npos) {
      pathIndex_ = hltConfig_.triggerIndex(pathFromConfig);
      pathFound_ = (pathIndex_ < hltConfig_.size());
      break;
    }
  }
  if (!pathFound_) {
    edm::LogError("TauTriggerValidator") << "Path '" << hltPathToCheck_ << "' could not be found in the HLT menu "
                                         << "and will not be used.";
    return;
  }

  l1Found_ = false;
  std::string l1Label;
  const auto& labels = hltConfig_.moduleLabels(pathIndex_);
  for (unsigned k = 0; k < labels.size(); ++k) {
    const std::string name = stripFlags(labels[k]);
    const bool isSeed = l1SeedModule_.empty() ? name.find("L1") != std::string::npos : name == l1SeedModule_;
    if (isSeed) {
      l1Label = name;
      l1ModuleIndex_ = k;
      l1Found_ = true;
      break;
    }
  }
  if (l1Found_) {
    edm::LogPrint("TauTriggerValidator") << label_ << ": L1 seed of " << hltConfig_.triggerName(pathIndex_) << " is '"
                                         << l1Label << "' (module " << l1ModuleIndex_ << ")";
  } else {
    edm::LogError("TauTriggerValidator") << label_ << ": no L1 seed module found in path "
                                         << hltConfig_.triggerName(pathIndex_)
                                         << ", the L1 level will stay empty. Set l1SeedModule to fix this.";
  }

  // Only saveTags filters reach the trigger summary, so a leg pointed at anything
  // else can never match and would quietly give a flat zero. Say so once per run,
  // and print the list a new channel has to pick its filters from.
  const std::vector<std::string>& saveTagsMods = hltConfig_.saveTagsModules(pathIndex_);
  std::string listed;
  for (auto const& mod : saveTagsMods)
    listed += (listed.empty() ? "" : ", ") + stripFlags(mod);
  edm::LogPrint("TauTriggerValidator") << label_ << ": saveTags filters of " << hltConfig_.triggerName(pathIndex_)
                                       << ": " << listed;

  for (auto const& leg : legs_) {
    const std::string filterName = stripFlags(leg.filterName);
    const bool onPath = std::any_of(saveTagsMods.begin(), saveTagsMods.end(), [&filterName](const std::string& mod) {
      return stripFlags(mod) == filterName;
    });
    if (!onPath) {
      edm::LogError("TauTriggerValidator")
          << label_ << ": filter '" << filterName << "' is not a saveTags module of "
          << hltConfig_.triggerName(pathIndex_) << ", so its leg will never match. Available: " << listed;
    }
  }
}

void TauTriggerValidator::bookStage(
    DQMStore::IBooker& ibooker, StageHists& h, const std::string& sfx, const std::string& stage, bool full) {
  constexpr int nPt = 60, nEta = 60, nPhi = 32;
  constexpr double etaMax = 3., phiMax = 3.2;

  h.pt.assign(nObjs_, nullptr);
  h.eta.assign(nObjs_, nullptr);
  if (full) {
    h.phi.assign(nObjs_, nullptr);
    h.ptEta.assign(nObjs_, nullptr);
    h.ptPhi.assign(nObjs_, nullptr);
    h.etaPhi.assign(nObjs_, nullptr);
  }

  for (unsigned i = 0; i < nObjs_; ++i) {
    const std::string idx = std::to_string(i + 1);
    const std::string base = "leg" + idx;
    const std::string title = label_ + " " + stage + " leg " + idx;

    h.pt[i] = ibooker.book1D(base + "_pt" + sfx, title + " p_{T};p_{T} [GeV];", nPt, 0., ptMax_);
    h.eta[i] = ibooker.book1D(base + "_eta" + sfx, title + " #eta;#eta;", nEta, -etaMax, etaMax);
    if (!full)
      continue;
    h.phi[i] = ibooker.book1D(base + "_phi" + sfx, title + " #phi;#phi;", nPhi, -phiMax, phiMax);
    h.ptEta[i] =
        ibooker.book2D(base + "_pt_eta" + sfx, title + ";p_{T} [GeV];#eta", nPt, 0., ptMax_, nEta, -etaMax, etaMax);
    h.ptPhi[i] =
        ibooker.book2D(base + "_pt_phi" + sfx, title + ";p_{T} [GeV];#phi", nPt, 0., ptMax_, nPhi, -phiMax, phiMax);
    h.etaPhi[i] =
        ibooker.book2D(base + "_eta_phi" + sfx, title + ";#eta;#phi", nEta, -etaMax, etaMax, nPhi, -phiMax, phiMax);
  }

  if (full && nObjs_ >= 2) {
    h.pt1Pt2 = ibooker.book2D("leg1pt_leg2pt" + sfx,
                              label_ + " " + stage + " leg1 vs leg2 p_{T};p_{T}^{leg1} [GeV];p_{T}^{leg2} [GeV]",
                              nPt,
                              0.,
                              ptMax_,
                              nPt,
                              0.,
                              ptMax_);
  }
}

void TauTriggerValidator::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  ibooker.setCurrentFolder(outFolder_);

  for (int s = 0; s < kNStages; ++s)
    bookStage(ibooker, hists_[s], kStageSuffix[s], kStageTitle[s], true);

  // "exactly one leg matched" is an N x N set of histograms, never 2^N. With one
  // slot neither this nor "_nomatch" is reachable beyond the trivial case, so the
  // breakdown only makes sense from two slots up.
  bookStage(ibooker, noMatch_, "_nomatch", "No leg matched", false);
  if (nObjs_ >= 2) {
    onlyMatched_.resize(nObjs_);
    for (unsigned m = 0; m < nObjs_; ++m) {
      const std::string idx = std::to_string(m + 1);
      bookStage(ibooker,
                onlyMatched_[m],
                "_leg" + idx + "_" + slotObjType_[m] + "_only",
                "Only leg " + idx + " (" + slotObjType_[m] + ") matched",
                false);
    }
  }
}

void TauTriggerValidator::fill(StageHists& h, const std::vector<LegObject>& objs) {
  // plateau cuts: a histogram that doesn't itself draw a variable gets that
  // variable's plateau cut applied before filling, so e.g. the eta efficiency is
  // measured at a fixed, well-above-threshold pt rather than mixing in the pt
  // turn-on. A histogram drawing both pt and eta (pt_eta) gets neither.
  auto etaOk = [this](const std::vector<LegObject>& o, unsigned i) {
    return std::abs(o[i].eta) < legs_[slotLeg_[i]].plateauEtaCut;
  };
  auto ptOk = [this](const std::vector<LegObject>& o, unsigned i) { return o[i].pt > legs_[slotLeg_[i]].plateauPtCut; };

  for (unsigned i = 0; i < nObjs_; ++i) {
    if (!h.pt[i])
      continue;
    if (etaOk(objs, i))
      h.pt[i]->Fill(objs[i].pt);
    if (ptOk(objs, i))
      h.eta[i]->Fill(objs[i].eta);
    if (h.phi.empty())
      continue;
    if (etaOk(objs, i) && ptOk(objs, i))
      h.phi[i]->Fill(objs[i].phi);
    h.ptEta[i]->Fill(objs[i].pt, objs[i].eta);
    if (etaOk(objs, i))
      h.ptPhi[i]->Fill(objs[i].pt, objs[i].phi);
    if (ptOk(objs, i))
      h.etaPhi[i]->Fill(objs[i].eta, objs[i].phi);
  }
  if (h.pt1Pt2 && etaOk(objs, 0) && etaOk(objs, 1))
    h.pt1Pt2->Fill(objs[0].pt, objs[1].pt);
}

void TauTriggerValidator::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  std::vector<LegObject> requiredObjs;
  requiredObjs.reserve(nObjs_);
  for (size_t i = 0; i < legs_.size(); ++i) {
    auto cands = getLegCandidates(iEvent, i);
    if (cands.size() < legs_[i].multiplicity)
      return;  // event does not satisfy the gen-level leg requirements
    requiredObjs.insert(requiredObjs.end(), cands.begin(), cands.end());
  }

  // Gen level :  every event with all required legs (TODO, add kinematic selections)
  fillStage(kGen, requiredObjs);

  // object matching, independent of how far the path got. Nothing here may return
  // early, the stage block below still has to run.
  unsigned nMatched = 0;
  int soleMatchedSlot = -1;
  auto trigEvent = iEvent.getHandle(trigEventToken_);
  if (trigEvent.isValid()) {
    const std::vector<int> matchedKey = matchToTriggerObjects(*trigEvent, requiredObjs);
    for (unsigned i = 0; i < nObjs_; ++i) {
      if (matchedKey[i] >= 0) {
        ++nMatched;
        soleMatchedSlot = static_cast<int>(i);
      }
    }
  }
  if (nMatched == nObjs_)
    fillStage(kMatched, requiredObjs);
  else if (nMatched == 0)
    fill(noMatch_, requiredObjs);
  else if (nMatched == 1 && !onlyMatched_.empty())
    fill(onlyMatched_[soleMatchedSlot], requiredObjs);

  if (!pathFound_)
    return;
  auto triggerResults = iEvent.getHandle(triggerResultsToken_);
  if (!triggerResults.isValid() || pathIndex_ >= triggerResults->size() || !triggerResults->wasrun(pathIndex_))
    return;

  const bool passHLT = triggerResults->accept(pathIndex_);
  const bool passL1 = l1Found_ && (passHLT || triggerResults->index(pathIndex_) > l1ModuleIndex_);

  if (passL1)
    fillStage(kL1, requiredObjs);
  if (passHLT)
    fillStage(kHLT, requiredObjs);
}

void TauTriggerValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("triggerResults", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<edm::InputTag>("trigEvent", edm::InputTag("hltTriggerSummaryAOD", "", "HLT"));
  desc.add<std::string>("hltProcessName", "HLT");
  desc.add<std::string>("hltPath");
  desc.add<std::string>("label");
  desc.add<std::string>("outFolder", "HLT/Tau/CrossTriggerValidation");
  desc.add<std::string>("l1SeedModule", "");  // empty: first module of the path with "L1" in its label
  desc.add<double>("ptMax", 300.);            // raise it for a sample that populates the overflow

  edm::ParameterSetDescription legDesc;
  legDesc.add<std::string>("objType");
  legDesc.add<unsigned int>("multiplicity", 1);
  legDesc.add<double>("ptMin", 20.);
  legDesc.add<double>("etaMax", 2.4);
  // no extra cut by default: plateauEtaCut=999 never excludes, plateauPtCut=0 never excludes
  legDesc.add<double>("plateauEtaCut", 999.);
  legDesc.add<double>("plateauPtCut", 0.);
  // the HLT filter this leg is matched against. It has to be a saveTags module of
  // hltPath, otherwise its objects never reach the trigger summary.
  legDesc.add<std::string>("filterName", "");
  legDesc.add<double>("dR2limit", 0.1);  // squared, as in HLTGenValSource: 0.1 is dR < 0.32
  legDesc.addOptional<edm::InputTag>("genCollection", edm::InputTag("tauGenJetsSelectorAllHadrons"));
  legDesc.addOptional<edm::InputTag>("genParticleCollection", edm::InputTag("genParticles"));
  legDesc.addOptional<bool>("fromTauDecay", true);
  desc.addVPSet("legs", legDesc, std::vector<edm::ParameterSet>());

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(TauTriggerValidator);

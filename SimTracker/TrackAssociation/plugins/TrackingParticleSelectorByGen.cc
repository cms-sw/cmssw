// -*- C++ -*-
//
// Package:    SimGeneral/TrackingParticleSelectorByGen
// Class:      TrackingParticleSelectorByGen
//
/**\class TrackingParticleSelectorByGen TrackingParticleSelectorByGen.cc SimGeneral/TrackingParticleSelectorByGen/plugins/TrackingParticleSelectorByGen.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Enrico Lusiani
//         Created:  Mon, 26 Apr 2021 16:44:34 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "PhysicsTools/HepMCCandAlgos/interface/PdgEntryReplacer.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

namespace helper {
  struct SelectCode {
    enum KeepOrDrop { kKeep, kDrop };
    enum FlagDepth { kNone, kFirst, kAll };
    KeepOrDrop keepOrDrop_;
    FlagDepth daughtersDepth_, mothersDepth_;
    bool all_;
  };
}  // namespace helper

class TrackingParticleSelectorByGen : public edm::stream::EDProducer<> {
  // a lot of this is copied from GenParticlePruner
  // refactor common parts in a separate class
public:
  explicit TrackingParticleSelectorByGen(const edm::ParameterSet &);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;
  void parse(const std::string &selection, helper::SelectCode &code, std::string &cut) const;
  void flagDaughters(const reco::GenParticle &, int);
  void flagMothers(const reco::GenParticle &, int);
  void recursiveFlagDaughters(size_t, const reco::GenParticleCollection &, int, std::vector<size_t> &);
  void recursiveFlagMothers(size_t, const reco::GenParticleCollection &, int, std::vector<size_t> &);
  void getDaughterKeys(std::vector<size_t> &, std::vector<size_t> &, const reco::GenParticleRefVector &) const;
  void getMotherKeys(std::vector<size_t> &, std::vector<size_t> &, const reco::GenParticleRefVector &) const;

  // ----------member data ---------------------------
  edm::EDGetTokenT<TrackingParticleCollection> tpToken_;
  edm::EDGetTokenT<reco::GenParticleCollection> gpToken_;
  bool firstEvent_;
  int keepOrDropAll_;
  std::vector<int> flags_;
  std::vector<std::string> selection_;
  std::vector<std::pair<StringCutObjectSelector<reco::GenParticle>, helper::SelectCode>> select_;
};

using namespace edm;
using namespace std;
using namespace reco;

const int keep = 1, drop = -1;

void TrackingParticleSelectorByGen::parse(const std::string &selection,
                                          ::helper::SelectCode &code,
                                          std::string &cut) const {
  using namespace ::helper;
  size_t f = selection.find_first_not_of(' ');
  size_t n = selection.size();
  string command;
  char c;
  for (; (c = selection[f]) != ' ' && f < n; ++f) {
    command.push_back(c);
  }
  if (command[0] == '+') {
    command.erase(0, 1);
    if (command[0] == '+') {
      command.erase(0, 1);
      code.mothersDepth_ = SelectCode::kAll;
    } else {
      code.mothersDepth_ = SelectCode::kFirst;
    }
  } else
    code.mothersDepth_ = SelectCode::kNone;

  if (command[command.size() - 1] == '+') {
    command.erase(command.size() - 1);
    if (command[command.size() - 1] == '+') {
      command.erase(command.size() - 1);
      code.daughtersDepth_ = SelectCode::kAll;
    } else {
      code.daughtersDepth_ = SelectCode::kFirst;
    }
  } else
    code.daughtersDepth_ = SelectCode::kNone;

  if (command == "keep")
    code.keepOrDrop_ = SelectCode::kKeep;
  else if (command == "drop")
    code.keepOrDrop_ = SelectCode::kDrop;
  else {
    throw Exception(errors::Configuration) << "invalid selection command: " << command << "\n" << endl;
  }
  for (; f < n; ++f) {
    if (selection[f] != ' ')
      break;
  }
  cut = string(selection, f);
  if (cut[0] == '*')
    cut = string(cut, 0, cut.find_first_of(' '));
  code.all_ = cut == "*";
}

TrackingParticleSelectorByGen::TrackingParticleSelectorByGen(const edm::ParameterSet &iConfig)
    : tpToken_(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("trackingParticles"))),
      gpToken_(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("genParticles"))),
      firstEvent_(true),
      keepOrDropAll_(drop),
      selection_(iConfig.getParameter<vector<string>>("select")) {
  produces<TrackingParticleCollection>();
}

void TrackingParticleSelectorByGen::flagDaughters(const reco::GenParticle &gen, int keepOrDrop) {
  const GenParticleRefVector &daughters = gen.daughterRefVector();
  for (GenParticleRefVector::const_iterator i = daughters.begin(); i != daughters.end(); ++i)
    flags_[i->key()] = keepOrDrop;
}

void TrackingParticleSelectorByGen::flagMothers(const reco::GenParticle &gen, int keepOrDrop) {
  const GenParticleRefVector &mothers = gen.motherRefVector();
  for (GenParticleRefVector::const_iterator i = mothers.begin(); i != mothers.end(); ++i)
    flags_[i->key()] = keepOrDrop;
}

void TrackingParticleSelectorByGen::recursiveFlagDaughters(size_t index,
                                                           const reco::GenParticleCollection &src,
                                                           int keepOrDrop,
                                                           std::vector<size_t> &allIndices) {
  GenParticleRefVector daughters = src[index].daughterRefVector();
  // avoid infinite recursion if the daughters are set to "this" particle.
  size_t cachedIndex = index;
  for (GenParticleRefVector::const_iterator i = daughters.begin(); i != daughters.end(); ++i) {
    index = i->key();
    // To also avoid infinite recursion if a "loop" is found in the daughter list,
    // check to make sure the index hasn't already been added.
    if (find(allIndices.begin(), allIndices.end(), index) == allIndices.end()) {
      allIndices.push_back(index);
      if (cachedIndex != index) {
        flags_[index] = keepOrDrop;
        recursiveFlagDaughters(index, src, keepOrDrop, allIndices);
      }
    }
  }
}

void TrackingParticleSelectorByGen::recursiveFlagMothers(size_t index,
                                                         const reco::GenParticleCollection &src,
                                                         int keepOrDrop,
                                                         std::vector<size_t> &allIndices) {
  GenParticleRefVector mothers = src[index].motherRefVector();
  // avoid infinite recursion if the mothers are set to "this" particle.
  size_t cachedIndex = index;
  for (GenParticleRefVector::const_iterator i = mothers.begin(); i != mothers.end(); ++i) {
    index = i->key();
    // To also avoid infinite recursion if a "loop" is found in the daughter list,
    // check to make sure the index hasn't already been added.
    if (find(allIndices.begin(), allIndices.end(), index) == allIndices.end()) {
      allIndices.push_back(index);
      if (cachedIndex != index) {
        flags_[index] = keepOrDrop;
        recursiveFlagMothers(index, src, keepOrDrop, allIndices);
      }
    }
  }
}

void TrackingParticleSelectorByGen::getDaughterKeys(vector<size_t> &daIndxs,
                                                    vector<size_t> &daNewIndxs,
                                                    const GenParticleRefVector &daughters) const {
  for (GenParticleRefVector::const_iterator j = daughters.begin(); j != daughters.end(); ++j) {
    GenParticleRef dau = *j;
    if (find(daIndxs.begin(), daIndxs.end(), dau.key()) == daIndxs.end()) {
      daIndxs.push_back(dau.key());
      int idx = flags_[dau.key()];
      if (idx > 0) {
        daNewIndxs.push_back(idx);
      } else {
        const GenParticleRefVector &daus = dau->daughterRefVector();
        if (!daus.empty())
          getDaughterKeys(daIndxs, daNewIndxs, daus);
      }
    }
  }
}

void TrackingParticleSelectorByGen::getMotherKeys(vector<size_t> &moIndxs,
                                                  vector<size_t> &moNewIndxs,
                                                  const GenParticleRefVector &mothers) const {
  for (GenParticleRefVector::const_iterator j = mothers.begin(); j != mothers.end(); ++j) {
    GenParticleRef mom = *j;
    if (find(moIndxs.begin(), moIndxs.end(), mom.key()) == moIndxs.end()) {
      moIndxs.push_back(mom.key());
      int idx = flags_[mom.key()];
      if (idx >= 0) {
        moNewIndxs.push_back(idx);
      } else {
        const GenParticleRefVector &moms = mom->motherRefVector();
        if (!moms.empty())
          getMotherKeys(moIndxs, moNewIndxs, moms);
      }
    }
  }
}

void TrackingParticleSelectorByGen::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  if (firstEvent_) {
    PdgEntryReplacer rep(iSetup);
    for (vector<string>::const_iterator i = selection_.begin(); i != selection_.end(); ++i) {
      string cut;
      ::helper::SelectCode code;
      parse(*i, code, cut);
      if (code.all_) {
        if (i != selection_.begin())
          throw Exception(errors::Configuration)
              << "selections \"keep *\" and \"drop *\" can be used only as first options. Here used in position # "
              << (i - selection_.begin()) + 1 << "\n"
              << endl;
        switch (code.keepOrDrop_) {
          case ::helper::SelectCode::kDrop:
            keepOrDropAll_ = drop;
            break;
          case ::helper::SelectCode::kKeep:
            keepOrDropAll_ = keep;
        };
      } else {
        cut = rep.replace(cut);
        select_.push_back(make_pair(StringCutObjectSelector<GenParticle>(cut), code));
      }
    }
    firstEvent_ = false;
  }

  const auto &tps = iEvent.get(tpToken_);

  const auto &gps = iEvent.get(gpToken_);

  using namespace ::helper;
  const size_t n = gps.size();
  flags_.clear();
  flags_.resize(n, keepOrDropAll_);
  for (size_t j = 0; j < select_.size(); ++j) {
    const pair<StringCutObjectSelector<GenParticle>, SelectCode> &sel = select_[j];
    SelectCode code = sel.second;
    const StringCutObjectSelector<GenParticle> &cut = sel.first;
    for (size_t i = 0; i < n; ++i) {
      const GenParticle &p = gps[i];
      if (cut(p)) {
        int keepOrDrop = keep;
        switch (code.keepOrDrop_) {
          case SelectCode::kKeep:
            keepOrDrop = keep;
            break;
          case SelectCode::kDrop:
            keepOrDrop = drop;
        };
        flags_[i] = keepOrDrop;
        std::vector<size_t> allIndicesDa;
        std::vector<size_t> allIndicesMo;
        switch (code.daughtersDepth_) {
          case SelectCode::kAll:
            recursiveFlagDaughters(i, gps, keepOrDrop, allIndicesDa);
            break;
          case SelectCode::kFirst:
            flagDaughters(p, keepOrDrop);
            break;
          case SelectCode::kNone:;
        };
        switch (code.mothersDepth_) {
          case SelectCode::kAll:
            recursiveFlagMothers(i, gps, keepOrDrop, allIndicesMo);
            break;
          case SelectCode::kFirst:
            flagMothers(p, keepOrDrop);
            break;
          case SelectCode::kNone:;
        };
      }
    }
  }

  auto out = std::make_unique<TrackingParticleCollection>();
  out->reserve(n);

  for (auto &&tp : tps) {
    auto &associatedGenParticles = tp.genParticles();

    bool isSelected = false;
    for (auto &&assocGen : associatedGenParticles) {
      if (flags_[assocGen.index()] == keep)
        isSelected = true;
    }

    if (isSelected) {
      out->emplace_back(tp);
    }
  }
  iEvent.put(std::move(out));
}

void TrackingParticleSelectorByGen::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("trackingParticles", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("genParticles", edm::InputTag("genParticles"));
  desc.add<vector<string>>("select");

  descriptions.add("tpSelectorByGenDefault", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackingParticleSelectorByGen);

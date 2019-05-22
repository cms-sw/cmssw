/*****************************************************************************
 * Project: CMS detector at the CERN
 *
 * Package: Validation/RecoTau
 *
 * Description:
 *   - Cleans a given object collection of other
 *     cross-object candidates using deltaR-matching.
 *   - For example: can clean a muon collection by
 *      removing all jets in the muon collection.
 *   - Saves collection of the reference vectors of cleaned objects.
 *****************************************************************************/
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <memory>
#include <sstream>
#include <vector>

namespace {

  // Job-level data
  struct Counters {
    explicit Counters(std::string const& label) : moduleLabel{label} {}
    std::string const moduleLabel;
    mutable std::atomic<std::size_t> nObjectsTot{};
    mutable std::atomic<std::size_t> nObjectsClean{};
  };

  template <typename T>
  class ObjectViewCleaner : public edm::stream::EDProducer<edm::GlobalCache<Counters>> {
  public:
    explicit ObjectViewCleaner(edm::ParameterSet const&, Counters const*);

    void produce(edm::Event&, edm::EventSetup const&) override;
    static auto initializeGlobalCache(edm::ParameterSet const& iConfig) {
      return std::make_unique<Counters>(iConfig.getParameter<std::string>("@module_label"));
    }
    static void globalEndJob(Counters const*);

  private:
    // member data
    edm::EDGetTokenT<edm::View<T>> srcCands_;
    std::vector<edm::EDGetTokenT<edm::View<reco::Candidate>>> srcObjectsToRemove_;
    double deltaRMin_;
    StringCutObjectSelector<T, true>
        objKeepCut_;  // lazy parsing, to allow cutting on variables not in reco::Candidate class
    StringCutObjectSelector<reco::Candidate, true> objRemoveCut_;  // lazy parsing, to allow cutting on variables

    auto tagsToTokens(std::vector<edm::InputTag> const&) -> decltype(srcObjectsToRemove_);
    bool isIsolated(edm::Event const&, T const&) const;
  };

  using namespace std;

  template <typename T>
  ObjectViewCleaner<T>::ObjectViewCleaner(edm::ParameterSet const& iConfig, Counters const*)
      : srcCands_{consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("srcObject"))},
        srcObjectsToRemove_{tagsToTokens(iConfig.getParameter<vector<edm::InputTag>>("srcObjectsToRemove"))},
        deltaRMin_{iConfig.getParameter<double>("deltaRMin")},
        objKeepCut_{iConfig.existsAs<std::string>("srcObjectSelection")
                        ? iConfig.getParameter<std::string>("srcObjectSelection")
                        : "",
                    true},
        objRemoveCut_{iConfig.existsAs<std::string>("srcObjectsToRemoveSelection")
                          ? iConfig.getParameter<std::string>("srcObjectsToRemoveSelection")
                          : "",
                      true} {
    produces<edm::RefToBaseVector<T>>();
  }

  //______________________________________________________________________________
  template <typename T>
  void ObjectViewCleaner<T>::produce(edm::Event& iEvent, edm::EventSetup const&) {
    edm::Handle<edm::View<T>> candidates;
    iEvent.getByToken(srcCands_, candidates);
    globalCache()->nObjectsTot += candidates->size();

    auto cleanObjects = std::make_unique<edm::RefToBaseVector<T>>();
    for (unsigned int iCand{}; iCand < candidates->size(); ++iCand) {
      auto const& candidate = candidates->at(iCand);
      if (objKeepCut_(candidate) && isIsolated(iEvent, candidate)) {
        cleanObjects->push_back(candidates->refAt(iCand));
      }
    }
    globalCache()->nObjectsClean += cleanObjects->size();

    iEvent.put(std::move(cleanObjects));
  }

  //______________________________________________________________________________
  template <typename T>
  void ObjectViewCleaner<T>::globalEndJob(Counters const* counters) {
    ostringstream oss;
    oss << "nObjectsTot=" << counters->nObjectsTot << " nObjectsClean=" << counters->nObjectsClean
        << " fObjectsClean=" << 100 * (counters->nObjectsClean / static_cast<double>(counters->nObjectsTot)) << "%\n";
    edm::LogInfo("ObjectViewCleaner") << "++++++++++++++++++++++++++++++++++++++++++++++++++\n"
                                      << counters->moduleLabel << "(ObjectViewCleaner) SUMMARY:\n"
                                      << oss.str() << '\n'
                                      << "++++++++++++++++++++++++++++++++++++++++++++++++++";
  }

  //______________________________________________________________________________
  template <typename T>
  bool ObjectViewCleaner<T>::isIsolated(edm::Event const& iEvent, T const& candidate) const {
    for (auto const& srcObject : srcObjectsToRemove_) {
      edm::Handle<edm::View<reco::Candidate>> objects;
      iEvent.getByToken(srcObject, objects);

      for (unsigned int iObj{}; iObj < objects->size(); ++iObj) {
        auto const& obj = objects->at(iObj);
        if (!objRemoveCut_(obj))
          continue;

        if (reco::deltaR(candidate, obj) < deltaRMin_) {
          return false;
        }
      }
    }
    return true;
  };

  //______________________________________________________________________________
  template <typename T>
  auto ObjectViewCleaner<T>::tagsToTokens(std::vector<edm::InputTag> const& tags) -> decltype(srcObjectsToRemove_) {
    std::vector<edm::EDGetTokenT<edm::View<reco::Candidate>>> result;
    std::transform(std::cbegin(tags), std::cend(tags), std::back_inserter(result), [this](auto const& tag) {
      return this->consumes<edm::View<reco::Candidate>>(tag);
    });
    return result;
  }

}  // anonymous namespace

////////////////////////////////////////////////////////////////////////////////
// plugin definitions
////////////////////////////////////////////////////////////////////////////////

typedef ObjectViewCleaner<reco::Candidate> TauValCandViewCleaner;
typedef ObjectViewCleaner<reco::Jet> TauValJetViewCleaner;
typedef ObjectViewCleaner<reco::Muon> TauValMuonViewCleaner;
typedef ObjectViewCleaner<reco::GsfElectron> TauValGsfElectronViewCleaner;
typedef ObjectViewCleaner<reco::Electron> TauValElectronViewCleaner;
typedef ObjectViewCleaner<reco::Photon> TauValPhotonViewCleaner;
typedef ObjectViewCleaner<reco::Track> TauValTrackViewCleaner;

DEFINE_FWK_MODULE(TauValCandViewCleaner);
DEFINE_FWK_MODULE(TauValJetViewCleaner);
DEFINE_FWK_MODULE(TauValMuonViewCleaner);
DEFINE_FWK_MODULE(TauValGsfElectronViewCleaner);
DEFINE_FWK_MODULE(TauValElectronViewCleaner);
DEFINE_FWK_MODULE(TauValPhotonViewCleaner);
DEFINE_FWK_MODULE(TauValTrackViewCleaner);

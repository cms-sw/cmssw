/*****************************************************************************
 * Project: CMS detector at the CERN
 *
 * Package: Validation/RecoTau
 *
 *
 * Authors:
 *
 *   Kalanand Mishra, Fermilab - kalanand@fnal.gov
 *
 * Description:
 *   - Cleans a given object collection of other
 *     cross-object candidates using deltaR-matching.
 *   - For example: can clean a muon collection by
 *      removing all jets in the muon collection.
 *   - Saves collection of the reference vectors of cleaned objects.
 * History:
 *   Generalized the existing CandViewCleaner
 *
 * 2010 FNAL
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

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <memory>
#include <sstream>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////
template <typename T>
class ObjectViewCleaner : public edm::EDProducer {
public:

  // construction/destruction
  explicit ObjectViewCleaner(edm::ParameterSet const& iConfig);

  // member functions
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;
  void endJob() override;

private:

  // member data
  edm::EDGetTokenT<edm::View<T>> srcCands_;
  std::vector<edm::EDGetTokenT<edm::View<reco::Candidate>>> srcObjectsToRemove_;
  double deltaRMin_;
  std::string moduleLabel_;
  StringCutObjectSelector<T,true> objKeepCut_; // lazy parsing, to allow cutting on variables not in reco::Candidate class
  StringCutObjectSelector<reco::Candidate,true> objRemoveCut_; // lazy parsing, to allow cutting on variables

  std::size_t nObjectsTot_ {};
  std::size_t nObjectsClean_ {};

  auto tagsToTokens(std::vector<edm::InputTag> const&) -> decltype(srcObjectsToRemove_);
  bool isIsolated(edm::Event const&, T const&) const;

};


using namespace std;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

template<typename T>
ObjectViewCleaner<T>::ObjectViewCleaner(edm::ParameterSet const& iConfig)
  : srcCands_{consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("srcObject"))}
  , srcObjectsToRemove_{tagsToTokens(iConfig.getParameter<vector<edm::InputTag>>("srcObjectsToRemove"))}
  , deltaRMin_{iConfig.getParameter<double>("deltaRMin")}
  , moduleLabel_{iConfig.getParameter<string>("@module_label")}
  , objKeepCut_{iConfig.existsAs<std::string>("srcObjectSelection") ? iConfig.getParameter<std::string>("srcObjectSelection") : "", true}
  , objRemoveCut_{iConfig.existsAs<std::string>("srcObjectsToRemoveSelection") ? iConfig.getParameter<std::string>("srcObjectsToRemoveSelection") : "", true}
{
  produces<edm::RefToBaseVector<T>>();
}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
template <typename T>
void ObjectViewCleaner<T>::produce(edm::Event& iEvent, edm::EventSetup const&)
{
  edm::Handle<edm::View<T>> candidates;
  iEvent.getByToken(srcCands_,candidates);
  nObjectsTot_ += candidates->size();

  auto cleanObjects = std::make_unique<edm::RefToBaseVector<T>>();
  for (unsigned int iCand {}; iCand < candidates->size(); ++iCand) {
    auto const& candidate = candidates->at(iCand);
    if (objKeepCut_(candidate) && isIsolated(iEvent, candidate)) {
      cleanObjects->push_back(candidates->refAt(iCand));
    }
  }
  nObjectsClean_ += cleanObjects->size();

  iEvent.put(std::move(cleanObjects));
}


//______________________________________________________________________________
template <typename T>
void ObjectViewCleaner<T>::endJob()
{
  ostringstream oss;
  oss << "nObjectsTot=" << nObjectsTot_ << " nObjectsClean=" << nObjectsClean_
     << " fObjectsClean=" << 100*(nObjectsClean_/static_cast<double>(nObjectsTot_)) << "%\n";
  edm::LogInfo("ObjectViewCleaner") << "++++++++++++++++++++++++++++++++++++++++++++++++++\n"
                                    << moduleLabel_ << "(ObjectViewCleaner) SUMMARY:\n"
                                    << oss.str() << '\n'
                                    << "++++++++++++++++++++++++++++++++++++++++++++++++++";
}

//______________________________________________________________________________
template <typename T>
bool ObjectViewCleaner<T>::isIsolated(edm::Event const& iEvent, T const& candidate) const
{
  for (auto const& srcObject : srcObjectsToRemove_) {
    edm::Handle<edm::View<reco::Candidate>> objects;
    iEvent.getByToken(srcObject, objects);

    for (unsigned int iObj {}; iObj < objects->size() ; ++iObj) {
      auto const& obj = objects->at(iObj);
      if (!objRemoveCut_(obj)) continue;

      if (reco::deltaR(candidate,obj) < deltaRMin_) {
        return false;
      }
    }
  }
  return true;
};

//______________________________________________________________________________
template <typename T>
auto ObjectViewCleaner<T>::tagsToTokens(std::vector<edm::InputTag> const& tags) -> decltype(srcObjectsToRemove_)
{
  std::vector<edm::EDGetTokenT<edm::View<reco::Candidate>>> result;
  std::transform(std::cbegin(tags), std::cend(tags), std::back_inserter(result),
                 [this](auto const& tag) { return this->consumes<edm::View<reco::Candidate>>(tag); });
  return result;
}


////////////////////////////////////////////////////////////////////////////////
// plugin definitions
////////////////////////////////////////////////////////////////////////////////

typedef ObjectViewCleaner<reco::Candidate>   TauValCandViewCleaner;
typedef ObjectViewCleaner<reco::Jet>         TauValJetViewCleaner;
typedef ObjectViewCleaner<reco::Muon>        TauValMuonViewCleaner;
typedef ObjectViewCleaner<reco::GsfElectron> TauValGsfElectronViewCleaner;
typedef ObjectViewCleaner<reco::Electron>    TauValElectronViewCleaner;
typedef ObjectViewCleaner<reco::Photon>      TauValPhotonViewCleaner;
typedef ObjectViewCleaner<reco::Track>       TauValTrackViewCleaner;

DEFINE_FWK_MODULE(TauValCandViewCleaner);
DEFINE_FWK_MODULE(TauValJetViewCleaner);
DEFINE_FWK_MODULE(TauValMuonViewCleaner);
DEFINE_FWK_MODULE(TauValGsfElectronViewCleaner);
DEFINE_FWK_MODULE(TauValElectronViewCleaner);
DEFINE_FWK_MODULE(TauValPhotonViewCleaner);
DEFINE_FWK_MODULE(TauValTrackViewCleaner);

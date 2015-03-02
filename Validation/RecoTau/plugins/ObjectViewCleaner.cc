/*****************************************************************************
 * Project: CMS detector at the CERN
 *
 * Package: PhysicsTools/TagAndProbe
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
////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>
#include <vector>
#include <sstream>


////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////
template<typename T>
class ObjectViewCleaner : public edm::EDProducer
{
public:
  // construction/destruction
  ObjectViewCleaner(const edm::ParameterSet& iConfig);
  virtual ~ObjectViewCleaner();
  
  // member functions
  void produce(edm::Event& iEvent,const edm::EventSetup& iSetup) override;
  void endJob() override;

private:  
  // member data
  edm::InputTag              srcCands_;
  std::vector<edm::InputTag> srcObjects_;
  double                     deltaRMin_;
  
  std::string  moduleLabel_;
  StringCutObjectSelector<T,true> objKeepCut_; // lazy parsing, to allow cutting on variables not in reco::Candidate class
  StringCutObjectSelector<reco::Candidate,true> objRemoveCut_; // lazy parsing, to allow cutting on variables 

  unsigned int nObjectsTot_;
  unsigned int nObjectsClean_;
};


using namespace std;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
template<typename T>
ObjectViewCleaner<T>::ObjectViewCleaner(const edm::ParameterSet& iConfig)
  : srcCands_    (iConfig.getParameter<edm::InputTag>         ("srcObject"))
  , srcObjects_ (iConfig.getParameter<vector<edm::InputTag> >("srcObjectsToRemove"))
  , deltaRMin_  (iConfig.getParameter<double>                ("deltaRMin"))
  , moduleLabel_(iConfig.getParameter<string>                ("@module_label"))
  , objKeepCut_(iConfig.existsAs<std::string>("srcObjectSelection") ? iConfig.getParameter<std::string>("srcObjectSelection") : "", true)
  ,objRemoveCut_(iConfig.existsAs<std::string>("srcObjectsToRemoveSelection") ? iConfig.getParameter<std::string>("srcObjectsToRemoveSelection") : "", true)
  , nObjectsTot_(0)
  , nObjectsClean_(0)
{
  produces<edm::RefToBaseVector<T> >();
}


//______________________________________________________________________________
template<typename T>
ObjectViewCleaner<T>::~ObjectViewCleaner()
{
  
}



////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
template<typename T>
void ObjectViewCleaner<T>::produce(edm::Event& iEvent,const edm::EventSetup& iSetup)
{
  auto_ptr<edm::RefToBaseVector<T> >
    cleanObjects(new edm::RefToBaseVector<T >());

  edm::Handle<edm::View<T> > candidates;
  iEvent.getByLabel(srcCands_,candidates);
  
  bool* isClean = new bool[candidates->size()];
  for (unsigned int iObject=0;iObject<candidates->size();iObject++) isClean[iObject] = true;
  
  for (unsigned int iSrc=0;iSrc<srcObjects_.size();iSrc++) {
    edm::Handle<edm::View<reco::Candidate> > objects;
    iEvent.getByLabel(srcObjects_[iSrc],objects);
    
    for (unsigned int iObject=0;iObject<candidates->size();iObject++) {
      const T& candidate = candidates->at(iObject);
      if (!objKeepCut_(candidate)) isClean[iObject] = false;

      for (unsigned int iObj=0;iObj<objects->size();iObj++) {
        const reco::Candidate& obj = objects->at(iObj);
        if (!objRemoveCut_(obj)) continue;

        double deltaR = reco::deltaR(candidate,obj);
        if (deltaR<deltaRMin_)  isClean[iObject] = false;
      }
    }
  }
  
  for (unsigned int iObject=0;iObject<candidates->size();iObject++)
    if (isClean[iObject]) cleanObjects->push_back(candidates->refAt(iObject));
  
  nObjectsTot_  +=candidates->size();
  nObjectsClean_+=cleanObjects->size();

  delete [] isClean;  
  iEvent.put(cleanObjects);
}


//______________________________________________________________________________
template<typename T>
void ObjectViewCleaner<T>::endJob()
{
  stringstream ss;
  ss<<"nObjectsTot="<<nObjectsTot_<<" nObjectsClean="<<nObjectsClean_
    <<" fObjectsClean="<<100.*(nObjectsClean_/(double)nObjectsTot_)<<"%\n";
  edm::LogInfo("ObjectViewCleaner")<<"++++++++++++++++++++++++++++++++++++++++++++++++++"
	      <<"\n"<<moduleLabel_<<"(ObjectViewCleaner) SUMMARY:\n"<<ss.str()
	      <<"++++++++++++++++++++++++++++++++++++++++++++++++++";
}


////////////////////////////////////////////////////////////////////////////////
// plugin definition
////////////////////////////////////////////////////////////////////////////////


typedef ObjectViewCleaner<reco::Candidate>   TauValCandViewCleaner;
typedef ObjectViewCleaner<reco::Jet>         TauValJetViewCleaner;
typedef ObjectViewCleaner<reco::Muon>        TauValMuonViewCleaner;
typedef ObjectViewCleaner<reco::GsfElectron> TauValGsfElectronViewCleaner;
typedef ObjectViewCleaner<reco::Electron>    TauValElectronViewCleaner;
typedef ObjectViewCleaner<reco::Photon>      TauValPhotonViewCleaner;
typedef ObjectViewCleaner<reco::Track>       TauValTrackViewCleaner;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TauValCandViewCleaner);
DEFINE_FWK_MODULE(TauValJetViewCleaner);
DEFINE_FWK_MODULE(TauValMuonViewCleaner);
DEFINE_FWK_MODULE(TauValGsfElectronViewCleaner);
DEFINE_FWK_MODULE(TauValElectronViewCleaner);
DEFINE_FWK_MODULE(TauValPhotonViewCleaner);
DEFINE_FWK_MODULE(TauValTrackViewCleaner);


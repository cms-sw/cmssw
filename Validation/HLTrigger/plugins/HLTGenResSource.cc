//********************************************************************************
//
//  Description:
//    Producing and filling resolution histograms
//
//
//  Author: Sam Harper, RAL, April 202
//
//********************************************************************************

// system include files
#include <memory>
#include <chrono>
#include <ctime>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

// including GenParticles
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

// icnluding GenMET
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"

// includes needed for histogram creation
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// FunctionDefs
#include "DQMOffline/Trigger/interface/FunctionDefs.h"

// includes of histogram collection class
#include "Validation/HLTrigger/interface/HLTGenResHistColl.h"

// DQMStore
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

// trigger Event
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

// object that can be a GenJet, GenParticle or energy sum
#include "Validation/HLTrigger/interface/HLTGenValObject.h"
#include "Validation/HLTrigger/interface/HLTGenValObjectMgr.h"

class HLTGenResSource : public DQMEDAnalyzer {
public:
  explicit HLTGenResSource(const edm::ParameterSet&);
  ~HLTGenResSource() override = default;
  HLTGenResSource(const HLTGenResSource&) = delete;
  HLTGenResSource& operator=(const HLTGenResSource&) = delete;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const& run, edm::EventSetup const& c) override;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void initCfgs(const edm::Run&, const edm::EventSetup&);

  // functions to get correct object collection for chosen object type
  std::vector<HLTGenValObject> getObjectCollection(const edm::Event&);
  std::vector<HLTGenValObject> getGenParticles(const edm::Event&);
  reco::GenParticle getLastCopyPreFSR(reco::GenParticle part);
  reco::GenParticle getLastCopy(reco::GenParticle part);
  bool passGenJetID(const reco::GenJet& jet);
  // ----------member data ---------------------------

  // tokens to get collections
  HLTGenValObjectMgr genObjMgr_;
  const edm::EDGetTokenT<trigger::TriggerEvent> trigEventToken_;

  bool initalised_;
  bool booked_;
  // config strings/Psets
  std::string objType_;
  std::string dirName_;
  std::vector<edm::ParameterSet> histConfigs_;
  std::vector<edm::ParameterSet> histConfigs2D_;
  std::vector<edm::ParameterSet> binnings_;
  std::string hltProcessName_;
  HLTConfigProvider hltConfig_;

  // constructing the info string, which will be written to the output file for display of information in the GUI
  // the string will have a JSON formating, thus starting here with the opening bracket, which will be close directly before saving to the root file
  std::string infoString_ = "{";

  // histogram collection per path
  std::vector<HLTGenResHistColl> collections_;
};

HLTGenResSource::HLTGenResSource(const edm::ParameterSet& iConfig)
    : genObjMgr_(iConfig.getParameter<edm::ParameterSet>("genConfig"), consumesCollector()),

      trigEventToken_(consumes<trigger::TriggerEvent>(iConfig.getParameter<edm::InputTag>("trigEvent"))),
      initalised_(false),
      booked_(false)

{
  // getting all other configurations
  dirName_ = iConfig.getParameter<std::string>("dqmDirName");
  hltProcessName_ = iConfig.getParameter<std::string>("hltProcessName");

  auto resCollections = iConfig.getParameter<std::vector<edm::ParameterSet>>("resCollConfigs");
  for (const auto& resCollConfig : resCollections) {
    collections_.emplace_back(HLTGenResHistColl(resCollConfig, hltProcessName_));
  }
}

void HLTGenResSource::initCfgs(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  // writing general information to info JSON
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);

  // date and time of running this
  std::ostringstream timeStringStream;
  timeStringStream << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
  auto timeString = timeStringStream.str();
  infoString_ += "\"date & time\":\"" + timeString + "\",";

  // CMSSW version
  std::string cmsswVersion = std::getenv("CMSSW_VERSION");
  infoString_ += std::string("\"CMSSW release\":\"") + cmsswVersion + "\",";

  // Initialize hltConfig, for cross-checking whether chosen paths exist
  bool changedConfig;
  if (!hltConfig_.init(iRun, iSetup, hltProcessName_, changedConfig)) {
    edm::LogError("HLTGenResSource") << "Initialization of HLTConfigProvider failed!";
    return;
  }

  // global tag
  infoString_ += std::string("\"global tag\":\"") + hltConfig_.globalTag() + "\",";

  // confDB table name
  infoString_ += std::string("\"HLT ConfDB table\":\"") + hltConfig_.tableName() + "\"}";

  initalised_ = true;
}

void HLTGenResSource::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  if (!initalised_)
    initCfgs(iRun, iSetup);
}

// ------------ method called for each event  ------------
void HLTGenResSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // creating the collection of HLTGenValObjects

  // init triggerEvent, which is always needed
  edm::Handle<trigger::TriggerEvent> triggerEvent;
  iEvent.getByToken(trigEventToken_, triggerEvent);

  // loop over all objects and fill hists
  for (auto& collection : collections_) {
    const std::vector<HLTGenValObject> objects = genObjMgr_.getGenValObjects(iEvent, collection.objType());
    for (const auto& object : objects) {
      collection.fillHists(object, triggerEvent);
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void HLTGenResSource::bookHistograms(DQMStore::IBooker& iBooker, const edm::Run& run, const edm::EventSetup& setup) {
  if (booked_)
    return;
  iBooker.setCurrentFolder(dirName_);

  iBooker.bookString("HLTGenResInfo", infoString_);

  for (long unsigned int collNr = 0; collNr < collections_.size(); collNr++) {
    collections_[collNr].bookHists(iBooker);
  }
  booked_ = true;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HLTGenResSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("dqmDirName", "HLTGenVal");
  desc.add<std::string>("hltProcessName", "HLT");
  desc.add<edm::InputTag>("trigEvent", edm::InputTag("hltTriggerSummaryAOD"));
  desc.add<edm::ParameterSetDescription>("genConfig", HLTGenValObjectMgr::makePSetDescription());
  desc.addVPSet("resCollConfigs", HLTGenResHistColl::makePSetDescription(), std::vector<edm::ParameterSet>());

  descriptions.addDefault(desc);
}

//define this as a framework plug-in
DEFINE_FWK_MODULE(HLTGenResSource);

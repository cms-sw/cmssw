//********************************************************************************
//
//  Description:
//    Producing and filling histograms for generator-level HLT path efficiency histograms. Harvested by a HLTGenValClient.
//
// Implementation:
//   Histograms for objects of a certain type are created for multiple paths chosen by the user: for all objects,
//   and for objects passing filters in the path, determined by deltaR matching.
//   Each HLTGenValSource can handle a single object type and any number of paths (and filters in them)
//
//  Author: Finn Labe, UHH, Jul. 2022
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
#include "Validation/HLTrigger/interface/HLTGenValHistCollPath.h"

// DQMStore
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

// trigger Event
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

// object that can be a GenJet, GenParticle or energy sum
#include "Validation/HLTrigger/interface/HLTGenValObject.h"
#include "Validation/HLTrigger/interface/HLTGenValObjectMgr.h"

class HLTGenValSource : public DQMEDAnalyzer {
public:
  explicit HLTGenValSource(const edm::ParameterSet&);
  ~HLTGenValSource() override = default;
  HLTGenValSource(const HLTGenValSource&) = delete;
  HLTGenValSource& operator=(const HLTGenValSource&) = delete;

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
  std::string sampleLabel_;  //this if set is the label in the legend

  // constructing the info string, which will be written to the output file for display of information in the GUI
  // the string will have a JSON formating, thus starting here with the opening bracket, which will be close directly before saving to the root file
  std::string infoString_ = "{";

  // histogram collection per path
  std::vector<HLTGenValHistCollPath> collectionPath_;

  // HLT config provider/getter
  HLTConfigProvider hltConfig_;

  // some miscellaneous member variables
  std::vector<std::string> hltPathsToCheck_;
  std::vector<std::string> hltPaths_;
  std::vector<std::string> hltPathSpecificCuts_;
  double dR2limit_;
  bool doOnlyLastFilter_;
};

HLTGenValSource::HLTGenValSource(const edm::ParameterSet& iConfig)
    : genObjMgr_(iConfig.getParameter<edm::ParameterSet>("genConfig"), consumesCollector()),
      trigEventToken_(consumes<trigger::TriggerEvent>(iConfig.getParameter<edm::InputTag>("trigEvent"))),
      initalised_(false),
      booked_(false)

{
  // getting the histogram configurations
  histConfigs_ = iConfig.getParameterSetVector("histConfigs");
  histConfigs2D_ = iConfig.getParameterSetVector("histConfigs2D");
  binnings_ = iConfig.getParameterSetVector("binnings");

  // getting all other configurations
  dirName_ = iConfig.getParameter<std::string>("dqmDirName");
  objType_ = iConfig.getParameter<std::string>("objType");
  dR2limit_ = iConfig.getParameter<double>("dR2limit");
  doOnlyLastFilter_ = iConfig.getParameter<bool>("doOnlyLastFilter");
  hltProcessName_ = iConfig.getParameter<std::string>("hltProcessName");
  sampleLabel_ = iConfig.getParameter<std::string>("sampleLabel");
  hltPathsToCheck_ = iConfig.getParameter<std::vector<std::string>>("hltPathsToCheck");
}

void HLTGenValSource::initCfgs(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  // writing general information to info JSON
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);

  // date and time of running this
  std::ostringstream timeStringStream;
  timeStringStream << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
  auto timeString = timeStringStream.str();
  infoString_ += "\"date & time\":\"" + timeString + "\",";

  std::string cmsswVersion;
  const edm::ProcessHistory& processHistory = iRun.processHistory();
  for (const auto& process : processHistory) {
    if (process.processName() == hltProcessName_) {
      cmsswVersion = process.releaseVersion();  //this has quotes around it
      break;
    }
  }
  if (cmsswVersion.empty()) {
    cmsswVersion =
        "\"" + std::string(std::getenv("CMSSW_VERSION")) + "\"";  //using convention it already has quotes on it
  }
  infoString_ += std::string("\"CMSSW release\":") + cmsswVersion + ",";
  infoString_ += std::string("\"sample label\":\"") + sampleLabel_ + "\",";

  // Initialize hltConfig, for cross-checking whether chosen paths exist
  bool changedConfig;
  if (!hltConfig_.init(iRun, iSetup, hltProcessName_, changedConfig)) {
    edm::LogError("HLTGenValSource") << "Initialization of HLTConfigProvider failed!";
    return;
  }

  // global tag
  infoString_ += std::string("\"global tag\":\"") + hltConfig_.globalTag() + "\",";

  // confDB table name
  infoString_ += std::string("\"HLT ConfDB table\":\"") + hltConfig_.tableName() + "\"}";

  // Get the set of trigger paths we want to make plots for
  std::vector<std::string> notFoundPaths;
  for (auto const& pathToCheck : hltPathsToCheck_) {
    // It is possible to add additional requirements to each path, seperated by a colon from the path name
    // these additional requirements are split from the path name here
    std::string cleanedPathToCheck;
    std::string pathSpecificCuts = "";
    if (pathToCheck.find(':') != std::string::npos) {
      // splitting the string
      std::stringstream hltPathToCheckInputStream(pathToCheck);
      std::string hltPathToCheckInputSegment;
      std::vector<std::string> hltPathToCheckInputSeglist;
      while (std::getline(hltPathToCheckInputStream, hltPathToCheckInputSegment, ':')) {
        hltPathToCheckInputSeglist.push_back(hltPathToCheckInputSegment);
      }

      // here, exactly two parts are expected
      if (hltPathToCheckInputSeglist.size() != 2)
        throw cms::Exception("InputError")
            << "Path string " << pathToCheck
            << " can not be properly split into path and cuts: please use exactly one colon!.\n";

      // the first part is the name of the path
      cleanedPathToCheck = hltPathToCheckInputSeglist.at(0);

      // second part are the cuts, to be parsed later
      pathSpecificCuts = hltPathToCheckInputSeglist.at(1);

    } else {
      cleanedPathToCheck = pathToCheck;
    }

    bool pathfound = false;
    for (auto const& pathFromConfig : hltConfig_.triggerNames()) {
      if (pathFromConfig.find(cleanedPathToCheck) != std::string::npos) {
        hltPaths_.push_back(pathFromConfig);

        // in case the path was added twice, we'll add a tag automatically
        int count = std::count(hltPaths_.begin(), hltPaths_.end(), pathFromConfig);
        if (count > 1) {
          pathSpecificCuts += std::string(",autotag=v") + std::to_string(count);
        }
        hltPathSpecificCuts_.push_back(pathSpecificCuts);
        pathfound = true;
      }
    }
    if (!pathfound)
      notFoundPaths.push_back(cleanedPathToCheck);
  }
  if (!notFoundPaths.empty()) {
    // error handling in case some paths do not exist
    std::string notFoundPathsMessage = "";
    for (const auto& path : notFoundPaths)
      notFoundPathsMessage += "- " + path + "\n";
    edm::LogError("HLTGenValSource") << "The following paths could not be found and will not be used: \n"
                                     << notFoundPathsMessage << std::endl;
  }

  // before creating the collections for each path, we'll store the needed configurations in a pset
  // we'll copy this base multiple times and add the respective path
  // most of these options are not needed in the pathColl, but in the filterColls created in the pathColl
  edm::ParameterSet pathCollConfig;
  pathCollConfig.addParameter<std::string>("objType", objType_);
  pathCollConfig.addParameter<double>("dR2limit", dR2limit_);
  pathCollConfig.addParameter<bool>("doOnlyLastFilter", doOnlyLastFilter_);
  pathCollConfig.addParameter<std::string>("hltProcessName", hltProcessName_);

  // creating a histogram collection for each path
  for (const auto& path : hltPaths_) {
    edm::ParameterSet pathCollConfigStep = pathCollConfig;
    pathCollConfigStep.addParameter<std::string>("triggerPath", path);
    collectionPath_.emplace_back(HLTGenValHistCollPath(pathCollConfigStep, hltConfig_));
  }
  initalised_ = true;
}

void HLTGenValSource::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  if (!initalised_)
    initCfgs(iRun, iSetup);
}

// ------------ method called for each event  ------------
void HLTGenValSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // creating the collection of HLTGenValObjects
  const std::vector<HLTGenValObject> objects = genObjMgr_.getGenValObjects(iEvent, objType_);

  // init triggerEvent, which is always needed
  edm::Handle<trigger::TriggerEvent> triggerEvent;
  iEvent.getByToken(trigEventToken_, triggerEvent);

  // loop over all objects and fill hists
  for (const auto& object : objects) {
    for (auto& collection_path : collectionPath_) {
      collection_path.fillHists(object, triggerEvent);
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void HLTGenValSource::bookHistograms(DQMStore::IBooker& iBooker, const edm::Run& run, const edm::EventSetup& setup) {
  iBooker.setCurrentFolder(dirName_);
  iBooker.bookString("HLTGenValInfo", infoString_);

  // booking all histograms
  for (long unsigned int i = 0; i < collectionPath_.size(); i++) {
    std::vector<edm::ParameterSet> histConfigs = histConfigs_;
    for (auto& histConfig : histConfigs) {
      histConfig.addParameter<std::string>("pathSpecificCuts", hltPathSpecificCuts_.at(i));
      histConfig.addParameter<std::vector<edm::ParameterSet>>("binnings",
                                                              binnings_);  // passing along the user-defined binnings
    }

    collectionPath_.at(i).bookHists(iBooker, histConfigs, histConfigs2D_);
  }
  booked_ = true;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HLTGenValSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // basic parameter strings
  desc.add<std::string>(
      "objType");  // this deliberately has no default, as this is the main thing the user needs to chose
  desc.add<std::vector<std::string>>(
      "hltPathsToCheck");  // this for the moment also has no default: maybe there can be some way to handle this later?
  desc.add<std::string>("dqmDirName", "HLTGenVal");
  desc.add<std::string>("hltProcessName", "HLT");
  desc.add<std::string>("sampleLabel", "");  //this is the label in the legend
  desc.add<double>("dR2limit", 0.1);
  desc.add<bool>("doOnlyLastFilter", false);

  desc.add<edm::ParameterSetDescription>("genConfig", HLTGenValObjectMgr::makePSetDescription());
  desc.add<edm::InputTag>("trigEvent", edm::InputTag("hltTriggerSummaryAOD"));

  // hist descriptors, which are a vector of PSets

  // defining single histConfig
  // this is generally without default, but a default set of histConfigs is specified below
  edm::ParameterSetDescription histConfig;
  histConfig.add<std::string>("vsVar");
  histConfig.add<std::vector<double>>("binLowEdges");
  histConfig.addVPSet(
      "rangeCuts", VarRangeCut<HLTGenValObject>::makePSetDescription(), std::vector<edm::ParameterSet>());

  // default set of histConfigs
  std::vector<edm::ParameterSet> histConfigDefaults;

  edm::ParameterSet histConfigDefault0;
  histConfigDefault0.addParameter<std::string>("vsVar", "pt");
  std::vector<double> defaultPtBinning{0,  5,  10, 12.5, 15,  17.5, 20,  22.5, 25,  30,  35, 40,
                                       45, 50, 60, 80,   100, 150,  200, 250,  300, 350, 400};
  histConfigDefault0.addParameter<std::vector<double>>("binLowEdges", defaultPtBinning);
  histConfigDefaults.push_back(histConfigDefault0);

  edm::ParameterSet histConfigDefault1;
  histConfigDefault1.addParameter<std::string>("vsVar", "eta");
  std::vector<double> defaultetaBinning{-10, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8, 10};
  histConfigDefault1.addParameter<std::vector<double>>("binLowEdges", defaultetaBinning);
  histConfigDefaults.push_back(histConfigDefault1);

  desc.addVPSet("histConfigs", histConfig, histConfigDefaults);

  // defining single histConfig2D
  edm::ParameterSetDescription histConfig2D;
  histConfig2D.add<std::string>("vsVarX");
  histConfig2D.add<std::string>("vsVarY");
  histConfig2D.add<std::vector<double>>("binLowEdgesX");
  histConfig2D.add<std::vector<double>>("binLowEdgesY");
  histConfig2D.addVPSet(
      "rangeCuts", VarRangeCut<HLTGenValObject>::makePSetDescription(), std::vector<edm::ParameterSet>());
  // default set of histConfigs
  std::vector<edm::ParameterSet> histConfigDefaults2D;

  edm::ParameterSet histConfigDefault2D0;
  histConfigDefault2D0.addParameter<std::string>("vsVarX", "pt");
  histConfigDefault2D0.addParameter<std::string>("vsVarY", "eta");
  histConfigDefault2D0.addParameter<std::vector<double>>("binLowEdgesX", defaultPtBinning);
  histConfigDefault2D0.addParameter<std::vector<double>>("binLowEdgesY", defaultetaBinning);

  histConfigDefaults2D.push_back(histConfigDefault2D0);

  desc.addVPSet("histConfigs2D", histConfig2D, histConfigDefaults2D);

  // binnings, which are vectors of PSets
  // there are no default for this
  edm::ParameterSetDescription binningConfig;
  binningConfig.add<std::string>("name");
  binningConfig.add<std::string>("vsVar");
  binningConfig.add<std::vector<double>>("binLowEdges");

  // this by default is empty
  std::vector<edm::ParameterSet> binningConfigDefaults;

  desc.addVPSet("binnings", binningConfig, binningConfigDefaults);

  descriptions.addDefault(desc);
}

//define this as a framework plug-in
DEFINE_FWK_MODULE(HLTGenValSource);

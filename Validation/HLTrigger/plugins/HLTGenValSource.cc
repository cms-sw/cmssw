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

  // functions to get correct object collection for chosen object type
  std::vector<HLTGenValObject> getObjectCollection(const edm::Event&);
  std::vector<HLTGenValObject> getGenParticles(const edm::Event&);
  reco::GenParticle getLastCopyPreFSR(reco::GenParticle part);
  reco::GenParticle getLastCopy(reco::GenParticle part);

  // ----------member data ---------------------------

  // tokens to get collections
  const edm::EDGetTokenT<reco::GenParticleCollection> genParticleToken_;
  const edm::EDGetTokenT<reco::GenMETCollection> genMETToken_;
  const edm::EDGetTokenT<reco::GenJetCollection> ak4genJetToken_;
  const edm::EDGetTokenT<reco::GenJetCollection> ak8genJetToken_;
  const edm::EDGetTokenT<trigger::TriggerEvent> trigEventToken_;

  // config strings/Psets
  std::string objType_;
  std::string dirName_;
  std::vector<edm::ParameterSet> histConfigs_;
  std::vector<edm::ParameterSet> histConfigs2D_;
  std::vector<edm::ParameterSet> binnings_;
  std::string hltProcessName_;

  // constructing the info string, which will be written to the output file for display of information in the GUI
  // the string will have a JSON formating, thus starting here with the opening bracket, which will be close directly before saving to the root file
  std::string infoString_ = "{";

  // histogram collection per path
  std::vector<HLTGenValHistCollPath> collectionPath_;

  // HLT config provider/getter
  HLTConfigProvider hltConfig_;

  // some miscellaneous member variables
  std::vector<std::string> hltPathsToCheck_;
  std::vector<std::string> hltPaths;
  std::vector<std::string> hltPathSpecificCuts;
  double dR2limit_;
  bool doOnlyLastFilter_;
};

HLTGenValSource::HLTGenValSource(const edm::ParameterSet& iConfig)
    : genParticleToken_(consumes<reco::GenParticleCollection>(
          iConfig.getParameterSet("inputCollections").getParameter<edm::InputTag>("genParticles"))),
      genMETToken_(consumes<reco::GenMETCollection>(
          iConfig.getParameterSet("inputCollections").getParameter<edm::InputTag>("genMET"))),
      ak4genJetToken_(consumes<reco::GenJetCollection>(
          iConfig.getParameterSet("inputCollections").getParameter<edm::InputTag>("ak4GenJets"))),
      ak8genJetToken_(consumes<reco::GenJetCollection>(
          iConfig.getParameterSet("inputCollections").getParameter<edm::InputTag>("ak8GenJets"))),
      trigEventToken_(consumes<trigger::TriggerEvent>(
          iConfig.getParameterSet("inputCollections").getParameter<edm::InputTag>("TrigEvent"))) {
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
  hltPathsToCheck_ = iConfig.getParameter<std::vector<std::string>>("hltPathsToCheck");
}

void HLTGenValSource::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
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
    edm::LogError("HLTGenValSource") << "Initialization of HLTConfigProvider failed!";
    return;
  }

  // global tag
  infoString_ += std::string("\"global tag\":\"") + hltConfig_.globalTag() + "\",";

  // confDB table name
  infoString_ += std::string("\"HLT ConfDB table\":\"") + hltConfig_.tableName() + "\",";

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
            << "Path string can not be properly split into path and cuts: please use exactly one colon!.\n";

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
        hltPaths.push_back(pathFromConfig);

        // in case the path was added twice, we'll add a tag automatically
        int count = std::count(hltPaths.begin(), hltPaths.end(), pathFromConfig);
        if (count > 1) {
          pathSpecificCuts += std::string(",autotag=v") + std::to_string(count);
        }
        hltPathSpecificCuts.push_back(pathSpecificCuts);
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
  for (const auto& path : hltPaths) {
    edm::ParameterSet pathCollConfigStep = pathCollConfig;
    pathCollConfigStep.addParameter<std::string>("triggerPath", path);
    collectionPath_.emplace_back(HLTGenValHistCollPath(pathCollConfigStep, hltConfig_));
  }
}

// ------------ method called for each event  ------------
void HLTGenValSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // creating the collection of HLTGenValObjects
  const std::vector<HLTGenValObject> objects = getObjectCollection(iEvent);

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

  if (infoString_.back() == ',')
    infoString_.pop_back();
  infoString_ += "}";  // adding the closing bracked to the JSON string
  iBooker.bookString("HLTGenValInfo", infoString_);

  // booking all histograms
  for (long unsigned int i = 0; i < collectionPath_.size(); i++) {
    std::vector<edm::ParameterSet> histConfigs = histConfigs_;
    for (auto& histConfig : histConfigs) {
      histConfig.addParameter<std::string>("pathSpecificCuts", hltPathSpecificCuts.at(i));
      histConfig.addParameter<std::vector<edm::ParameterSet>>("binnings",
                                                              binnings_);  // passing along the user-defined binnings
    }

    collectionPath_.at(i).bookHists(iBooker, histConfigs, histConfigs2D_);
  }
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
  desc.add<double>("dR2limit", 0.1);
  desc.add<bool>("doOnlyLastFilter", false);

  // input collections, a PSet
  edm::ParameterSetDescription inputCollections;
  inputCollections.add<edm::InputTag>("genParticles", edm::InputTag("genParticles"));
  inputCollections.add<edm::InputTag>("genMET", edm::InputTag("genMetTrue"));
  inputCollections.add<edm::InputTag>("ak4GenJets", edm::InputTag("ak4GenJets"));
  inputCollections.add<edm::InputTag>("ak8GenJets", edm::InputTag("ak8GenJets"));
  inputCollections.add<edm::InputTag>("TrigEvent", edm::InputTag("hltTriggerSummaryAOD"));
  desc.add<edm::ParameterSetDescription>("inputCollections", inputCollections);

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

// this method handles the different object types and collections that can be used for efficiency calculation
std::vector<HLTGenValObject> HLTGenValSource::getObjectCollection(const edm::Event& iEvent) {
  std::vector<HLTGenValObject> objects;  // the vector of objects to be filled

  // handle object type
  std::vector<std::string> implementedGenParticles = {"ele", "pho", "mu", "tau"};
  if (std::find(implementedGenParticles.begin(), implementedGenParticles.end(), objType_) !=
      implementedGenParticles.end()) {
    objects = getGenParticles(iEvent);
  } else if (objType_ == "AK4jet") {  // ak4 jets, using the ak4GenJets collection
    const auto& genJets = iEvent.getHandle(ak4genJetToken_);
    for (size_t i = 0; i < genJets->size(); i++) {
      const reco::GenJet p = (*genJets)[i];
      objects.emplace_back(p);
    }
  } else if (objType_ == "AK8jet") {  // ak8 jets, using the ak8GenJets collection
    const auto& genJets = iEvent.getHandle(ak8genJetToken_);
    for (size_t i = 0; i < genJets->size(); i++) {
      const reco::GenJet p = (*genJets)[i];
      objects.emplace_back(p);
    }
  } else if (objType_ == "AK4HT") {  // ak4-based HT, using the ak4GenJets collection
    const auto& genJets = iEvent.getHandle(ak4genJetToken_);
    if (!genJets->empty()) {
      double HTsum = 0.;
      for (const auto& genJet : *genJets) {
        if (genJet.pt() > 30 && std::abs(genJet.eta()) < 2.5)
          HTsum += genJet.pt();
      }
      if (HTsum > 0)
        objects.emplace_back(reco::Candidate::PolarLorentzVector(HTsum, 0, 0, 0));
    }
  } else if (objType_ == "AK8HT") {  // ak8-based HT, using the ak8GenJets collection
    const auto& genJets = iEvent.getHandle(ak8genJetToken_);
    if (!genJets->empty()) {
      double HTsum = 0.;
      for (const auto& genJet : *genJets) {
        if (genJet.pt() > 200 && std::abs(genJet.eta()) < 2.5)
          HTsum += genJet.pt();
      }
      if (HTsum > 0)
        objects.emplace_back(reco::Candidate::PolarLorentzVector(HTsum, 0, 0, 0));
    }
  } else if (objType_ == "MET") {  // MET, using genMET
    const auto& genMET = iEvent.getHandle(genMETToken_);
    if (!genMET->empty()) {
      auto genMETpt = (*genMET)[0].pt();
      objects.emplace_back(reco::Candidate::PolarLorentzVector(genMETpt, 0, 0, 0));
    }
  } else
    throw cms::Exception("InputError") << "Generator-level validation is not available for type " << objType_ << ".\n"
                                       << "Please check for a potential spelling error.\n";

  return objects;
}

// in case of GenParticles, a subset of the entire collection needs to be chosen
std::vector<HLTGenValObject> HLTGenValSource::getGenParticles(const edm::Event& iEvent) {
  std::vector<HLTGenValObject> objects;  // vector to be filled

  const auto& genParticles = iEvent.getHandle(genParticleToken_);  // getting all GenParticles

  // we need to ge the ID corresponding to the desired GenParticle type
  int pdgID = -1;  // setting to -1 should not be needed, but prevents the compiler warning :)
  if (objType_ == "ele")
    pdgID = 11;
  else if (objType_ == "pho")
    pdgID = 22;
  else if (objType_ == "mu")
    pdgID = 13;
  else if (objType_ == "tau")
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
      if ((objType_ == "ele") || (objType_ == "pho"))
        objects.emplace_back(getLastCopyPreFSR(p));
      else if ((objType_ == "mu") || (objType_ == "tau"))
        objects.emplace_back(getLastCopy(p));
    }
  }

  return objects;
}

// function returning the last GenParticle in a decay chain before FSR
reco::GenParticle HLTGenValSource::getLastCopyPreFSR(reco::GenParticle part) {
  const auto& daughters = part.daughterRefVector();
  if (daughters.size() == 1 && daughters.at(0)->pdgId() == part.pdgId())
    return getLastCopyPreFSR(*daughters.at(0).get());  // recursion, whooo
  else
    return part;
}

// function returning the last GenParticle in a decay chain
reco::GenParticle HLTGenValSource::getLastCopy(reco::GenParticle part) {
  for (const auto& daughter : part.daughterRefVector()) {
    if (daughter->pdgId() == part.pdgId())
      return getLastCopy(*daughter.get());
  }
  return part;
}

//define this as a framework plug-in
DEFINE_FWK_MODULE(HLTGenValSource);

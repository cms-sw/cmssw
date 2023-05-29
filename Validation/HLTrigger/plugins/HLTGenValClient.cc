//********************************************************************************
//
//  Description:
//    DQM histogram post processor for the HLT Gen validation source module
//    Given a folder name, this module will find histograms before and after
//    HLT filters and produce efficiency histograms from these.
//    The structure of this model is strongly inspired by the DQMGenericClient,
//    replacing most user input parameters by the automatic parsing of the given directory.
//
//  Author: Finn Labe, UHH, Jul. 2022
//          Inspired by DQMGenericClient from Junghwan Goh - SungKyunKwan University
//********************************************************************************

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <TH1.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TClass.h>
#include <TString.h>
#include <TPRegexp.h>
#include <TDirectory.h>
#include <TEfficiency.h>

#include <set>
#include <cmath>
#include <string>
#include <vector>
#include <climits>
#include <boost/tokenizer.hpp>

class HLTGenValClient : public DQMEDHarvester {
public:
  HLTGenValClient(const edm::ParameterSet& pset);
  ~HLTGenValClient() override{};

  void dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                             DQMStore::IGetter& igetter,
                             const edm::LuminosityBlock& lumiSeg,
                             const edm::EventSetup& c) override;
  void dqmEndRun(DQMStore::IBooker&, DQMStore::IGetter&, edm::Run const&, edm::EventSetup const&) override;
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override{};

  struct EfficOption {
    std::string name, title;
    std::string numerator, denominator;
  };

  void computeEfficiency(DQMStore::IBooker& ibooker,
                         DQMStore::IGetter& igetter,
                         const std::string& dirName,
                         const std::string& efficMEName,
                         const std::string& efficMETitle,
                         const std::string& numeratorMEName,
                         const std::string& denominatorMEName);

private:
  TPRegexp metacharacters_;
  TPRegexp nonPerlWildcard_;
  unsigned int verbose_;
  bool runOnEndLumi_;
  bool runOnEndJob_;
  bool makeGlobalEffPlot_;
  bool isWildcardUsed_;

  DQMStore* theDQM;
  std::vector<std::string> subDirs_;
  std::string outputFileName_;

  std::vector<EfficOption> efficOptions_;

  const std::string separator_ = "__";

  void makeAllPlots(DQMStore::IBooker&, DQMStore::IGetter&);

  void findAllSubdirectories(DQMStore::IBooker& ibooker,
                             DQMStore::IGetter& igetter,
                             std::string dir,
                             std::set<std::string>* myList,
                             const TString& pattern);

  void genericEff(TH1* denom, TH1* numer, MonitorElement* efficiencyHist);
};

HLTGenValClient::HLTGenValClient(const edm::ParameterSet& pset)
    : metacharacters_("[\\^\\$\\.\\*\\+\\?\\|\\(\\)\\{\\}\\[\\]]"), nonPerlWildcard_("\\w\\*|^\\*") {
  boost::escaped_list_separator<char> commonEscapes("\\", " \t", "\'");

  verbose_ = pset.getUntrackedParameter<unsigned int>("verbose", 0);
  runOnEndLumi_ = pset.getUntrackedParameter<bool>("runOnEndLumi", false);
  runOnEndJob_ = pset.getUntrackedParameter<bool>("runOnEndJob", true);
  makeGlobalEffPlot_ = pset.getUntrackedParameter<bool>("makeGlobalEffienciesPlot", true);

  outputFileName_ = pset.getUntrackedParameter<std::string>("outputFileName", "");
  subDirs_ = pset.getUntrackedParameter<std::vector<std::string>>("subDirs");

  isWildcardUsed_ = false;
}

void HLTGenValClient::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                                            DQMStore::IGetter& igetter,
                                            const edm::LuminosityBlock& lumiSeg,
                                            const edm::EventSetup& c) {
  if (runOnEndLumi_) {
    makeAllPlots(ibooker, igetter);
  }
}

void HLTGenValClient::dqmEndRun(DQMStore::IBooker& ibooker,
                                DQMStore::IGetter& igetter,
                                edm::Run const&,
                                edm::EventSetup const&) {
  // Create new MEs in endRun, even though we are requested to do it in endJob.
  // This gives the QTests a chance to run, before summaries are created in
  // endJob. The negative side effect is that we cannot run the GenericClient
  // for plots produced in Harvesting, but that seems rather rare.
  //
  // It is important that this is still save in the presence of multiple runs,
  // first because in multi-run harvesting, we accumulate statistics over all
  // runs and have full statistics at the endRun of the last run, and second,
  // because we set the efficiencyFlag so any further aggregation should produce
  // correct results. Also, all operations should be idempotent; running them
  // more than once does no harm.

  theDQM = edm::Service<DQMStore>().operator->();

  if (runOnEndJob_) {
    makeAllPlots(ibooker, igetter);
  }

  if (!outputFileName_.empty())
    theDQM->save(outputFileName_);
}

// the main method that creates the plots
void HLTGenValClient::makeAllPlots(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  // Process wildcard in the sub-directory
  std::set<std::string> subDirSet;
  for (auto& subDir : subDirs_) {
    if (subDir[subDir.size() - 1] == '/')
      subDir.erase(subDir.size() - 1);

    if (TString(subDir).Contains(metacharacters_)) {
      isWildcardUsed_ = true;

      const std::string::size_type shiftPos = subDir.rfind('/');
      const std::string searchPath = subDir.substr(0, shiftPos);
      const std::string pattern = subDir.substr(shiftPos + 1, subDir.length());

      findAllSubdirectories(ibooker, igetter, searchPath, &subDirSet, pattern);

    } else {
      subDirSet.insert(subDir);
    }
  }

  // loop through all sub-directories
  // from the current implementation of the HLTGenValSource, we expect all histograms in a single directory
  // however, this module is also capable of handling sub-directories, if needed
  for (std::set<std::string>::const_iterator iSubDir = subDirSet.begin(); iSubDir != subDirSet.end(); ++iSubDir) {
    const std::string& dirName = *iSubDir;

    // construct efficiency options automatically from systematically names histograms^
    const auto contents = igetter.getAllContents(dirName);
    for (const auto& content : contents) {
      // splitting the input string
      std::string name = content->getName();
      std::vector<std::string> seglist;
      size_t pos = 0;
      std::string token;
      while ((pos = name.find(separator_)) != std::string::npos) {
        token = name.substr(0, pos);
        seglist.push_back(token);
        name.erase(0, pos + separator_.length());
      }
      seglist.push_back(name);

      if (seglist.size() == 4 ||
          seglist.size() ==
              5) {  // this should be the only "proper" files we want to look at. 5 means that a custom tag was set!
        if (seglist.at(2) == "GEN")
          continue;  // this is the "before" hist, we won't create an effiency from this alone

        // if a fifth entry exists, it is expected to be the custom tag
        std::string tag = "";
        if (seglist.size() == 5)
          tag = seglist.at(4);

        // first we determing whether we have the 1D or 2D case
        if (seglist.at(3).rfind("2D", 0) == 0) {
          // 2D case
          EfficOption opt;
          opt.name = seglist.at(0) + separator_ + seglist.at(1) + separator_ + seglist.at(2) + separator_ +
                     seglist.at(3) + separator_ + "eff";  // efficiency histogram name
          opt.title = seglist.at(0) + " " + seglist.at(1) + " " + seglist.at(2) + " " + seglist.at(3) +
                      " efficiency";           // efficiency histogram title
          opt.numerator = content->getName();  // numerator histogram (after a filter)
          opt.denominator = seglist.at(0) + separator_ + seglist.at(1) + separator_ + "GEN" + separator_ +
                            seglist.at(3);  // denominator histogram (before all filters)

          efficOptions_.push_back(opt);

        } else {
          // 1D case
          EfficOption opt;
          opt.name = seglist.at(0) + separator_ + seglist.at(1) + separator_ + seglist.at(2) + separator_ +
                     seglist.at(3) + separator_ + "eff";  // efficiency histogram name
          opt.title = seglist.at(0) + " " + seglist.at(1) + " " + seglist.at(2) + " " + seglist.at(3) +
                      " efficiency";           // efficiency histogram title
          opt.numerator = content->getName();  // numerator histogram (after a filter)
          opt.denominator = seglist.at(0) + separator_ + seglist.at(1) + separator_ + "GEN" + separator_ +
                            seglist.at(3);  // denominator histogram (before all filters)

          // propagating the custom tag to the efficiency
          if (!tag.empty()) {
            opt.name += separator_ + tag;
            opt.title += " " + tag;
            opt.denominator += separator_ + tag;
          }

          efficOptions_.push_back(opt);
        }
      }
    }

    // now that we have all EfficOptions, we create the histograms
    for (const auto& efficOption : efficOptions_) {
      computeEfficiency(ibooker,
                        igetter,
                        dirName,
                        efficOption.name,
                        efficOption.title,
                        efficOption.numerator,
                        efficOption.denominator);
    }
  }
}

// main method of efficiency computation, called once for each EfficOption
void HLTGenValClient::computeEfficiency(DQMStore::IBooker& ibooker,
                                        DQMStore::IGetter& igetter,
                                        const std::string& dirName,
                                        const std::string& efficMEName,
                                        const std::string& efficMETitle,
                                        const std::string& numeratorMEName,
                                        const std::string& denominatorMEName) {
  // checking if directory exists
  if (!igetter.dirExists(dirName)) {
    if (verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_)) {
      edm::LogError("HLTGenValClient") << "computeEfficiency() : "
                                       << "Cannot find sub-directory " << dirName << std::endl;
    }
    return;
  }

  ibooker.cd();

  // getting input MEs
  HLTGenValClient::MonitorElement* denominatorME = igetter.get(dirName + "/" + denominatorMEName);
  HLTGenValClient::MonitorElement* numeratorME = igetter.get(dirName + "/" + numeratorMEName);

  // checking of input MEs exist
  if (!denominatorME) {
    if (verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_)) {
      edm::LogError("HLTGenValClient") << "computeEfficiency() : "
                                       << "No denominator-ME '" << denominatorMEName << "' found\n";
    }
    return;
  }
  if (!numeratorME) {
    if (verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_)) {
      edm::LogError("HLTGenValClient") << "computeEfficiency() : "
                                       << "No numerator-ME '" << numeratorMEName << "' found\n";
    }
    return;
  }

  // Treat everything as the base class, TH1
  TH1* hDenominator = denominatorME->getTH1();
  TH1* hNumerator = numeratorME->getTH1();

  // check if TH1 extraction has succeeded
  if (!hDenominator || !hNumerator) {
    if (verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_)) {
      edm::LogError("HLTGenValClient") << "computeEfficiency() : "
                                       << "Cannot create TH1 from ME\n";
    }
    return;
  }

  // preparing efficiency output path and name
  std::string efficDir = dirName;
  std::string newEfficMEName = efficMEName;
  std::string::size_type shiftPos;
  if (std::string::npos != (shiftPos = efficMEName.rfind('/'))) {
    efficDir += "/" + efficMEName.substr(0, shiftPos);
    newEfficMEName.erase(0, shiftPos + 1);
  }
  ibooker.setCurrentFolder(efficDir);

  // creating the efficiency MonitorElement
  HLTGenValClient::MonitorElement* efficME = nullptr;

  // We need to know what kind of TH1 we have
  // That information is obtained from the class name of the hDenominator
  // Then we use the appropriate booking function
  TH1* efficHist = static_cast<TH1*>(hDenominator->Clone(newEfficMEName.c_str()));
  efficHist->SetDirectory(nullptr);
  efficHist->SetTitle(efficMETitle.c_str());
  TClass* myHistClass = efficHist->IsA();
  std::string histClassName = myHistClass->GetName();
  if (histClassName == "TH1F") {
    efficME = ibooker.book1D(newEfficMEName, (TH1F*)efficHist);
  } else if (histClassName == "TH2F") {
    efficME = ibooker.book2D(newEfficMEName, (TH2F*)efficHist);
  } else if (histClassName == "TH3F") {
    efficME = ibooker.book3D(newEfficMEName, (TH3F*)efficHist);
  }
  delete efficHist;

  // checking whether efficME was succesfully created
  if (!efficME) {
    edm::LogError("HLTGenValClient") << "computeEfficiency() : "
                                     << "Cannot book effic-ME from the DQM\n";
    return;
  }

  // actually calculating the efficiency and filling the ME
  genericEff(hDenominator, hNumerator, efficME);
  efficME->setEntries(denominatorME->getEntries());

  // Putting total efficiency in "GLobal efficiencies" histogram
  if (makeGlobalEffPlot_) {
    // getting global efficiency ME
    HLTGenValClient::MonitorElement* globalEfficME = igetter.get(efficDir + "/globalEfficiencies");
    if (!globalEfficME)  // in case it does not exist yet, we create it
      globalEfficME = ibooker.book1D("globalEfficiencies", "Global efficiencies", 1, 0, 1);
    if (!globalEfficME) {  // error handling in case creation failed
      edm::LogError("HLTGenValClient") << "computeEfficiency() : "
                                       << "Cannot book globalEffic-ME from the DQM\n";
      return;
    }
    globalEfficME->setEfficiencyFlag();

    // extracting histogram
    TH1F* hGlobalEffic = globalEfficME->getTH1F();
    if (!hGlobalEffic) {
      edm::LogError("HLTGenValClient") << "computeEfficiency() : "
                                       << "Cannot create TH1F from ME, globalEfficME\n";
      return;
    }

    // getting total counts
    const float nDenominatorAll = hDenominator->GetEntries();
    const float nNumeratorAll = hNumerator->GetEntries();

    // calculating total efficiency
    float efficAll = 0;
    float errorAll = 0;
    efficAll = nDenominatorAll ? nNumeratorAll / nDenominatorAll : 0;
    errorAll = nDenominatorAll && efficAll < 1 ? sqrt(efficAll * (1 - efficAll) / nDenominatorAll) : 0;

    // Filling the histogram bin
    const int iBin = hGlobalEffic->Fill(newEfficMEName.c_str(), 0);
    hGlobalEffic->SetBinContent(iBin, efficAll);
    hGlobalEffic->SetBinError(iBin, errorAll);
  }
}

// method to find all subdirectories of the given directory
// goal is to fill myList with paths to all subdirectories
void HLTGenValClient::findAllSubdirectories(DQMStore::IBooker& ibooker,
                                            DQMStore::IGetter& igetter,
                                            std::string dir,
                                            std::set<std::string>* myList,
                                            const TString& _pattern = TString("")) {
  TString patternTmp = _pattern;

  // checking if directory exists
  if (!igetter.dirExists(dir)) {
    edm::LogError("HLTGenValClient") << " HLTGenValClient::findAllSubdirectories ==> Missing folder " << dir << " !!!";
    return;
  }

  // replacing wildcards
  if (patternTmp != "") {
    if (patternTmp.Contains(nonPerlWildcard_))
      patternTmp.ReplaceAll("*", ".*");
    TPRegexp regexp(patternTmp);
    ibooker.cd(dir);
    std::vector<std::string> foundDirs = igetter.getSubdirs();
    for (const auto& iDir : foundDirs) {
      TString dirName = iDir.substr(iDir.rfind('/') + 1, iDir.length());
      if (dirName.Contains(regexp))
        findAllSubdirectories(ibooker, igetter, iDir, myList);
    }
  } else if (igetter.dirExists(dir)) {
    // we have found a subdirectory - adding it to the list
    myList->insert(dir);

    // moving into the found subdirectory and recursively continue
    ibooker.cd(dir);
    findAllSubdirectories(ibooker, igetter, dir, myList, "*");

  } else {
    // error handling in case found directory does not exist
    edm::LogError("HLTGenValClient") << "Trying to find sub-directories of " << dir << " failed because " << dir
                                     << " does not exist";
  }
  return;
}

// efficiency calculation from two histograms
void HLTGenValClient::genericEff(TH1* denom, TH1* numer, MonitorElement* efficiencyHist) {
  // looping over all bins. Up to three dimentions can be handled
  // in case of less dimensions, the inner for loops are excecuted only once
  for (int iBinX = 1; iBinX < denom->GetNbinsX() + 1; iBinX++) {
    for (int iBinY = 1; iBinY < denom->GetNbinsY() + 1; iBinY++) {
      for (int iBinZ = 1; iBinZ < denom->GetNbinsZ() + 1; iBinZ++) {
        int globalBinNum = denom->GetBin(iBinX, iBinY, iBinZ);

        // getting numerator and denominator values
        float numerVal = numer->GetBinContent(globalBinNum);
        float denomVal = denom->GetBinContent(globalBinNum);

        // calculating effiency
        float effVal = 0;
        effVal = denomVal ? numerVal / denomVal : 0;

        // calculating error
        float errVal = 0;
        errVal = (denomVal && (effVal <= 1)) ? sqrt(effVal * (1 - effVal) / denomVal) : 0;

        // inserting value into the efficiency histogram
        efficiencyHist->setBinContent(globalBinNum, effVal);
        efficiencyHist->setBinError(globalBinNum, errVal);
        efficiencyHist->setEfficiencyFlag();
      }
    }
  }
}

DEFINE_FWK_MODULE(HLTGenValClient);

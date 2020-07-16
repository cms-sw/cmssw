#include "TROOT.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TTree.h"
//#include "Riosteram.h"
#include "TPDF.h"
#include "TF1.h"
#include "TGraphAsymmErrors.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <map>
#include <cctype>
#include <boost/tokenizer.hpp>

using namespace std;
using namespace boost;

class PlotManager {
public:
  PlotManager(const string& inFileName, const string& outFileName);
  //  PlotManager(std::vector<std::string> inFileNames, std::string outFileName);
  ~PlotManager();

  void processCommand(istream& in);

protected:
  bool saveEfficiency(const string& histName,
                      const string& histTitle,
                      const string& numeHistName,
                      const string& denoHistName);
  bool saveBayesEfficiency(const string& graphName,
                           const string& graphTitle,
                           const string& numeHistName,
                           const string& denoHistName);
  bool saveFakeRate(const string& histName,
                    const string& histTitle,
                    const string& numeHistName,
                    const string& denoHistName);
  bool saveResolution(const string& histName,
                      const string& histTitle,
                      const string& srcHistName,
                      const char sliceDirection = 'Y');
  bool dumpObject(const string& objName, const string& objTitle, const string& srcObjName);

protected:
  bool setSrcFile(const string& fileName);
  bool setOutFile(const string& fileName);

protected:
  bool isSetup_;

  TFile* theSrcFile_;
  TFile* theOutFile_;
};

// super-mkdir : recursively run mkdir in root file
TDirectory* mkdirs(TDirectory* dir, string path);
string dirname(const string& path);
string basename(const string& path);

int main(int argc, const char* argv[]) {
  string srcFileName, outFileName;
  if (argc > 2) {
    srcFileName = argv[1];
    outFileName = argv[2];
  } else {
    cout << "Usage : " << argv[0] << " sourceFile.root outputFile.root < commands.txt" << endl;
    cout << "Usage : " << argv[0] << " sourceFile.root outputFile.root" << endl;
    return 0;
  }

  PlotManager plotMan(srcFileName, outFileName);
  plotMan.processCommand(cin);
}

void PlotManager::processCommand(istream& in) {
  string buffer, cmd;
  while (!in.eof()) {
    buffer.clear();
    cmd.clear();
    getline(in, buffer);

    // Extract command from buffer
    stringstream ss(buffer);
    ss >> cmd;
    if (cmd.empty() || cmd[0] == '#')
      continue;
    int (*pf)(int) = std::toupper;
    transform(cmd.begin(), cmd.end(), cmd.begin(), pf);

    buffer.erase(0, cmd.size() + 1);

    typedef escaped_list_separator<char> elsc;
    tokenizer<elsc> tokens(buffer, elsc("\\", " \t", "\""));

    vector<tokenizer<elsc>::value_type> args;

    for (tokenizer<elsc>::const_iterator tok_iter = tokens.begin(); tok_iter != tokens.end(); ++tok_iter) {
      args.push_back(*tok_iter);
    }

    if (cmd == "EFFICIENCY" && args.size() >= 4) {
      if (!saveEfficiency(args[0], args[1], args[2], args[3])) {
        cerr << "Error : cannot make efficiency plot" << endl;
      }
      continue;
    }

    if (cmd == "BAYESIANEFFICIENCY" && args.size() >= 4) {
      if (!saveBayesEfficiency(args[0], args[1], args[2], args[3])) {
        cerr << "Error : cannot make bayesian efficiency plot" << endl;
      }
      continue;
    }

    if (cmd == "FAKERATE" && args.size() == 4) {
      if (!saveFakeRate(args[0], args[1], args[2], args[3])) {
        cerr << "Error : cannot make fakerate plot" << endl;
      }
      continue;
    }

    if (cmd == "RESOLUTIONX" && args.size() == 3) {
      if (!saveResolution(args[0], args[1], args[2], 'X')) {
        cerr << "Error : cannot make resolution-X plot" << endl;
      }
      continue;
    }

    if (cmd == "RESOLUTIONY" && args.size() == 3) {
      if (!saveResolution(args[0], args[1], args[2], 'Y')) {
        cerr << "Error : cannot make resolution-Y plot" << endl;
      }
      continue;
    }

    if (cmd == "DUMP" && args.size() == 3) {
      if (!dumpObject(args[0], args[1], args[2])) {
        cerr << "Error : cannot copy histogram" << endl;
      }
      continue;
    }

    cerr << "Unknown command <" << cmd << ">" << endl;
  }
}

PlotManager::PlotManager(const string& srcFileName, const string& outFileName) {
  // Set default values
  theSrcFile_ = nullptr;
  theOutFile_ = nullptr;

  isSetup_ = (setSrcFile(srcFileName) && setOutFile(outFileName));
}

PlotManager::~PlotManager() {
  if (theSrcFile_) {
    theSrcFile_->Close();
  }
  if (theOutFile_) {
    theOutFile_->Write();
    theOutFile_->Close();
  }
}

bool PlotManager::setSrcFile(const string& fileName) {
  if (theSrcFile_)
    theSrcFile_->Close();

  string pwd(gDirectory->GetPath());

  theSrcFile_ = new TFile(fileName.c_str());
  if (!theSrcFile_ || theSrcFile_->IsZombie())
    return false;

  gDirectory->cd(pwd.c_str());
  return true;
}

bool PlotManager::setOutFile(const string& fileName) {
  if (theOutFile_)
    theOutFile_->Close();

  string pwd(gDirectory->GetPath());

  theOutFile_ = new TFile(fileName.c_str(), "RECREATE");
  if (!theOutFile_ || theOutFile_->IsZombie())
    return false;

  gDirectory->cd(pwd.c_str());
  return true;
}

bool PlotManager::saveEfficiency(const string& histName,
                                 const string& histTitle,
                                 const string& numeHistName,
                                 const string& denoHistName) {
  if (!isSetup_)
    return false;

  TH1F* numeHist = dynamic_cast<TH1F*>(theSrcFile_->Get(numeHistName.c_str()));
  TH1F* denoHist = dynamic_cast<TH1F*>(theSrcFile_->Get(denoHistName.c_str()));

  // Check validity of objects
  if (numeHist == nullptr) {
    cerr << "Cannot get object : " << numeHistName << endl;
    return false;
  }
  if (denoHist == nullptr) {
    cerr << "Cannot get object : " << denoHistName << endl;
    return false;
  }

  if (!numeHist->IsA()->InheritsFrom("TH1") || !denoHist->IsA()->InheritsFrom("TH1") ||
      numeHist->IsA()->InheritsFrom("TH2") || denoHist->IsA()->InheritsFrom("TH2")) {
    return false;
  }

  // Check bin size
  if (numeHist->GetNbinsX() != denoHist->GetNbinsX()) {
    cerr << "Bin size of two histograms are not same" << endl;
    return false;
  }

  // Push to base directory
  string pwd(gDirectory->GetPath());

  string newHistPath = dirname(histName);
  string newHistName = basename(histName);

  if (newHistPath.empty()) {
    theOutFile_->cd();
  } else if (theOutFile_->cd(newHistPath.c_str()) == kFALSE) {
    cout << "Cannot find directory, do mkdirs : " << newHistPath << endl;
    mkdirs(theOutFile_, newHistPath)->cd();
  }

  // Create new histogram
  TH1F* effHist = dynamic_cast<TH1F*>(numeHist->Clone(newHistName.c_str()));

  // effHist->Divide(denoHist);
  // Set the error to binomial statistics
  int nBinsX = effHist->GetNbinsX();
  for (int bin = 1; bin <= nBinsX; bin++) {
    float nNume = numeHist->GetBinContent(bin);
    float nDeno = denoHist->GetBinContent(bin);
    float eff = nDeno == 0 ? 0 : nNume / nDeno;
    float err = 0;
    if (nDeno != 0 && eff <= 1) {
      err = sqrt(eff * (1 - eff) / nDeno);
    }
    effHist->SetBinContent(bin, eff);
    effHist->SetBinError(bin, err);
  }

  // Cosmetics
  //effHist->SetName(newHistName.c_str());
  effHist->SetTitle(histTitle.c_str());
  effHist->SetMinimum(0.0);
  effHist->SetMaximum(1.0);
  effHist->GetXaxis()->SetTitle(numeHist->GetXaxis()->GetTitle());
  effHist->GetYaxis()->SetTitle("Efficiency");

  // Save histogram
  effHist->Write();

  // Pop directory
  gDirectory->cd(pwd.c_str());

  return true;
}

bool PlotManager::saveBayesEfficiency(const string& graphName,
                                      const string& graphTitle,
                                      const string& numeHistName,
                                      const string& denoHistName) {
  if (!isSetup_)
    return false;

  TH1F* numeHist = (TH1F*)(theSrcFile_->Get(numeHistName.c_str()));
  TH1F* denoHist = (TH1F*)(theSrcFile_->Get(denoHistName.c_str()));

  // Check validity of objects
  if (numeHist == nullptr || denoHist == nullptr) {
    cerr << "Cannot get object : " << graphName << endl;
    return false;
  }

  if (!numeHist->IsA()->InheritsFrom("TH1") || !denoHist->IsA()->InheritsFrom("TH1") ||
      numeHist->IsA()->InheritsFrom("TH2") || denoHist->IsA()->InheritsFrom("TH2")) {
    return false;
  }

  // Check bin size
  if (numeHist->GetNbinsX() != denoHist->GetNbinsX()) {
    cerr << "Bin size of two histograms are not same" << endl;
    return false;
  }

  // Push to base directory
  string pwd(gDirectory->GetPath());

  string newGraphPath = dirname(graphName);
  string newGraphName = basename(graphName);

  if (newGraphPath.empty()) {
    theOutFile_->cd();
  } else if (theOutFile_->cd(newGraphPath.c_str()) == kFALSE) {
    cout << "Cannot find directory, do mkdirs" << endl;
    mkdirs(theOutFile_, newGraphPath)->cd();
  }

  // Create new TGraphAsymmErrors
  TGraphAsymmErrors* effGraph = new TGraphAsymmErrors(numeHist, denoHist);

  // Cosmetics
  effGraph->SetName(newGraphName.c_str());
  effGraph->SetTitle(graphTitle.c_str());
  effGraph->SetMinimum(0.8);
  effGraph->SetMaximum(1.0);
  effGraph->GetXaxis()->SetTitle(numeHist->GetXaxis()->GetTitle());
  effGraph->GetYaxis()->SetTitle("Efficiency");

  // Save histogram
  effGraph->Write();

  // Pop directory
  gDirectory->cd(pwd.c_str());

  return true;
}

bool PlotManager::saveFakeRate(const string& histName,
                               const string& histTitle,
                               const string& numeHistName,
                               const string& denoHistName) {
  if (!isSetup_)
    return false;

  TH1F* numeHist = (TH1F*)(theSrcFile_->Get(numeHistName.c_str()));
  TH1F* denoHist = (TH1F*)(theSrcFile_->Get(denoHistName.c_str()));

  // Check validity of objects
  if (numeHist == nullptr || denoHist == nullptr) {
    cerr << "Cannot get object : " << histName << endl;
    return false;
  }

  if (!numeHist->IsA()->InheritsFrom("TH1") || !denoHist->IsA()->InheritsFrom("TH1") ||
      numeHist->IsA()->InheritsFrom("TH2") || denoHist->IsA()->InheritsFrom("TH2")) {
    return false;
  }

  // Check bin size
  if (numeHist->GetNbinsX() != denoHist->GetNbinsX()) {
    cerr << "Bin size of two histograms are not same" << endl;
    return false;
  }

  // Push to base directory
  string pwd(gDirectory->GetPath());

  string newHistPath = dirname(histName);
  string newHistName = basename(histName);

  if (newHistPath.empty()) {
    theOutFile_->cd();
  } else if (theOutFile_->cd(newHistPath.c_str()) == kFALSE) {
    cout << "Cannot find directory, do mkdirs" << endl;
    mkdirs(theOutFile_, newHistPath)->cd();
  }

  // Create new histogram
  TH1F* fakeHist = dynamic_cast<TH1F*>(numeHist->Clone());

  // effHist->Divide(denoHist);
  // Set the error to binomial statistics
  int nBinsX = fakeHist->GetNbinsX();
  for (int bin = 1; bin <= nBinsX; bin++) {
    float nNume = numeHist->GetBinContent(bin);
    float nDeno = denoHist->GetBinContent(bin);
    float fakeRate = nDeno == 0 ? 0 : 1.0 - nNume / nDeno;
    float err = 0;
    if (nDeno != 0 && fakeRate <= 1) {
      err = sqrt(fakeRate * (1 - fakeRate) / nDeno);
    }
    fakeHist->SetBinContent(bin, fakeRate);
    fakeHist->SetBinError(bin, err);
  }

  // Cosmetics
  fakeHist->SetName(newHistName.c_str());
  fakeHist->SetTitle(histTitle.c_str());
  fakeHist->SetMinimum(0.8);
  fakeHist->SetMaximum(1.0);
  fakeHist->GetXaxis()->SetTitle(numeHist->GetXaxis()->GetTitle());
  fakeHist->GetYaxis()->SetTitle("Efficiency");

  // Save histogram
  fakeHist->Write();

  // Pop directory
  gDirectory->cd(pwd.c_str());

  return true;
}

bool PlotManager::saveResolution(const string& histName,
                                 const string& histTitle,
                                 const string& srcHistName,
                                 const char sliceDirection) {
  if (!isSetup_)
    return false;

  TH2F* srcHist = dynamic_cast<TH2F*>(theSrcFile_->Get(srcHistName.c_str()));

  // Check validity of objects
  if (srcHist == nullptr) {
    cerr << "Cannot get object : " << histName << endl;
    return false;
  }

  if (srcHist->IsA()->InheritsFrom("TH2"))
    return false;

  // Push to base directory
  string pwd(gDirectory->GetPath());

  string newHistPath = dirname(histName);
  string newHistName = basename(histName);

  if (newHistPath.empty()) {
    theOutFile_->cd();
  } else if (theOutFile_->cd(newHistPath.c_str()) == kFALSE) {
    cout << "Cannot find directory, do mkdirs" << endl;
    mkdirs(theOutFile_, newHistPath)->cd();
  }

  // Create a function for resolution model
  TF1 gaus("gaus", "gaus");
  gaus.SetParameters(1.0, 0.0, 0.1);
  //gaus.SetRange(yMin, yMax);

  // Do FitSlices.
  if (sliceDirection == 'X')
    srcHist->FitSlicesX(&gaus);
  else
    srcHist->FitSlicesY(&gaus);

  TH1F* meanHist = dynamic_cast<TH1F*>(theOutFile_->Get((srcHistName + "_1").c_str()));
  TH1F* widthHist = dynamic_cast<TH1F*>(theOutFile_->Get((srcHistName + "_2").c_str()));
  TH1F* chi2Hist = dynamic_cast<TH1F*>(theOutFile_->Get((srcHistName + "_chi2").c_str()));

  // Cosmetics
  meanHist->SetName((newHistName + "_Mean").c_str());
  widthHist->SetName((newHistName + "_Width").c_str());
  chi2Hist->SetName((newHistName + "_Chi2").c_str());

  meanHist->SetTitle((histTitle + " Mean").c_str());
  widthHist->SetTitle((histTitle + " Width").c_str());
  chi2Hist->SetTitle((histTitle + " Chi2").c_str());

  meanHist->GetYaxis()->SetTitle("Gaussian mean");
  widthHist->GetYaxis()->SetTitle("Gaussian width");
  chi2Hist->GetYaxis()->SetTitle("Gaussian fit #Chi^{2}");

  // Save histograms
  meanHist->Write();
  widthHist->Write();
  chi2Hist->Write();

  // Pop directory
  gDirectory->cd(pwd.c_str());

  return true;
}

bool PlotManager::dumpObject(const string& objName, const string& objTitle, const string& srcObjName) {
  if (!isSetup_)
    return false;

  // Push to base directory
  string pwd(gDirectory->GetPath());

  string newObjPath = dirname(objName);
  string newObjName = basename(objName);

  if (newObjPath.empty()) {
    theOutFile_->cd();
  } else if (theOutFile_->cd(newObjPath.c_str()) == kFALSE) {
    cout << "Cannot find dir, do mkdirs : " << newObjPath << endl;
    mkdirs(theOutFile_, newObjPath)->cd();
  }

  TNamed* srcObj = dynamic_cast<TNamed*>(theSrcFile_->Get(srcObjName.c_str()));

  if (srcObj == nullptr) {
    cerr << "Cannot get object : " << srcObjName << endl;
    return false;
  }

  TNamed* saveObj = dynamic_cast<TNamed*>(srcObj->Clone(newObjName.c_str()));
  saveObj->SetTitle(objTitle.c_str());

  // Save histogram
  saveObj->Write();

  // Pop directory
  gDirectory->cd(pwd.c_str());

  return true;
}

TDirectory* mkdirs(TDirectory* dir, string path) {
  // Push to directory passed into argument
  string pwd(gDirectory->GetPath());

  while (true) {
    if (path.empty())
      break;

    string::size_type slashPos = path.find_first_of('/');
    if (slashPos != string::npos) {
      string newDirName = path.substr(0, slashPos);
      TDirectory* tmpDir = dir->GetDirectory(newDirName.c_str());
      dir = (tmpDir == nullptr) ? dir->mkdir(newDirName.c_str()) : tmpDir;

      path.erase(0, slashPos + 1);
    } else {
      TDirectory* tmpDir = dir->GetDirectory(path.c_str());
      dir = (tmpDir == nullptr) ? dir->mkdir(path.c_str()) : tmpDir;

      break;
    }
  }

  return dir;
}

string dirname(const string& path) {
  string::size_type slashPos = path.find_last_of('/');
  if (slashPos == string::npos) {
    return "";
  }

  return path.substr(0, slashPos);
}

string basename(const string& path) {
  string::size_type slashPos = path.find_last_of('/');
  if (slashPos == string::npos) {
    return path;
  }

  return path.substr(slashPos + 1, path.size());
}

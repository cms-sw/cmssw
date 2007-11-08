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
//#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>

class PlotManager
{
public:
  PlotManager(std::string inFileName, std::string outFileName);
  //  PlotManager(std::vector<std::string> inFileNames, std::string outFileName);
  ~PlotManager();
  
  void processCommand(std::istream& in);
  
protected:
  bool saveEfficiency(std::string histName, std::string histTitle, 
		      std::string numeHistName, std::string denoHistName);
  bool saveBayesEfficiency(std::string graphName, std::string graphTitle,
			   std::string numeHistName, std::string denoHistName);
  bool saveFakeRate(std::string histName, std::string histTitle,
		    std::string numeHistName, std::string denoHistName);
  bool saveResolution(std::string histName, std::string histTitle, 
		      std::string srcHistName, const char sliceDirection = 'Y');
  bool saveHistogram(std::string histName, std::string histTitle, std::string srcHistName);

protected:
  bool setSrcFile(std::string fileName);
  bool setOutFile(std::string fileName);

protected:
  bool isSetup_;

  TFile* theSrcFile_;
  TFile* theOutFile_;
  
};

// super-mkdir : recursively run mkdir in root file
TDirectory* mkdirs(TDirectory* dir, std::string path);

int main(int argc, const char* argv[])
{
  if ( argc != 4 ) return 1;

  std::string srcFileName = argv[1];
  std::string outFileName = argv[2];
  //  string cmdFileName = argv[3];

  PlotManager plotMan(srcFileName, outFileName);
  plotMan.processCommand(std::cin);
  
}

void PlotManager::processCommand(std::istream& in)
{
  typedef boost::tokenizer< boost::char_separator<char> > Tokenizer;
  boost::char_separator<char> sep(",");
  
  std::string buffer;
  while ( ! in.eof() ) {
    buffer.clear();
    in >> buffer;
    
    //transform(buffer.begin(), buffer.end(), _tolower);
    Tokenizer tokens(buffer, sep);
    
    Tokenizer::const_iterator itk = tokens.begin();
    std::string cmd = *(itk);
    std::vector<std::string> args;
    for(++itk; itk != tokens.end(); ++itk ) { 
      args.push_back(*itk); 
    }
    
    if ( cmd == "eff" ) {
      if ( args.size() != 4 ) {
	std::cerr << "Error : eff,name,title,numeratorName,denominatorName\n";
	continue;
      }
      if ( ! saveEfficiency(args[0], args[1], args[2], args[3]) ) {
	std::cerr << "Error : cannot save efficiency plot (" << args[0] << ")\n";
      }
    }
    else if ( cmd == "effb" ) {
      if ( args.size() != 4 ) {
	std::cerr << "Error : effb,name,title,numeratorName,denominatorName\n";
	continue;
      }
      if ( ! saveBayesEfficiency(args[0], args[1], args[2], args[3]) ) {
	std::cerr << "Error : cannot save bayesian efficiency plot (" << args[0] << ")\n";
      }
    }
    else if ( cmd == "fake" ) {
      if ( args.size() != 4 ) {
	std::cerr << "Error : fake,name,title,numeratorName,denominatorName\n";
	continue;
      }
      if ( ! saveFakeRate(args[0], args[1], args[2], args[3]) ) {
	std::cerr << "Error : cannot save fake rate plot (" << args[0] << ")\n";
      }
    }
    else if ( cmd == "resolx" ) {
      if ( args.size() != 3 ) {
	std::cerr << "Error : resolx,name,title,sourceHistName\n";
	continue;
      }
      if ( ! saveResolution(args[0], args[1], args[2], 'X') ) {
	std::cerr << "Error : cannot save resolution plot (" << args[0] << ")\n";
      }
    }
    else if ( cmd == "resoly" ) {
      if ( args.size() != 3 ) {
	std::cerr << "Error : resoly,name,title,sourceHistName\n";
	continue;
      }
      if ( ! saveResolution(args[0], args[1], args[2], 'Y') ) {
	std::cerr << "Error : cannot save resolution plot (" << args[0] << ")\n";
      }
    }
    else if ( cmd == "hist" ) {
      if ( args.size() != 3 ) {
	std::cerr << "Error : hist,name,title,sourceHistName\n";
	continue;
      }
      if ( ! saveHistogram(args[0], args[1], args[2]) ) {
	std::cerr << "Error : cannot save histogra (" << args[0] << ")\n";
      }
    }
    else {
      std::cerr << "Unknown command\n";
    }
  }
}

PlotManager::PlotManager(std::string srcFileName, std::string outFileName)
{
  // Set default values
  theSrcFile_ = 0;
  theOutFile_ = 0;

  isSetup_ = (setSrcFile(srcFileName) && setOutFile(outFileName));
}

PlotManager::~PlotManager()
{
  if ( theSrcFile_ ) {
    theSrcFile_->Close();
  }
  if ( theOutFile_ ) {
    theOutFile_->Write();
    theOutFile_->Close();
  }
}

bool PlotManager::setSrcFile(std::string fileName)
{
  if ( theSrcFile_ ) theSrcFile_->Close();

  std::string pwd(gDirectory->GetPath());

  theSrcFile_ = new TFile(fileName.c_str());
  if ( ! theSrcFile_ || theSrcFile_->IsZombie() ) return false;

  gDirectory->cd(pwd.c_str());
  return true;
}

bool PlotManager::setOutFile(std::string fileName)
{
  if ( theOutFile_ ) theOutFile_->Close();

  if ( (fileName.size()-fileName.rfind(".root")) != 5 ) {
    fileName += ".root";
  }

  std::string pwd(gDirectory->GetPath());
  
  theOutFile_ = new TFile(fileName.c_str(), "RECREATE");
  if ( !theOutFile_ || theOutFile_->IsZombie() ) return false;

  gDirectory->cd(pwd.c_str());
  return true;
}

bool PlotManager::saveEfficiency(std::string histName, std::string histTitle, 
				 std::string numeHistName, std::string denoHistName)
{
  if ( ! isSetup_ ) return false;

  TH1F* numeHist = (TH1F*)(theSrcFile_->Get(numeHistName.c_str()));
  TH1F* denoHist = (TH1F*)(theSrcFile_->Get(denoHistName.c_str()));
  
  // Check validity of objects
  if ( numeHist == 0 || denoHist == 0 ) return false;
  if ( ! numeHist->IsA()->InheritsFrom("TH1") || 
       ! denoHist->IsA()->InheritsFrom("TH1") ||
       numeHist->IsA()->InheritsFrom("TH2") ||
       denoHist->IsA()->InheritsFrom("TH2") ) {
    return false;
  }

  // Check bin size
  if ( numeHist->GetNbinsX() != denoHist->GetNbinsX() ) {
    return false;
  }

  // Push to base directory
  std::string pwd(gDirectory->GetPath());

  theOutFile_->cd();

  // Create new histogram
  TH1F* effHist = dynamic_cast<TH1F*>(numeHist->Clone());
  
  // effHist->Divide(denoHist);
  // Set the error to binomial statistics
  int nBinsX = effHist->GetNbinsX();
  for(int bin = 1; bin <= nBinsX; bin++) {
    float nNume = numeHist->GetBinContent(bin);
    float nDeno = denoHist->GetBinContent(bin);
    float eff = nDeno==0 ? 0 : nNume/nDeno;
    float err = 0;
    if ( nDeno != 0 && eff <= 1 ) {
      err = sqrt(eff*(1-eff)/nDeno);
    }
    effHist->SetBinContent(bin, eff);
    effHist->SetBinError(bin, err);
  }

  // Cosmetics
  effHist->SetName(histName.c_str());
  effHist->SetTitle(histTitle.c_str());
  effHist->SetMinimum(0.8);
  effHist->SetMaximum(1.0);
  effHist->GetXaxis()->SetTitle(numeHist->GetXaxis()->GetTitle());
  effHist->GetXaxis()->SetTitle("Efficiency");

  // Save histogram
  effHist->Write();

  // Pop directory
  gDirectory->cd(pwd.c_str());
			   
  return true;
}

bool PlotManager::saveBayesEfficiency(std::string graphName, std::string graphTitle,
				 std::string numeHistName, std::string denoHistName)
{
  if ( ! isSetup_ ) return false;
  
  TH1F* numeHist = (TH1F*)(theSrcFile_->Get(numeHistName.c_str()));
  TH1F* denoHist = (TH1F*)(theSrcFile_->Get(denoHistName.c_str()));
  
  // Check validity of objects
  if ( ! numeHist->IsA()->InheritsFrom("TH1") || 
       ! denoHist->IsA()->InheritsFrom("TH1") ||
       numeHist->IsA()->InheritsFrom("TH2") ||
       denoHist->IsA()->InheritsFrom("TH2") ) {
    return false;
  }

  // Check bin size
  if ( numeHist->GetNbinsX() != denoHist->GetNbinsX() ) {
    return false;
  }

  // Push to base directory
  std::string pwd(gDirectory->GetPath());

  theOutFile_->cd();

  // Create new TGraphAsymmErrors
  TGraphAsymmErrors* effGraph = new TGraphAsymmErrors(numeHist, denoHist);
  
  // Cosmetics
  effGraph->SetName(graphName.c_str());
  effGraph->SetTitle(graphTitle.c_str());
  effGraph->SetMinimum(0.8);
  effGraph->SetMaximum(1.0);
  effGraph->GetXaxis()->SetTitle(numeHist->GetXaxis()->GetTitle());
  effGraph->GetXaxis()->SetTitle("Efficiency");

  // Save histogram
  effGraph->Write();

  // Pop directory
  gDirectory->cd(pwd.c_str());
  
  return true;
}

bool PlotManager::saveFakeRate(std::string histName, std::string histTitle,
			       std::string numeHistName, std::string denoHistName)
{
  if ( ! isSetup_ ) return false;
  
  TH1F* numeHist = (TH1F*)(theSrcFile_->Get(numeHistName.c_str()));
  TH1F* denoHist = (TH1F*)(theSrcFile_->Get(denoHistName.c_str()));
  
  // Check validity of objects
  if ( ! numeHist->IsA()->InheritsFrom("TH1") || 
       ! denoHist->IsA()->InheritsFrom("TH1") ||
       numeHist->IsA()->InheritsFrom("TH2") ||
       denoHist->IsA()->InheritsFrom("TH2") ) {
    return false;
  }
  
  // Check bin size
  if ( numeHist->GetNbinsX() != denoHist->GetNbinsX() ) {
    return false;
  }
  
  // Push to base directory
  std::string pwd(gDirectory->GetPath());
  
  theOutFile_->cd();

  // Create new histogram
  TH1F* fakeHist = dynamic_cast<TH1F*>(numeHist->Clone());
  
  // effHist->Divide(denoHist);
  // Set the error to binomial statistics
  int nBinsX = fakeHist->GetNbinsX();
  for(int bin = 1; bin <= nBinsX; bin++) {
    float nNume = numeHist->GetBinContent(bin);
    float nDeno = denoHist->GetBinContent(bin);
    float fakeRate = nDeno==0 ? 0 : 1.0 - nNume/nDeno;
    float err = 0;
    if ( nDeno != 0 && fakeRate <= 1 ) {
      err = sqrt(fakeRate*(1-fakeRate)/nDeno);
    }
    fakeHist->SetBinContent(bin, fakeRate);
    fakeHist->SetBinError(bin, err);
  }

  // Cosmetics
  fakeHist->SetName(histName.c_str());
  fakeHist->SetTitle(histTitle.c_str());
  fakeHist->SetMinimum(0.8);
  fakeHist->SetMaximum(1.0);
  fakeHist->GetXaxis()->SetTitle(numeHist->GetXaxis()->GetTitle());
  fakeHist->GetXaxis()->SetTitle("Efficiency");

  // Save histogram
  fakeHist->Write();

  // Pop directory
  gDirectory->cd(pwd.c_str());
			   
  return true;
}

bool PlotManager::saveResolution(std::string histName, std::string histTitle, 
  			         std::string srcHistName, const char sliceDirection)
{
  if ( ! isSetup_ ) return false;
  
  TH2F* srcHist = (TH2F*)(theSrcFile_->Get(srcHistName.c_str()));
  
  // Check validity
  if ( srcHist->IsA()->InheritsFrom("TH2") ) return false;

  // Push to base directory
  std::string pwd(gDirectory->GetPath());
  theOutFile_->cd();

  // Create a function for resolution model
  TF1 gaus("gaus", "gaus");
  gaus.SetParameters(1.0, 0.0, 0.1);
  //gaus.SetRange(yMin, yMax);

  // Do FitSlices.
  if ( sliceDirection == 'X' ) srcHist->FitSlicesX(&gaus);
  else srcHist->FitSlicesY(&gaus);

  TH1F* meanHist  = (TH1F*)theOutFile_->Get((srcHistName+"_1").c_str());
  TH1F* widthHist = (TH1F*)theOutFile_->Get((srcHistName+"_2").c_str());
  TH1F* chi2Hist  = (TH1F*)theOutFile_->Get((srcHistName+"_chi2").c_str());

  // Cosmetics
  meanHist ->SetName((histName+"_Mean" ).c_str());
  widthHist->SetName((histName+"_Width").c_str());
  chi2Hist ->SetName((histName+"_Chi2" ).c_str());

  meanHist ->SetTitle((histTitle+" Mean" ).c_str());
  widthHist->SetTitle((histTitle+" Width").c_str());
  chi2Hist ->SetTitle((histTitle+" Chi2" ).c_str());

  meanHist ->GetYaxis()->SetTitle("Gaussian mean"        );
  widthHist->GetYaxis()->SetTitle("Gaussian width"       );
  chi2Hist ->GetYaxis()->SetTitle("Gaussian fit #Chi^{2}");

  // Save histograms
  meanHist ->Write();
  widthHist->Write();
  chi2Hist ->Write();

  // Pop directory
  gDirectory->cd(pwd.c_str());

  return true;
}

bool PlotManager::saveHistogram(std::string histName, std::string histTitle,
				std::string srcHistName)
{
  if ( ! isSetup_ ) return false;

  TH1* srcHist = (TH1*)theSrcFile_->Get(srcHistName.c_str());

  // Push to base directory
  std::string pwd(gDirectory->GetPath());
  theOutFile_->cd();

  TH1* saveHist = dynamic_cast<TH1*>(srcHist->Clone());
  saveHist->SetName(histName.c_str());
  saveHist->SetTitle(histName.c_str());

  // Save histogram
  saveHist->Write();

  // Pop directory
  gDirectory->cd(pwd.c_str());

  return true;
}

TDirectory* mkdirs(TDirectory* dir, std::string path)
{
  // Push to directory passed into argument
  std::string pwd(gDirectory->GetPath());

  std::string::size_type slashPos = path.find_first_of('/');
  if ( slashPos != std::string::npos ) {
    TDirectory * newDir = 0;
    std::string newPath = path.substr(0, slashPos);

    newDir = dir->mkdir(newPath.c_str());
    // Pop directory
    gDirectory->cd(pwd.c_str());
    if ( newDir == 0 ) return dir;
    // Go to next subdirectory
    return mkdirs(newDir, path.substr(slashPos+1));
  }
  else {
    // finish recursive-mkdir if there are no remaining subdirectories
    // Pop directory
    gDirectory->cd(pwd.c_str());
    return dir;
  }

  // Nothing. just to hide warning message.
  return NULL;
}

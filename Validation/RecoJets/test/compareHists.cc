#include <iostream>
#include <string>
#include <vector>

#include "TROOT.h"
#include "TFile.h"
#include "TKey.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TPostScript.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TText.h"

std::vector<std::string> getAllKeys (const TDirectory* fDir, const std::string& fClassName) {
  std::cout << "getAllKeys-> " << fDir->GetName() << ", " <<  fClassName << std::endl;
  //  fDir->ls();
  std::vector<std::string> result;
  TIter next (fDir->GetListOfKeys ());
  for (TKey* key = 0; (key = (TKey *) next());) {
    std::cout << "key from list: " << key->GetName()  << '/' << key->GetClassName () << std::endl;
    if (fClassName == key->GetClassName ()) {
      result.push_back (std::string (key->GetName ()));
    } 
  }
  return result;
} 

std::vector<std::string> getAllObjects (const TDirectory* fDir, const std::string& fClassName) {
  std::cout << "getAllObjects-> " << fDir->GetName() << ", " <<  fClassName << std::endl;
  //  fDir->ls();
  std::vector<std::string> result;
  TIter next (fDir->GetList ());
  for (TObject* obj = 0; (obj = (TObject *) next());) {
    std::cout << "name from list: " << obj->GetName()  << '/' << obj->ClassName () << std::endl;
    if (fClassName == obj->ClassName ()) {
      result.push_back (std::string (obj->GetName ()));
    } 
  }
  return result;
} 

TObject* getObject (TDirectory* fDir, const std::vector <std::string>& fObjectName) {
  TObject* result = 0; // nothing so far
  TDirectory* dir = fDir;
  for (unsigned i = 0; i < fObjectName.size (); ++i) {
    dir->GetObject (fObjectName[i].c_str(), result);
    if (result) {
      if (i < fObjectName.size () - 1) {
	dir = (TDirectory*) result;
	result = 0;
      }
    }
    else {
      std::cerr << "getObject-> Can not find (sub)dir/object " << fObjectName[i] << " in directory " << dir->GetName () << std::endl;
      return 0;
    }
  }
  return result;
}

double makeGifHists (TH1* fHist, TH1* fRefHist, TCanvas* fCanvas, const std::string& fPrefix = "") {
  double pv = fHist->KolmogorovTest (fRefHist, "OU");
  // set style
  TPad pad ("pad", "pad", 0, 0, 1, 0.9, 0);
  pad.SetLogy ();
  pad.Draw();

  char buf [1024];
  sprintf (buf, "%s: Kolmogorov Test PV = %5.3f", fPrefix.c_str(), pv);
  TPaveText title (0.3,0.85,0.95, 0.99, buf);
  title.SetFillColor(pv > 0.01 ? 3 : 2);
  TText* t1 = title.AddText (fPrefix.c_str());
  sprintf (buf, "Kolmogorov Test PV = %6.4f", pv);
  TText* t2 = title.AddText (buf);
  // t2->SetTextSize(0.3);
  title.Draw();

  pad.cd();

  fHist->Sumw2 ();
  fHist->Scale (fRefHist->GetSumOfWeights () / fHist->GetSumOfWeights ());

  fHist->SetMarkerStyle (21);
  fHist->SetMarkerSize (0.7);
  fRefHist->SetLineColor (2);
  fRefHist->SetFillColor (42);
  std::string name = fRefHist->GetTitle ();
  int blank = name.rfind (' ');
  if (blank >= 0) name.erase (0, blank+1);
  fHist->SetXTitle (name.c_str());
  fHist->SetTitle ("");

  fRefHist->Draw ();
  fHist->Draw ("e1p,same");
  std::string filename = name + ".gif";
  fCanvas->Print (filename.c_str());
  fCanvas->Update ();
  return pv;
}



int main (int argn, char* argv []) {
  int result = 0; // OK

  if (argn < 5) {
    std::cout << "Usage: " << argv[0] << " <file_name> <reference file_name> <module_name> <description" << std::endl;
    return 1;
  }

  std::string inputFileName (argv[1]);
  std::string refFileName (argv[2]);
  std::string moduleName (argv[3]);
  std::string globalTitle = argv[4];
  std::cout << "Processing file " << inputFileName << std::endl;
  TFile* inputFile = TFile::Open (inputFileName.c_str());
  if (!inputFile) {
    std::cerr << "Cannot open file " << inputFileName << std::endl;
    return 1;
  }
  TDirectory* dirIn = 0;
  std::string workDir = std::string ("DQMData/RecoJetsV/CaloJetTask_") + moduleName; // new format
  inputFile->GetObject (workDir.c_str(), dirIn);
  if (!dirIn) {
    std::cout << "Fall back to old format for file " << inputFileName << std::endl;
    workDir = std::string ("DQMData/CaloJetTask_") + moduleName; // old format
    inputFile->GetObject (workDir.c_str(), dirIn);
    if (!dirIn) {
      std::cerr << "Can't access workDir in file " << inputFileName << std::endl;
      return 1;
    }
  }
  TFile* refFile = TFile::Open (refFileName.c_str());
  if (!refFile) {
    std::cerr << "Cannot open file " << refFileName << std::endl;
    return 1;
  }
  TDirectory* dirRef = 0;
  workDir = std::string ("DQMData/RecoJetsV/CaloJetTask_") + moduleName; // new format
  refFile->GetObject (workDir.c_str(), dirRef);
  if (!dirRef) {
    std::cout << "Fall back to old format for file " << refFileName << std::endl;
    workDir = std::string ("DQMData/CaloJetTask_") + moduleName; // old format
    refFile->GetObject (workDir.c_str(), dirRef);
    if (!dirRef) {
      std::cerr << "Can't access workDir in file " << refFileName << std::endl;
      return 1;
    }
  }
  
  std::vector<std::string> histKeys = getAllKeys (dirIn, "TH1F");
  // output
  gStyle->SetOptStat (kFALSE);
  TCanvas canvas ("Jets","Jets",800,600);
  for (unsigned ihist = 0; ihist < histKeys.size (); ++ihist) {
    TH1* hist = 0;
    dirIn->GetObject (histKeys[ihist].c_str(), hist);
    if (hist) {
      TH1* refhist = 0;
      dirRef->GetObject (histKeys[ihist].c_str(), refhist);
      if (refhist) {
	std::string title = globalTitle;
	double pv = makeGifHists (hist, refhist, &canvas, title);
	std::cout << "pv for hist " << histKeys[ihist] << " is " << pv << std::endl; 
      }
      else {
	std::cerr << "Can not get reference histogram " << histKeys[ihist] << std::endl;
      }
    }
    else {
      std::cerr << "Can not get histogram " << histKeys[ihist] << std::endl;
    }
  }
  return 0;
}

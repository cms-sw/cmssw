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
#include "TGaxis.h"

//
// Methods:
//
// getAllKeys
// getAllObjects
// getObject
// makeGifHists
// makeGifHists2
// makeGifHists3
// makeGifHists4
// main
//

//----------
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

//----------
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

//----------
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

//----------
double makeGifHists (TH1* fHist, TH1* fRefHist, TCanvas* fCanvas, const std::string& fPrefix = "", const std::string& events="", double scalebyevents=0) {
  double pv = fHist->KolmogorovTest (fRefHist, "OU");
  // set style
  TPad pad ("pad", "pad", 0, 0, 1, 0.9, 0);
  pad.SetLogy ();
  pad.Draw();

  char buf [1024];
  sprintf (buf, "%s: Kolmogorov Test PV = %5.3f", fPrefix.c_str(), pv);
  TPaveText title (0.3,0.85,0.95, 0.99, buf);
  title.SetFillColor(pv > 0.01 ? 3 : 2);
  title.AddText (fPrefix.c_str());
  sprintf (buf, "Kolmogorov Test PV = %6.4f", pv);
  title.AddText (buf);
  //TText* t1 = title.AddText (fPrefix.c_str());
  //sprintf (buf, "Kolmogorov Test PV = %6.4f", pv);
  //TText* t2 = title.AddText (buf);
  //t1->SetTextSize(0.3);
  //t2->SetTextSize(0.3);
  title.Draw();

  pad.cd();

  if ( events == 'y') {
    fHist->Scale (scalebyevents);
  }
  else {
    fHist->Sumw2 ();
    fHist->Scale (fRefHist->GetSumOfWeights () / fHist->GetSumOfWeights ());
  }

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

//----------
double makeGifHists2 (TH1* fHist, TH1* fRefHist, TCanvas* fCanvas, const std::string& fPrefix = "", const std::string& events="", double scalebyevents=0) {
  double pv = fHist->KolmogorovTest (fRefHist, "OU");
  // set style
  TPad pad ("pad", "pad", 0, 0, 1, 0.9, 0);
  pad.SetLogy ();
  pad.Draw();

  char buf [1024];
  sprintf (buf, "%s: Kolmogorov Test PV = %5.3f", fPrefix.c_str(), pv);
  TPaveText title (0.3,0.85,0.95, 0.99, buf);
  title.SetFillColor(pv > 0.01 ? 3 : 2);
  title.AddText (fPrefix.c_str());
  sprintf (buf, "Kolmogorov Test PV = %6.4f", pv);
  title.AddText (buf);
  //TText* t1 = title.AddText (fPrefix.c_str());
  //sprintf (buf, "Kolmogorov Test PV = %6.4f", pv);
  //TText* t2 = title.AddText (buf);
  //t1->SetTextSize(0.3);
  //t2->SetTextSize(0.3);
  title.Draw();

  pad.cd();

  // Do not normalize the profile plots
  /*
  if ( events == 'y') {
    fHist->Scale (scalebyevents);
  }
  else {
    fHist->Sumw2 ();
    fHist->Scale (fRefHist->GetSumOfWeights () / fHist->GetSumOfWeights ());
  }
  */

  fHist->SetMarkerStyle (21);
  fHist->SetMarkerSize (0.7);
  fRefHist->SetLineColor (2);
  std::string name = fRefHist->GetTitle ();
  int blank = name.rfind (' ');
  if (blank >= 0) name.erase (0, blank+1);
  fHist->SetXTitle (name.c_str());
  fHist->SetTitle ("");

  fRefHist->Draw ();
  fHist->Draw ("e1p,same");
  std::string filename = name + "_logy.gif";
  fCanvas->Print (filename.c_str());
  fCanvas->Update ();
  return pv;
}

//----------
double makeGifHists3 (TH1* fHist, TH1* fRefHist, TCanvas* fCanvas, const std::string& fPrefix = "", const std::string& events="", double scalebyevents=0) {
  double pv = fHist->KolmogorovTest (fRefHist, "OU");
  // set style
  TPad pad ("pad", "pad", 0, 0, 1, 0.9, 0);
  pad.SetLogx ();
  pad.Draw();

  char buf [1024];
  sprintf (buf, "%s: Kolmogorov Test PV = %5.3f", fPrefix.c_str(), pv);
  TPaveText title (0.3,0.85,0.95, 0.99, buf);
  title.SetFillColor(pv > 0.01 ? 3 : 2);
  title.AddText (fPrefix.c_str());
  sprintf (buf, "Kolmogorov Test PV = %6.4f", pv);
  title.AddText (buf);
  //TText* t1 = title.AddText (fPrefix.c_str());
  //sprintf (buf, "Kolmogorov Test PV = %6.4f", pv);
  //TText* t2 = title.AddText (buf);
  //t1->SetTextSize(0.3);
  //t2->SetTextSize(0.3);
  title.Draw();

  pad.cd();

  // Do not normalize the profile plots
  /*
  if ( events == 'y') {
    fHist->Scale (scalebyevents);
  }
  else {
    fHist->Sumw2 ();
    fHist->Scale (fRefHist->GetSumOfWeights () / fHist->GetSumOfWeights ());
  }
  */

  fHist->SetMarkerStyle (21);
  fHist->SetMarkerSize (0.7);
  fRefHist->SetLineColor (2);
  std::string name = fRefHist->GetTitle ();
  int blank = name.rfind (' ');
  if (blank >= 0) name.erase (0, blank+1);
  fHist->SetXTitle (name.c_str());
  fHist->SetTitle ("");

  fRefHist->Draw ();
  fHist->Draw ("e1p,same");
  std::string filename = name + "_logx.gif";
  fCanvas->Print (filename.c_str());
  fCanvas->Update ();
  return pv;
}

//----------
double makeGifHists4 (TH1* fHist, TH1* fRefHist, TCanvas* fCanvas, const std::string& fPrefix = "", const std::string& events="", double scalebyevents=0) {
  double pv = fHist->KolmogorovTest (fRefHist, "OU");
  // set style
  TPad pad ("pad", "pad", 0, 0, 1, 0.9, 0);
  pad.Draw();

  char buf [1024];
  sprintf (buf, "%s: Kolmogorov Test PV = %5.3f", fPrefix.c_str(), pv);
  TPaveText title (0.3,0.85,0.95, 0.99, buf);
  title.SetFillColor(pv > 0.01 ? 3 : 2);
  title.AddText (fPrefix.c_str());
  sprintf (buf, "Kolmogorov Test PV = %6.4f", pv);
  title.AddText (buf);
  //TText* t1 = title.AddText (fPrefix.c_str());
  //sprintf (buf, "Kolmogorov Test PV = %6.4f", pv);
  //TText* t2 = title.AddText (buf);
  //t1->SetTextSize(0.3);
  //t2->SetTextSize(0.3);
  title.Draw();

  pad.cd();

  // Do not normalize the profile plots
  /*
  if ( events == 'y') {
    fHist->Scale (scalebyevents);
  }
  else {
    fHist->Sumw2 ();
    fHist->Scale (fRefHist->GetSumOfWeights () / fHist->GetSumOfWeights ());
  }
  */

  fHist->SetMarkerStyle (21);
  fHist->SetMarkerSize (0.7);
  fRefHist->SetLineColor (2);
  std::string name = fRefHist->GetTitle ();
  int blank = name.rfind (' ');
  if (blank >= 0) name.erase (0, blank+1);
  fHist->SetXTitle (name.c_str());
  fHist->SetTitle ("");

  // Checking low and high edge of the x-axis
  double lowedge = fHist->GetBinLowEdge(1);
  int hibin=fHist->GetNbinsX()+1;
  double hiedge = fHist->GetBinLowEdge(hibin);

  fRefHist->SetMaximum(1.3);
  fRefHist->SetMinimum(0.0);

  if (lowedge==0.5 && hiedge==3.75){
    fRefHist->GetXaxis()->SetTickLength(0);
    fRefHist->GetXaxis()->SetLabelSize(0);
    TGaxis *axis = new TGaxis(0.5,0.0,3.75,0.0,3.1622766,5623.41325,50510,"G");
    fRefHist->Draw ();
    axis->Draw();
  } else {
    fRefHist->Draw ();
  }

  fHist->Draw ("e1p,same");
  std::string filename = name + ".gif";
  fCanvas->Print (filename.c_str());
  fCanvas->Update ();
  return pv;
}

//----------
int main (int argn, char* argv []) {
  //int result = 0; // OK

  if (argn < 6) {
    std::cout << "Usage: " 
	      << argv[0] 
	      << " <file_name> <reference file_name> <module_name> <module_name_for_reference> <description> <use number of events for normalization? (y or n)>, default is n " << std::endl;
    return 1;
  }

  //
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetCanvasColor(0);
  gStyle->SetFrameLineWidth(2);
  gStyle->SetPadColor(0);
  gStyle->SetTitleFillColor(0);
  gStyle->SetStatColor(0);

  gStyle->SetOptStat(0);
  gStyle->SetOptFit(1100);

  gStyle->SetStatX(0.92);
  gStyle->SetStatY(0.88);
  gStyle->SetStatW(0.20);
  gStyle->SetStatH(0.10);

  gStyle->SetStatY(0.86);
  gStyle->SetStatW(0.16);
  gStyle->SetStatH(0.06);

  gStyle->SetTitleX(0.05);
  gStyle->SetTitleW(0.25);
  //

  std::string inputFileName (argv[1]);
  std::string refFileName (argv[2]);
  std::string moduleName (argv[3]);
  std::string refmoduleName (argv[4]);
  std::string globalTitle = argv[5];
  std::string normalization = "n";
  if (argn == 7) {
    normalization = (argv[6]);
  }
  std::cout << normalization << std::endl;
  std::cout << "Processing file " << inputFileName << std::endl;

  //
  // --- Open input file
  TFile* inputFile = TFile::Open (inputFileName.c_str());
  if (!inputFile) {
    std::cerr << "Cannot open file " << inputFileName << std::endl;
    return 1;
  }

  TDirectory* dirIn = 0;
  std::string workDir = std::string ("DQMData/JetMET/RecoJetsV/") + moduleName ; // new format
  inputFile->GetObject (workDir.c_str(), dirIn);

  if (!dirIn) {
    std::cout << "Fall back to old format for file " << inputFileName << std::endl;
    workDir = std::string ("DQMData/") + moduleName; // old format
    inputFile->GetObject (workDir.c_str(), dirIn);
    if (!dirIn) {
      std::cerr << "Can't access workDir in file " << inputFileName << std::endl;
      return 1;
    }
  }

  //
  // --- Open output file
  TFile* refFile = TFile::Open (refFileName.c_str());
  if (!refFile) {
    std::cerr << "Cannot open file " << refFileName << std::endl;
    return 1;
  }

  TDirectory* dirRef = 0;
  //  workDir = std::string ("DQMData/JetMET/RecoJetsV/CaloJetTask_") + refmoduleName; // new format
  workDir = std::string ("DQMData/JetMET/RecoJetsV/") + refmoduleName; // new format
  refFile->GetObject (workDir.c_str(), dirRef);

  if (!dirRef) {
    std::cout << "Fall back to old format for file " << refFileName << std::endl;
    //    workDir = std::string ("DQMData/CaloJetTask_") + refmoduleName; // old format
    workDir = std::string ("DQMData/") + refmoduleName; // old format
    refFile->GetObject (workDir.c_str(), dirRef);
    if (!dirRef) {
      std::cerr << "Can't access workDir in file " << refFileName << std::endl;
      return 1;
    }
  }
  
  //
  // --- Get lists of objects 
  std::vector<std::string> histKeys  = getAllKeys (dirIn, "TH1F");
  std::vector<std::string> histKeys2 = getAllKeys (dirIn, "TProfile");
  std::vector<std::string> histKeys3 = getAllKeys (dirIn, "TProfile");
  std::vector<std::string> histKeys4 = getAllKeys (dirIn, "TProfile");

  // output
  gStyle->SetOptStat (kFALSE);
  TCanvas canvas ("Jets","Jets",800,600);

  //
  // Obtain the scale factor based on the "numberofevents" histogram
  bool found=false;
  double scaleforevents = 0;
  for (unsigned ihist = 0; ihist < histKeys.size (); ++ihist) {
    TH1* histforcheck = 0;
    dirIn->GetObject (histKeys[ihist].c_str(), histforcheck);
    std::string nameforcheck = histforcheck->GetTitle ();
    if ( nameforcheck == "numberofevents") {
      TH1* refhistforcheck = 0;
      dirRef->GetObject (histKeys[ihist].c_str(), refhistforcheck);
      std::cout << "hist=numberofevnets" << std::endl;
      histforcheck->Sumw2 ();
      refhistforcheck->Sumw2 ();
      double scaleforcheck=histforcheck->GetSumOfWeights ();
      double refscaleforcheck=refhistforcheck->GetSumOfWeights ();
      scaleforevents = refscaleforcheck/scaleforcheck;
      found=true;
    }
  }
  if (!found) printf("numberofevents histogram not found in histKeys loop.\n");
  else        printf("numberofevents histogram found in histKeys loop.\n");

  //
  // Make the 1D histogram gifs
  for (unsigned ihist = 0; ihist < histKeys.size (); ++ihist) {
    TH1* hist = 0;
    dirIn->GetObject (histKeys[ihist].c_str(), hist);
    if (hist) {
      TH1* refhist = 0;
      dirRef->GetObject (histKeys[ihist].c_str(), refhist);
      if (refhist) {
	std::string title = globalTitle;
	double pv = makeGifHists (hist, refhist, &canvas, title, normalization, scaleforevents);
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

//   found=false;
//   for (unsigned ihist = 0; ihist < histKeys2.size (); ++ihist) {
//     TH1* histforcheck = 0;
//     dirIn->GetObject (histKeys2[ihist].c_str(), histforcheck);
//     std::string nameforcheck = histforcheck->GetTitle ();
//     if ( nameforcheck == "numberofevents") {
//       TH1* refhistforcheck = 0;
//       dirRef->GetObject (histKeys2[ihist].c_str(), refhistforcheck);
//       std::cout << "hist=numberofevnets" << std::endl;
//       histforcheck->Sumw2 ();
//       refhistforcheck->Sumw2 ();
//       double scaleforcheck=histforcheck->GetSumOfWeights ();
//       double refscaleforcheck=refhistforcheck->GetSumOfWeights ();
//       scaleforevents = refscaleforcheck/scaleforcheck;
//       found=true;
//     }
//   }
//   if (!found) printf("numberofevents histogram not found in histKeys2 loop.\n");
//   else        printf("numberofevents histogram found in histKeys2 loop.\n");

  //
  // Make the profile histogram gifs in logy scale
  for (unsigned ihist = 0; ihist < histKeys2.size (); ++ihist) {
    TH1* hist = 0;
    dirIn->GetObject (histKeys2[ihist].c_str(), hist);
    if (hist) {
      TH1* refhist = 0;
      dirRef->GetObject (histKeys2[ihist].c_str(), refhist);
      if (refhist) {
	std::string title = globalTitle;
	double pv = makeGifHists2 (hist, refhist, &canvas, title, normalization, scaleforevents);
	std::cout << "pv for hist " << histKeys2[ihist] << " is " << pv << std::endl; 
      }
      else {
	std::cerr << "Can not get reference histogram " << histKeys2[ihist] << std::endl;
      }
    }
    else {
      std::cerr << "Can not get histogram " << histKeys2[ihist] << std::endl;
    }
  }

//   found=false;
//   for (unsigned ihist = 0; ihist < histKeys3.size (); ++ihist) {
//     TH1* histforcheck = 0;
//     dirIn->GetObject (histKeys3[ihist].c_str(), histforcheck);
//     std::string nameforcheck = histforcheck->GetTitle ();
//     if ( nameforcheck == "numberofevents") {
//       TH1* refhistforcheck = 0;
//       dirRef->GetObject (histKeys3[ihist].c_str(), refhistforcheck);
//       std::cout << "hist=numberofevnets" << std::endl;
//       histforcheck->Sumw2 ();
//       refhistforcheck->Sumw2 ();
//       double scaleforcheck=histforcheck->GetSumOfWeights ();
//       double refscaleforcheck=refhistforcheck->GetSumOfWeights ();
//       scaleforevents = refscaleforcheck/scaleforcheck;
//       found=true;
//     }
//   }
//   if (!found) printf("numberofevents histogram not found in histKeys3 loop.\n");
//   else        printf("numberofevents histogram found in histKeys3 loop.\n");

  //
  // Make the profile histogram gifs in logx scale
  for (unsigned ihist = 0; ihist < histKeys3.size (); ++ihist) {
    TH1* hist = 0;
    dirIn->GetObject (histKeys3[ihist].c_str(), hist);
    if (hist) {
      TH1* refhist = 0;
      dirRef->GetObject (histKeys3[ihist].c_str(), refhist);
      if (refhist) {
	std::string title = globalTitle;
	double pv = makeGifHists3 (hist, refhist, &canvas, title, normalization, scaleforevents);
	std::cout << "pv for hist " << histKeys3[ihist] << " is " << pv << std::endl; 
      }
      else {
	std::cerr << "Can not get reference histogram " << histKeys3[ihist] << std::endl;
      }
    }
    else {
      std::cerr << "Can not get histogram " << histKeys3[ihist] << std::endl;
    }
  }

//   found=false;
//   for (unsigned ihist = 0; ihist < histKeys4.size (); ++ihist) {
//     TH1* histforcheck = 0;
//     dirIn->GetObject (histKeys4[ihist].c_str(), histforcheck);
//     std::string nameforcheck = histforcheck->GetTitle ();
//     if ( nameforcheck == "numberofevents") {
//       TH1* refhistforcheck = 0;
//       dirRef->GetObject (histKeys4[ihist].c_str(), refhistforcheck);
//       std::cout << "hist=numberofevnets" << std::endl;
//       histforcheck->Sumw2 ();
//       refhistforcheck->Sumw2 ();
//       double scaleforcheck=histforcheck->GetSumOfWeights ();
//       double refscaleforcheck=refhistforcheck->GetSumOfWeights ();
//       scaleforevents = refscaleforcheck/scaleforcheck;
//       found=true;
//     }
//   }
//   if (!found) printf("numberofevents histogram not found in histKeys4 loop.\n");
//   else        printf("numberofevents histogram found in histKeys4 loop.\n");

  //
  // Make the profile histogram gifs in linear scale
  for (unsigned ihist = 0; ihist < histKeys4.size (); ++ihist) {
    TH1* hist = 0;
    dirIn->GetObject (histKeys4[ihist].c_str(), hist);
    if (hist) {
      TH1* refhist = 0;
      dirRef->GetObject (histKeys4[ihist].c_str(), refhist);
      if (refhist) {
	std::string title = globalTitle;
	double pv = makeGifHists4 (hist, refhist, &canvas, title, normalization, scaleforevents);
	std::cout << "pv for hist " << histKeys4[ihist] << " is " << pv << std::endl; 
      }
      else {
	std::cerr << "Can not get reference histogram " << histKeys4[ihist] << std::endl;
      }
    }
    else {
      std::cerr << "Can not get histogram " << histKeys4[ihist] << std::endl;
    }
  }

  return 0;
}

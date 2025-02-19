#include <iostream>
// comparison of digi histograms with reference ones.
//  root -b -q hcaloval.C\(\"HB\"\) - just for PV comparison.
//  root -b -q hcaloval.C\(\"HB\",\"gif\"\) - PV comparison and creation of gif for each histo
//  root -b -q hcaloval.C\(\"HB\",\"ps\"\) - PV comparison and creation of ps file for each subdetector

#include "TFile.h"
#include "TTree.h"
#include "TText.h"
#include "TStyle.h"
#include "TPostScript.h"
#include "TString.h"
#include "HistoCompare.C"


class CSCOval
{
public:
  CSCOval(const char * suffix, const char* drawhisto="none");
  
  ~CSCOval();

  void process(string name);
  void plot3x3(string histName);
  void plot(string histName);
  void runStrips();
  void runWires();
  void runComparators();
  void run();

private:
  TFile * rfile;
  TFile * sfile;
  HistoCompare * theComp;
  string theSuffix;
  string theDirectory;
};


CSCOval::CSCOval(const char* suffix, const char* drawhisto)
: rfile(0),
  sfile(0),
  theComp(0),
  theSuffix(suffix),
  theDirectory("DQMData/CSCDigiTask")
{
  gROOT->Reset();

  string PathToRef = "../data/";
  string rfilename = PathToRef+ "CSCDigiValidation_ref.root";
  string sfilename = "CSCDigiValidation" + theSuffix + ".root";

  delete gROOT->GetListOfFiles()->FindObject(rfilename.c_str());
  delete gROOT->GetListOfFiles()->FindObject(sfilename.c_str());

  // TText* te = new TText();
  // te->SetTextSize(0.1);
  rfile = new TFile(rfilename.c_str());
  sfile = new TFile(sfilename.c_str());
  rfile->cd(theDirectory.c_str());
  gDirectory->ls();

  sfile->cd(theDirectory.c_str());
  gDirectory->ls();

  gStyle->SetOptStat("n");

  //gROOT->ProcessLine(".x HistoCompare.C");
  if(drawhisto == "gif")
  {
    theComp = new HistoCompareGif("CSC", theSuffix);
  }
  else if(drawhisto == "ps")
  {
    theComp = new HistoComparePS("cscDigiValidation.ps");
  }
  else
  {
    HistoCompare * comp = new HistoCompare();
    theComp = new HistoCompare();
  }
  run();
}

CSCOval::~CSCOval()
{
  delete theComp;
}


void CSCOval::process(string histname)
{
  TH1 * oldHist;
  TH1 * newHist;
  string tname = theDirectory + "/" + histname + ";1";
  cout << tname << endl;
  rfile->GetObject(tname.c_str(), oldHist);
  sfile->GetObject(tname.c_str(), newHist); 
  theComp->compare(oldHist, newHist);

}


void CSCOval::plot3x3(string histName)
{
  theComp->openPage(histName, 3, 3);
  for(int i = 1; i <=9; ++i)
  {
    strstream hist;
    hist << histName << i;
    process(hist.str());
  }
  theComp->closePage();
}


void CSCOval::plot(string histName)
{
  theComp->openPage(histName);
  process(histName);
  theComp->closePage();
}


void CSCOval::runStrips()
{
  plot("CSCPedestal");
  plot("CSCStripDigisPerLayer");
  plot("CSCStripDigisPerEvent");
  plot("CSCStripAmplitude");
  plot("CSCStrip4to5");
  plot("CSCStrip6to5");
  plot3x3("CSCStripDigiResolution");
  plot3x3("CSCWireDigiTimeType");
  plot3x3("CSCComparatorDigiTimeType");
}


void CSCOval::runWires()
{
}

void runComparators()
{
}


void CSCOval::run()
{
  runStrips();
  runWires();
  runComparators();
}



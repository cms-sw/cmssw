#ifndef ROOT_PREFS_H
#define ROOT_PREFS_H

#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include "TMath.h"
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TString.h"
#include "TCut.h"
#include "TClonesArray.h"
#include "TProfile.h"
#include "TF1.h"
#include "TStyle.h"
#include "Rtypes.h"
#include "TText.h"
#include "TLine.h"

void initStyle(TStyle *sty);

void SetupTowerDisplay(TH2F *hist);

void SetStatus(TH1F* hist, string status);

void SetupTitle(TH1F* hist, char* xtitle, char* ytitle);

void SetupTitle(TProfile* hist, char* xtitle, char* ytitle);

void SetupTitle(TH2F* hist, char* xtitle, char* ytitle);

class index_map{
 public:
  index_map();
  typedef map<int,int> i2i;
  ~index_map();

  int ntpg(int index) const
    {
      return nvec2ntpg.find(index)->second;
    }

  int nvec(int index) const
    {
      return ntpg2nvec.find(index)->second;
    }

 private:
  i2i ntpg2nvec;
  i2i nvec2ntpg;
};


#endif //ROOT_PREFS_H

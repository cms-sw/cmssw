#include "Validation/RecoTrack/interface/FitSlicesYTool.h"

using namespace std;

void FitSlicesYTool::run(TH2F* h2, MonitorElement * me, int kind){
  h2->Write();
  h2->FitSlicesY();
  string name(h2->GetName());
  string title(h2->GetTitle());
  string n;//t;
  if (kind==1) { 
    n="_1";
    //t="Mean";
  } else {
    n="_2";
    //t="Sigma";
  }
  TH1* h1 = (TH1*)gDirectory->Get((name+n).c_str());
  h1->Write();
  for (int bin=0;bin!=h1->GetNbinsX();bin++){
    me->setBinContent(bin+1,h1->GetBinContent(bin+1));
  }
  delete gDirectory->Get((name+"_0").c_str());
  delete gDirectory->Get((name+"_1").c_str());
  delete gDirectory->Get((name+"_2").c_str());
  delete gDirectory->Get((name+"_chi2").c_str());

}

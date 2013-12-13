#include "TFile.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TLegend.h"
#include "TLine.h"

void recovsmcdraw(TFile* _file0, const char* dir, const char* label, const int color, TLegend* leg, const bool isFirst) {

  TProfile* prof=0;
  if(_file0->cd(dir)) {
    prof = (TProfile*)gDirectory->Get("recovsmcnvtxprof");
    if(prof) {
      prof->SetMarkerStyle(20);
      prof->SetMarkerColor(color);
      prof->SetLineColor(color);
      if(isFirst) {prof->Draw();} else {prof->Draw("same");}
      prof->GetXaxis()->SetRangeUser(-0.5,30.5);
      if(leg) {
	leg->AddEntry(prof,label,"p");
      }
    }
  }
  
}

void recovsmcalgoplot(TFile* _file0, const char* dir, const char* name, const double offset)
{

  char dir1[300]; 
  char dir2[300]; 
  char dir3[300]; 

  sprintf(dir1,"%sanalyzer",dir);
  sprintf(dir2,"%sD0s51mm",dir);
  sprintf(dir3,"%sDA100um",dir);

  TLegend leg(.4,.2,.6,.4,name);

  recovsmcdraw(_file0,dir3,"DA 100um",kRed,&leg,true);
  recovsmcdraw(_file0,dir1,"2010 reco",kBlack,&leg,false);
  recovsmcdraw(_file0,dir2,"gap=1mm, d0 sig=5",kBlue,&leg,false);

  TLine ll(0,offset,30,offset+30*0.7); 
  ll.DrawClone();
  leg.AddEntry(&ll,"70% efficiency","l");
  leg.DrawClone();
}

void recovsmcplot(TFile* _file0, const char* dir, const char* name, const double offset)
{

  char dir1[300]; 
  char dir2[300]; 
  char dir3[300]; 

  sprintf(dir1,"%s",dir);
  sprintf(dir2,"weighted%s",dir);
  sprintf(dir3,"weighted45mm%s",dir);

  TLegend leg(.4,.2,.6,.4,name);

  recovsmcdraw(_file0,dir1,"sigmaZ=6.26cm",kBlack,&leg,true);
  recovsmcdraw(_file0,dir2,"sigmaZ=5.20cm",kBlue,&leg,false);
  recovsmcdraw(_file0,dir3,"sigmaZ=4.50cm",kRed,&leg,false);

  TLine ll(0,offset,30,offset+30*0.7); 
  ll.DrawClone();
  leg.AddEntry(&ll,"70% efficiency","l");
  leg.DrawClone();
}

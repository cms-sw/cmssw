#include "BSvsPVPlots.h"
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <iostream>
#include "TPad.h"
#include "TH1D.h"
#include "TFile.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TF1.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "DPGAnalysis/SiStripTools/interface/CommonAnalyzer.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TText.h"
#include "TLegend.h"

void BSvsPVPlots(const char* fullname,const char* module, const char* label, const char* postfix, const char* shortname, const char* outtrunk) {

  char modfull[300];
  sprintf(modfull,"%s%s",module,postfix);
  char labfull[300];
  sprintf(labfull,"%s%s",label,postfix);

  char dirname[400];
  sprintf(dirname,"%s",shortname);

  //  char fullname[300];
  //  if(strlen(family)==0) {  sprintf(fullname,"rootfiles/Tracking_PFG_%s.root",filename);}
  //  else {  sprintf(fullname,"rootfiles/%s.root",dirname); }



  std::string workdir = outtrunk ;
  workdir += dirname;
  gSystem->cd(workdir.c_str());
  gSystem->MakeDirectory(labfull);
  //  gSystem->cd("/afs/cern.ch/cms/tracking/output");


  TFile ff(fullname);

  gStyle->SetOptStat(111111);
  // Colliding events

  
  CommonAnalyzer castat(&ff,"",modfull);
  {
    TH1F* deltax  = (TH1F*)castat.getObject("deltax");
    if (deltax && deltax->GetEntries()>0) {
      deltax->Draw();
      gPad->SetLogy(1);
      std::string plotfilename;
      plotfilename += outtrunk;
      plotfilename += dirname;
      plotfilename += "/";
      plotfilename += labfull;
      plotfilename += "/deltax_";
      plotfilename += labfull;
      plotfilename += "_";
      plotfilename += dirname;
      plotfilename += ".gif";
      gPad->Print(plotfilename.c_str());
      gPad->SetLogy(0);
      delete deltax;
    }
    TH1F* deltay  = (TH1F*)castat.getObject("deltay");
    if (deltay && deltay->GetEntries()>0) {
      deltay->Draw();
      gPad->SetLogy(1);
      std::string plotfilename;
      plotfilename += outtrunk;
      plotfilename += dirname;
      plotfilename += "/";
      plotfilename += labfull;
      plotfilename += "/deltay_";
      plotfilename += labfull;
      plotfilename += "_";
      plotfilename += dirname;
      plotfilename += ".gif";
      gPad->Print(plotfilename.c_str());
      gPad->SetLogy(0);
      delete deltay;
    }
    TH1F* deltaz  = (TH1F*)castat.getObject("deltaz");
    if (deltaz && deltaz->GetEntries()>0) {
      deltaz->Draw();
      gPad->SetLogy(1);
      std::string plotfilename;
      plotfilename += outtrunk;
      plotfilename += dirname;
      plotfilename += "/";
      plotfilename += labfull;
      plotfilename += "/deltaz_";
      plotfilename += labfull;
      plotfilename += "_";
      plotfilename += dirname;
      plotfilename += ".gif";
      gPad->Print(plotfilename.c_str());
      gPad->SetLogy(0);
      delete deltaz;
    }
    gStyle->SetOptStat(111);
    gStyle->SetOptFit(111);
    TProfile* deltaxvsz  = (TProfile*)castat.getObject("deltaxvsz");
    if (deltaxvsz && deltaxvsz->GetEntries()>0) {
      //    deltaxvsz->Draw();
      deltaxvsz->Fit("pol1","","",-10.,10.);
      if(deltaxvsz->GetFunction("pol1")) {
	deltaxvsz->GetFunction("pol1")->SetLineColor(kRed);
	deltaxvsz->GetFunction("pol1")->SetLineWidth(1);
      }
      deltaxvsz->GetYaxis()->SetRangeUser(-0.001,0.001);
      std::string plotfilename;
      plotfilename += outtrunk;
      plotfilename += dirname;
      plotfilename += "/";
      plotfilename += labfull;
      plotfilename += "/deltaxvsz_";
      plotfilename += labfull;
      plotfilename += "_";
      plotfilename += dirname;
      plotfilename += ".gif";
      gPad->Print(plotfilename.c_str());
      delete deltaxvsz;
    }
    TProfile* deltayvsz  = (TProfile*)castat.getObject("deltayvsz");
    if (deltayvsz && deltayvsz->GetEntries()>0) {
      //    deltayvsz->Draw();
      deltayvsz->Fit("pol1","","",-10.,10.);
      if(deltayvsz->GetFunction("pol1")) {
	deltayvsz->GetFunction("pol1")->SetLineColor(kRed);
	deltayvsz->GetFunction("pol1")->SetLineWidth(1);
      }
      deltayvsz->GetYaxis()->SetRangeUser(-0.001,0.001);
      std::string plotfilename;
      plotfilename += outtrunk;
      plotfilename += dirname;
      plotfilename += "/";
      plotfilename += labfull;
      plotfilename += "/deltayvsz_";
      plotfilename += labfull;
      plotfilename += "_";
      plotfilename += dirname;
      plotfilename += ".gif";
      gPad->Print(plotfilename.c_str());
      delete deltayvsz;
    }
  }
  gStyle->SetOptStat(111111);
  gStyle->SetOptFit(1111);
  
  // Define fitting functions
  TF1* fdoubleg = new TF1("doubleg","[1]*exp(-0.5*((x-[0])/[2])**2)+[3]*exp(-0.5*((x-[0])/[4])**2)+[5]*exp(sqrt((x-[0])**2)*[6])",-.2,.2);
  fdoubleg->SetLineColor(kRed);
  fdoubleg->SetLineWidth(1);
  /*
  fdoubleg->SetParLimits(1,0.,1e9);
  fdoubleg->SetParLimits(3,0.,1e9);
  fdoubleg->SetParLimits(5,0.,1e9);
  */

  gStyle->SetOptFit(1111);

  // Summary histograms
  TH1D* deltaxsum = new TH1D("deltaxsum","(PV-BS) Fitted X position vs run",10,0.,10.);
  deltaxsum->SetCanExtend(TH1::kAllAxes);
  TH1D* deltaysum = new TH1D("deltaysum","(PV-BS) Fitted Y position vs run",10,0.,10.);
  deltaysum->SetCanExtend(TH1::kAllAxes);

  TH1D* deltaxmeansum = new TH1D("deltaxmeansum","(PV-BS) Mean X position vs run",10,0.,10.);
  deltaxmeansum->SetCanExtend(TH1::kAllAxes);
  TH1D* deltaymeansum = new TH1D("deltaymeansum","(PV-BS) Mean Y position vs run",10,0.,10.);
  deltaymeansum->SetCanExtend(TH1::kAllAxes);

  TH1D* deltazmeansum = new TH1D("deltazmeansum","(PV-BS) Mean Z position vs run",10,0.,10.);
  deltazmeansum->SetCanExtend(TH1::kAllAxes);

  std::vector<unsigned int> runs = castat.getRunList();
  std::sort(runs.begin(),runs.end());

  {
 
    std::cout << "Found " << runs.size() << " runs" << std::endl;

    for(unsigned int i=0;i<runs.size();++i) {
      
      char runlabel[100];
      sprintf(runlabel,"%d",runs[i]);
      char runpath[100];
      sprintf(runpath,"run_%d",runs[i]);
      castat.setPath(runpath);
      std::cout << runpath << std::endl;
       
      TH1F* deltax  = (TH1F*)castat.getObject("deltaxrun");
      if (deltax && deltax->GetEntries()>0) {
	//	deltax->Draw();
	fdoubleg->SetParameter(0,deltax->GetMean());
	fdoubleg->SetParameter(2,deltax->GetRMS());
	fdoubleg->SetParameter(4,deltax->GetRMS());
	fdoubleg->SetParameter(1,deltax->GetMaximum());
	fdoubleg->SetParameter(3,0.1*deltax->GetMaximum());
	fdoubleg->SetParameter(5,0.1*deltax->GetMaximum());
	const int result = deltax->Fit(fdoubleg,"b","",-.05,.05);
	gPad->SetLogy(1);
	char tresult[100];
	sprintf(tresult,"%d",result);
	TText res; res.SetTextColor(kRed);
	if(result!=0) res.DrawTextNDC(.2,.8,tresult);

	int bin = deltaxsum->Fill(runlabel,fdoubleg->GetParameter(0));
	deltaxsum->SetBinError(bin,fdoubleg->GetParError(0));

	bin = deltaxmeansum->Fill(runlabel,deltax->GetMean());
	deltaxmeansum->SetBinError(bin,deltax->GetMeanError());

	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/deltaxrun_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	gPad->SetLogy(0);
	delete deltax;
      }
      TH1F* deltay  = (TH1F*)castat.getObject("deltayrun");
      if (deltay && deltay->GetEntries()>0) {
	deltay->Draw();
	fdoubleg->SetParameter(0,deltay->GetMean());
	fdoubleg->SetParameter(2,deltay->GetRMS());
	fdoubleg->SetParameter(4,deltay->GetRMS());
	fdoubleg->SetParameter(1,deltay->GetMaximum());
	fdoubleg->SetParameter(3,0.1*deltay->GetMaximum());
	fdoubleg->SetParameter(5,0.1*deltay->GetMaximum());
	const int result = deltay->Fit(fdoubleg,"b","",-.05,.05);
	gPad->SetLogy(1);
	char tresult[100];
	sprintf(tresult,"%d",result);
	TText res; res.SetTextColor(kRed);
	if(result!=0) res.DrawTextNDC(.2,.8,tresult);

	int bin = deltaysum->Fill(runlabel,fdoubleg->GetParameter(0));
	deltaysum->SetBinError(bin,fdoubleg->GetParError(0));

	bin = deltaymeansum->Fill(runlabel,deltay->GetMean());
	deltaymeansum->SetBinError(bin,deltay->GetMeanError());

	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/deltayrun_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	gPad->SetLogy(0);
	delete deltay;
      }
      TH1F* deltaz  = (TH1F*)castat.getObject("deltazrun");
      if (deltaz && deltaz->GetEntries()>0) {
	deltaz->Draw();
	gPad->SetLogy(1);

	int bin = deltazmeansum->Fill(runlabel,deltaz->GetMean());
	deltazmeansum->SetBinError(bin,deltaz->GetMeanError());

	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/deltazrun_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	gPad->SetLogy(0);
	delete deltaz;
      }

      TProfile* deltaxvsz  = (TProfile*)castat.getObject("deltaxvszrun");
      if (deltaxvsz && deltaxvsz->GetEntries()>0) {
	deltaxvsz->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/deltaxvszrun_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete deltaxvsz;
      }
      
      TProfile* deltayvsz  = (TProfile*)castat.getObject("deltayvszrun");
      if (deltayvsz && deltayvsz->GetEntries()>0) {
	deltayvsz->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/deltayvszrun_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete deltayvsz;
      }
      
      
      TH1F* deltaxvsorb  = (TH1F*)castat.getObject("deltaxvsorbrun");
      if (deltaxvsorb && deltaxvsorb->GetEntries()>0) {
	deltaxvsorb->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/deltaxvsorb_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete deltaxvsorb;
      }
      TH1F* deltayvsorb  = (TH1F*)castat.getObject("deltayvsorbrun");
      if (deltayvsorb && deltayvsorb->GetEntries()>0) {
	deltayvsorb->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/deltayvsorb_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete deltayvsorb;
      }
      TH1F* deltazvsorb  = (TH1F*)castat.getObject("deltazvsorbrun");
      if (deltazvsorb && deltazvsorb->GetEntries()>0) {
	deltazvsorb->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/deltazvsorb_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete deltazvsorb;
      }
    }
  }

  gStyle->SetOptStat(1111);
  gStyle->SetOptFit(0);

  if(runs.size()) {
    std::string plotfilename;
    
    plotfilename = outtrunk;
    plotfilename += dirname;
    plotfilename += "/";
    plotfilename += labfull;
    plotfilename += "/deltaxsum_";
    plotfilename += labfull;
    plotfilename += "_";
    plotfilename += dirname;
    plotfilename += ".gif";

    TCanvas * cwidedeltax = new TCanvas(plotfilename.c_str(),plotfilename.c_str(),1500,500);

    deltaxsum->SetLineColor(kRed);    deltaxsum->SetMarkerColor(kRed);
    deltaxsum->GetYaxis()->SetRangeUser(-.002,.002);
    deltaxsum->GetYaxis()->SetTitle("#Delta x (cm)");
    deltaxsum->Draw();
    deltaxmeansum->Draw("esame");
    TLegend deltaxleg(.7,.8,.85,.9,"#Delta(x)");
    deltaxleg.AddEntry(deltaxsum,"fitted mean","l");
    deltaxleg.AddEntry(deltaxmeansum,"aritm. mean","l");
    deltaxleg.Draw();
    gPad->Print(plotfilename.c_str());
    delete cwidedeltax;

    plotfilename = outtrunk;
    plotfilename += dirname;
    plotfilename += "/";
    plotfilename += labfull;
    plotfilename += "/deltaysum_";
    plotfilename += labfull;
    plotfilename += "_";
    plotfilename += dirname;
    plotfilename += ".gif";

    TCanvas * cwidedeltay = new TCanvas(plotfilename.c_str(),plotfilename.c_str(),1500,500);

    deltaysum->SetLineColor(kRed);    deltaysum->SetMarkerColor(kRed);
    deltaysum->GetYaxis()->SetRangeUser(-.002,.002);
    deltaysum->GetYaxis()->SetTitle("#Delta y (cm)");
    deltaysum->Draw();
    deltaymeansum->Draw("esame");
    TLegend deltayleg(.7,.8,.85,.9,"#Delta(y)");
    deltayleg.AddEntry(deltaysum,"fitted mean","l");
    deltayleg.AddEntry(deltaymeansum,"aritm. mean","l");
    deltayleg.Draw();
    gPad->Print(plotfilename.c_str());
    delete cwidedeltay;


    plotfilename = outtrunk;
    plotfilename += dirname;
    plotfilename += "/";
    plotfilename += labfull;
    plotfilename += "/deltazsum_";
    plotfilename += labfull;
    plotfilename += "_";
    plotfilename += dirname;
    plotfilename += ".gif";

    TCanvas * cwidedeltaz = new TCanvas(plotfilename.c_str(),plotfilename.c_str(),1500,500);

    deltazmeansum->GetYaxis()->SetRangeUser(-2.,2.);
    deltazmeansum->GetYaxis()->SetTitle("#Delta z (cm)");
    deltazmeansum->Draw();
    gPad->Print(plotfilename.c_str());
    delete cwidedeltaz;
  }
  delete deltaxsum;
  delete deltaysum;
  delete deltaxmeansum;
  delete deltaymeansum;
  delete deltazmeansum;


  ff.Close();
  delete fdoubleg;
}


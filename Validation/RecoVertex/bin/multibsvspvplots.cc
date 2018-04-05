#include "multibsvspvplots.h"
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

void multibsvspvplots(const char* module, const char* label, const char* postfix, const char* /* family="" */) {

  TFile* file[5];

  file[0] = new TFile("rootfiles/Tracking_PFG_Run2012A_prompt_minbias_v1_190456-194076_muon_relumi_v55_fittedV0.root");
  file[1] = new TFile("rootfiles/Tracking_PFG_Run2012B_prompt_minbias_v1_190456-196531_muon_relumi_v55_fittedV0.root");
  file[2] = new TFile("rootfiles/Tracking_PFG_Run2012C_prompt_minbias_v1_190456-199011_muon_relumi_v55_fittedV0.root");
  file[3] = new TFile("rootfiles/Tracking_PFG_Run2012C_prompt_minbias_v2_190456-203002_muon_relumi_v57_fittedV0.root");
  file[4] = new TFile("rootfiles/Tracking_PFG_Run2012D_prompt_minbias_v1_190456-208686_muon_relumi_v57_fittedV0.root");

  char modfull[300];
  sprintf(modfull,"%s%s",module,postfix);

  char labfull[300];
  sprintf(labfull,"%s%s",label,postfix);

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

  TF1* fdoubleg = new TF1("doubleg","[1]*exp(-0.5*((x-[0])/[2])**2)+[3]*exp(-0.5*((x-[0])/[4])**2)+[5]*exp(sqrt((x-[0])**2)*[6])",-.2,.2);
  fdoubleg->SetLineColor(kRed);
  fdoubleg->SetLineWidth(1);

  for(unsigned int ifile=0;ifile<5;++ifile) {
    
    CommonAnalyzer castat(file[ifile],"",modfull);
    
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

	  fdoubleg->SetParameter(0,deltax->GetMean());
	  fdoubleg->SetParameter(2,deltax->GetRMS());
	  fdoubleg->SetParameter(4,deltax->GetRMS());
	  fdoubleg->SetParameter(1,deltax->GetMaximum());
	  fdoubleg->SetParameter(3,0.1*deltax->GetMaximum());
	  fdoubleg->SetParameter(5,0.1*deltax->GetMaximum());
	  /* const int result = */ deltax->Fit(fdoubleg,"q0b","",-.05,.05);
	  
	  int bin = deltaxsum->Fill(runlabel,fdoubleg->GetParameter(0));
	  deltaxsum->SetBinError(bin,fdoubleg->GetParError(0));
	  
	  bin = deltaxmeansum->Fill(runlabel,deltax->GetMean());
	  deltaxmeansum->SetBinError(bin,deltax->GetMeanError());
	  
	}
	TH1F* deltay  = (TH1F*)castat.getObject("deltayrun");
	if (deltay && deltay->GetEntries()>0) {

	  fdoubleg->SetParameter(0,deltay->GetMean());
	  fdoubleg->SetParameter(2,deltay->GetRMS());
	  fdoubleg->SetParameter(4,deltay->GetRMS());
	  fdoubleg->SetParameter(1,deltay->GetMaximum());
	  fdoubleg->SetParameter(3,0.1*deltay->GetMaximum());
	  fdoubleg->SetParameter(5,0.1*deltay->GetMaximum());
	  /* const int result = */ deltay->Fit(fdoubleg,"q0b","",-.05,.05);
	  int bin = deltaysum->Fill(runlabel,fdoubleg->GetParameter(0));
	  deltaysum->SetBinError(bin,fdoubleg->GetParError(0));
	  
	  bin = deltaymeansum->Fill(runlabel,deltay->GetMean());
	  deltaymeansum->SetBinError(bin,deltay->GetMeanError());
	  
	}
	TH1F* deltaz  = (TH1F*)castat.getObject("deltazrun");
	if (deltaz && deltaz->GetEntries()>0) {
	  
	  int bin = deltazmeansum->Fill(runlabel,deltaz->GetMean());
	  deltazmeansum->SetBinError(bin,deltaz->GetMeanError());
	  
	}
	
	
	
      }
      
    }
  }  
  
  std::string plotfilename;
  
  plotfilename = "/afs/cern.ch/cms/tracking/output/";
  //    plotfilename += dirname;
  //    plotfilename += "/";
  //  plotfilename += labfull;
  plotfilename += "/deltaxsum_";
  plotfilename += labfull;
  //    plotfilename += "_";
  //    plotfilename += dirname;
  plotfilename += ".gif";
  
  /* TCanvas * cwidedeltax = */ new TCanvas(plotfilename.c_str(),plotfilename.c_str(),1500,500);
  
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
  //  delete cwidedeltax;
  
  plotfilename = "/afs/cern.ch/cms/tracking/output/";
  //    plotfilename += dirname;
  //    plotfilename += "/";
  //  plotfilename += labfull;
  plotfilename += "/deltaysum_";
  plotfilename += labfull;
  //    plotfilename += "_";
  //    plotfilename += dirname;
  plotfilename += ".gif";
  
  /* TCanvas * cwidedeltay = */ new TCanvas(plotfilename.c_str(),plotfilename.c_str(),1500,500);
  
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
  //  delete cwidedeltay;
  
  
  plotfilename = "/afs/cern.ch/cms/tracking/output/";
  //    plotfilename += dirname;
  //    plotfilename += "/";
  //  plotfilename += labfull;
  plotfilename += "/deltazsum_";
  plotfilename += labfull;
  //    plotfilename += "_";
  //    plotfilename += dirname;
  plotfilename += ".gif";
  
  /* TCanvas * cwidedeltaz = */ new TCanvas(plotfilename.c_str(),plotfilename.c_str(),1500,500);
  
  deltazmeansum->GetYaxis()->SetRangeUser(-2.,2.);
  deltazmeansum->GetYaxis()->SetTitle("#Delta z (cm)");
  deltazmeansum->Draw();
  gPad->Print(plotfilename.c_str());
  //  delete cwidedeltaz;
  /*
  delete deltaxsum;
  delete deltaysum;
  delete deltaxmeansum;
  delete deltaymeansum;
  delete deltazmeansum;
  */
  delete fdoubleg;
  
}

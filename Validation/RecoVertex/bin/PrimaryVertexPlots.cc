#include "PrimaryVertexPlots.h"
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
#include "TH1D.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "DPGAnalysis/SiStripTools/interface/CommonAnalyzer.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TGraphErrors.h"

void PrimaryVertexPlots(const char* fullname,const char* module, const char* postfix, const char* label, const char* shortname, const char* outtrunk) {

  std::cout << shortname << module << postfix << label << std::endl;

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

  // Colliding events

  
  CommonAnalyzer castat(&ff,"",modfull);

  char bsmodule[300];
  sprintf(bsmodule,"beamspotanalyzer%s",postfix);
  CommonAnalyzer cabs(&ff,"",bsmodule);
  sprintf(bsmodule,"onlinebsanalyzer%s",postfix);
  CommonAnalyzer cabsonl(&ff,"",bsmodule);
  sprintf(bsmodule,"testbsanalyzer%s",postfix);
  CommonAnalyzer cabstest(&ff,"",bsmodule);


  std::cout << "ready" << std::endl;
    
  TH1F* ntrvtx = (TH1F*)castat.getObject("ntruevtx");
  if(ntrvtx) {
	ntrvtx->Draw();
	gPad->SetLogy(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/ntrvtx_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete ntrvtx;
	gPad->SetLogy(0);
  }

  gStyle->SetOptStat(111);
  gStyle->SetOptFit(111);
  TProfile* ntrvtxvslumi = (TProfile*)castat.getObject("ntruevtxvslumi");
  if(ntrvtxvslumi && ntrvtxvslumi->GetEntries()>0 ) {
    //	ntrvtxvslumi->Draw();
	ntrvtxvslumi->Fit("pol2","","",0.5,3.0);
	if(ntrvtxvslumi->GetFunction("pol2")) {
	  ntrvtxvslumi->GetFunction("pol2")->SetLineColor(kBlack);
	  ntrvtxvslumi->GetFunction("pol2")->SetLineWidth(1);
	}
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/ntrvtxvslumi_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
  }

  gStyle->SetOptStat(1111);

  TH2D* ntrvtxvslumi2D = (TH2D*)castat.getObject("ntruevtxvslumi2D");
  if(ntrvtxvslumi2D && ntrvtxvslumi2D->GetEntries()>0 ) {
	ntrvtxvslumi2D->Draw("colz");
	if(ntrvtxvslumi) {
	  ntrvtxvslumi->SetMarkerStyle(20);
	  ntrvtxvslumi->SetMarkerSize(.3);
	  ntrvtxvslumi->Draw("same");
	}
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/ntrvtxvslumi2D_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->SetLogz(1);
	gPad->Print(plotfilename.c_str());
	gPad->SetLogz(0);
	delete ntrvtxvslumi2D;
  }
  delete ntrvtxvslumi;

  std::cout << "ready2" << std::endl;

  TH1F* ndofvtx = (TH1F*)castat.getObject("ndof");
  if(ndofvtx) {
	ndofvtx->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/ndofvtx_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete ndofvtx;
  }

  TH1F* ntracksvtx = (TH1F*)castat.getObject("ntracks");
  if(ntracksvtx) {
	ntracksvtx->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/ntracksvtx_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete ntracksvtx;
  }

  TH1F* aveweight = (TH1F*)castat.getObject("aveweight");
  if(aveweight) {
	aveweight->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/aveweight_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete aveweight;
  }

  TProfile* aveweightvsvtxz = (TProfile*)castat.getObject("aveweightvsvtxz");
  if(aveweightvsvtxz) {
	aveweightvsvtxz->Draw();
	aveweightvsvtxz->GetYaxis()->SetRangeUser(0.75,1.05);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/aveweightvsvtxz_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete aveweightvsvtxz;
  }

  TProfile* ndofvsvtxz = (TProfile*)castat.getObject("ndofvsvtxz");
  if(ndofvsvtxz) {
	ndofvsvtxz->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/ndofvsvtxz_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete ndofvsvtxz;
  }

  TProfile* ntracksvsvtxz = (TProfile*)castat.getObject("ntracksvsvtxz");
  if(ntracksvsvtxz) {
	ntracksvsvtxz->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/ntracksvsvtxz_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete ntracksvsvtxz;
  }

  std::cout << "ready3" << std::endl;

  TH2F* ndofvsntrks = (TH2F*)castat.getObject("ndofvstracks");
  if(ndofvsntrks) {
	ndofvsntrks->Draw("colz");
	gPad->SetLogz(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/ndofvsntrks_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete ndofvsntrks;
	gPad->SetLogz(0);
  }
  std::cout << "ready4" << std::endl;

  TH1F* trkweights = (TH1F*)castat.getObject("weights");
  if(trkweights) {
	trkweights->Draw();
	gPad->SetLogy(1);
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/trkweights_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete trkweights;
	gPad->SetLogy(0);
  }

  {
    TH1F* vtxx  = (TH1F*)castat.getObject("vtxx");
    if (vtxx) {
      vtxx->Draw();
      gPad->SetLogy(1);
      
      std::string plotfilename;
      plotfilename += outtrunk;
      plotfilename += dirname;
      plotfilename += "/";
      plotfilename += labfull;
      plotfilename += "/pvtxx_";
      plotfilename += labfull;
      plotfilename += "_";
      plotfilename += dirname;
      plotfilename += ".gif";
      gPad->Print(plotfilename.c_str());
      delete vtxx;
      gPad->SetLogy(0);
    }
    
    TH1F* vtxy  = (TH1F*)castat.getObject("vtxy");
    if (vtxy) {
      vtxy->Draw();
      gPad->SetLogy(1);
      
      std::string plotfilename;
      plotfilename += outtrunk;
      plotfilename += dirname;
      plotfilename += "/";
      plotfilename += labfull;
      plotfilename += "/pvtxy_";
      plotfilename += labfull;
      plotfilename += "_";
      plotfilename += dirname;
      plotfilename += ".gif";
      gPad->Print(plotfilename.c_str());
      delete vtxy;
      gPad->SetLogy(0);
    }
    
    TH1F* vtxz  = (TH1F*)castat.getObject("vtxz");
    if (vtxz) {
      vtxz->Draw();
      gPad->SetLogy(1);
      
      std::string plotfilename;
      plotfilename += outtrunk;
      plotfilename += dirname;
      plotfilename += "/";
      plotfilename += labfull;
      plotfilename += "/pvtxz_";
      plotfilename += labfull;
      plotfilename += "_";
      plotfilename += dirname;
      plotfilename += ".gif";
      gPad->Print(plotfilename.c_str());
      delete vtxz;
      gPad->SetLogy(0);
    }
  }

  std::cout << "ready4" << std::endl;
  

  // Summary histograms
  /*
  TH1D* vtxxsum = new TH1D("vtxxsum","(BS-PV) Fitted X position vs run",10,0.,10.);
  vtxxsum->SetCanExtend(TH1::kAllAxes);
  TH1D* vtxysum = new TH1D("vtxysum","(BS-PV) Fitted Y position vs run",10,0.,10.);
  vtxysum->SetCanExtend(TH1::kAllAxes);
  TH1D* vtxzsum = new TH1D("vtxzsum","(BS-PV) Fitted Y position vs run",10,0.,10.);
  vtxzsum->SetCanExtend(TH1::kAllAxes);
  */

  TH1D* vtxxmeansum = new TH1D("vtxxmeansum","PV mean X position vs run",10,0.,10.);
  vtxxmeansum->SetCanExtend(TH1::kAllAxes);
  TH1D* vtxymeansum = new TH1D("vtxymeansum","PV mean Y position vs run",10,0.,10.);
  vtxymeansum->SetCanExtend(TH1::kAllAxes);
  TH1D* vtxzmeansum = new TH1D("vtxzmeansum","PV mean Z position vs run",10,0.,10.);
  vtxzmeansum->SetCanExtend(TH1::kAllAxes);
  TH1D* vtxzsigmasum = new TH1D("vtxzsigmasum","PV sigma Z position vs run",10,0.,10.);
  vtxzsigmasum->SetCanExtend(TH1::kAllAxes);

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
      cabs.setPath(runpath);
      cabsonl.setPath(runpath);
      cabstest.setPath(runpath);
      std::cout << runpath << std::endl;
       
      TH1F* vtxx  = (TH1F*)castat.getObject("vtxxrun");
      if (vtxx && vtxx->GetEntries()>0) {
	vtxx->Draw();
	gPad->SetLogy(1);


	int bin = vtxxmeansum->Fill(runlabel,vtxx->GetMean());
	vtxxmeansum->SetBinError(bin,vtxx->GetMeanError());


	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/pvtxx_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete vtxx;
      }
      gPad->SetLogy(0);
      TH1F* vtxy  = (TH1F*)castat.getObject("vtxyrun");
      if (vtxy && vtxy->GetEntries()>0) {
	vtxy->Draw();
	gPad->SetLogy(1);

	int bin = vtxymeansum->Fill(runlabel,vtxy->GetMean());
	vtxymeansum->SetBinError(bin,vtxy->GetMeanError());

	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/pvtxy_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete vtxy;
      }
      gPad->SetLogy(0);
      double vtxsigmazrunvalue = -1.;      double vtxsigmazrunerror = -1.;
      double vtxsigmazrunfitvalue = -1.;      double vtxsigmazrunfiterror = -1.;
      TH1F* vtxz  = (TH1F*)castat.getObject("vtxzrun");
      if (vtxz && vtxz->GetEntries()>0) {
	vtxz->Fit("gaus","","",-3.*vtxz->GetRMS(),3.*vtxz->GetRMS());
	//	vtxz->Draw();
	if(vtxz->GetFunction("gaus")) {
	  vtxz->GetFunction("gaus")->SetLineColor(kRed);
	  vtxz->GetFunction("gaus")->SetLineWidth(1);
	  vtxsigmazrunfitvalue = vtxz->GetFunction("gaus")->GetParameter(2);
	  vtxsigmazrunfiterror = vtxz->GetFunction("gaus")->GetParError(2);
	}
	gPad->SetLogy(1);

	int bin = vtxzmeansum->Fill(runlabel,vtxz->GetMean());
	vtxzmeansum->SetBinError(bin,vtxz->GetMeanError());

	bin = vtxzsigmasum->Fill(runlabel,vtxz->GetRMS());
	vtxzsigmasum->SetBinError(bin,vtxz->GetRMSError());

	vtxsigmazrunvalue = vtxz->GetRMS();
	vtxsigmazrunerror = vtxz->GetRMSError();

	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/pvtxz_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete vtxz;
      }
      gPad->SetLogy(0);


      TH1F* vtxxvsorb  = (TH1F*)castat.getObject("vtxxvsorbrun");
      TH1F* bsxvsorb  = (TH1F*)cabs.getObject("bsxvsorbrun");
      TH1F* onlbsxvsorb  = (TH1F*)cabsonl.getObject("bsxvsorbrun");
      TH1F* testbsxvsorb  = (TH1F*)cabstest.getObject("bsxvsorbrun");
      if (vtxxvsorb && vtxxvsorb->GetEntries()>0) {
	//	vtxxvsorb->GetYaxis()->SetRangeUser(0.0650,0.07);
	vtxxvsorb->Draw();
	if(bsxvsorb) {
	  bsxvsorb->SetMarkerColor(kGreen); 	  bsxvsorb->SetLineColor(kGreen); 	  bsxvsorb->SetLineWidth(2);
	  bsxvsorb->Draw("esame");
	}
	if(onlbsxvsorb) {
	  onlbsxvsorb->SetMarkerColor(kRed); 	  onlbsxvsorb->SetLineColor(kRed);
	  onlbsxvsorb->Draw("esame");
	}
	if(testbsxvsorb) {
	  testbsxvsorb->SetMarkerColor(kBlue); 	  testbsxvsorb->SetLineColor(kBlue);
	  testbsxvsorb->Draw("esame");
	}
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/pvtxxvsorb_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete vtxxvsorb;	delete bsxvsorb;	delete onlbsxvsorb;	delete testbsxvsorb;
      }
      TH1F* vtxyvsorb  = (TH1F*)castat.getObject("vtxyvsorbrun");
      TH1F* bsyvsorb  = (TH1F*)cabs.getObject("bsyvsorbrun");
      TH1F* onlbsyvsorb  = (TH1F*)cabsonl.getObject("bsyvsorbrun");
      TH1F* testbsyvsorb  = (TH1F*)cabstest.getObject("bsyvsorbrun");
      if (vtxyvsorb && vtxyvsorb->GetEntries()>0) {
	//	vtxyvsorb->GetYaxis()->SetRangeUser(0.0620,0.0670);
	vtxyvsorb->Draw();
	if(bsyvsorb) {
	  bsyvsorb->SetMarkerColor(kGreen); 	  bsyvsorb->SetLineColor(kGreen); 	  bsyvsorb->SetLineWidth(2);
	  bsyvsorb->Draw("esame");
	}
	if(onlbsyvsorb) {
	  onlbsyvsorb->SetMarkerColor(kRed); 	  onlbsyvsorb->SetLineColor(kRed);
	  onlbsyvsorb->Draw("esame");
	}
	if(testbsyvsorb) {
	  testbsyvsorb->SetMarkerColor(kCyan); 	  testbsyvsorb->SetLineColor(kCyan);
	  testbsyvsorb->Draw("esame");
	}
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/pvtxyvsorb_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete vtxyvsorb;	delete bsyvsorb;	delete onlbsyvsorb;	delete testbsyvsorb;
      }
      TH1F* vtxzvsorb  = (TH1F*)castat.getObject("vtxzvsorbrun");
      TH1F* bszvsorb  = (TH1F*)cabs.getObject("bszvsorbrun");
      TH1F* onlbszvsorb  = (TH1F*)cabsonl.getObject("bszvsorbrun");
      TH1F* testbszvsorb  = (TH1F*)cabstest.getObject("bszvsorbrun");
      if (vtxzvsorb && vtxzvsorb->GetEntries()>0) {
	vtxzvsorb->Draw();
	if(bszvsorb) {
	  bszvsorb->SetMarkerColor(kGreen); 	  bszvsorb->SetLineColor(kGreen); 	  bszvsorb->SetLineWidth(2);
	  bszvsorb->Draw("esame");
	}
	if(onlbszvsorb) {
	  onlbszvsorb->SetMarkerColor(kRed); 	  onlbszvsorb->SetLineColor(kRed);
	  onlbszvsorb->Draw("esame");
	}
	if(testbszvsorb) {
	  testbszvsorb->SetMarkerColor(kCyan); 	  testbszvsorb->SetLineColor(kCyan);
	  testbszvsorb->Draw("esame");
	}
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/pvtxzvsorb_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete vtxzvsorb;	delete bszvsorb;    delete onlbszvsorb;    delete testbszvsorb;
      }

      TProfile* bssigmazvsorb  = (TProfile*)cabs.getObject("bssigmazvsorbrun");
      TProfile* onlbssigmazvsorb  = (TProfile*)cabsonl.getObject("bssigmazvsorbrun");
      TProfile* testbssigmazvsorb  = (TProfile*)cabstest.getObject("bssigmazvsorbrun");
      TGraphErrors gr;TGraphErrors grfit;
      if ( bssigmazvsorb || onlbssigmazvsorb || testbssigmazvsorb) {
	if(bssigmazvsorb) {
	  bssigmazvsorb->SetMarkerColor(kGreen); 	  bssigmazvsorb->SetLineColor(kGreen); 	  bssigmazvsorb->SetLineWidth(2);
	  bssigmazvsorb->Draw();
	  bssigmazvsorb->GetYaxis()->SetRangeUser(0.,7.);
	  if(vtxsigmazrunvalue >= 0.) {
	    // look for last filled bin
	    int lastbin= bssigmazvsorb->GetNbinsX()+1;
	    int firstbin= 1;
	    for(int ibin=bssigmazvsorb->GetNbinsX()+1;ibin>0;--ibin) {
	      if(bssigmazvsorb->GetBinEntries(ibin)!=0) {
		lastbin=ibin;
		break;
	      }
	    }
	    for(int ibin=1;ibin<=bssigmazvsorb->GetNbinsX()+1;++ibin) {
	      if(bssigmazvsorb->GetBinEntries(ibin)!=0) {
		firstbin=ibin;
		break;
	      }
	    }
	    gr.SetMarkerStyle(20);
	    gr.SetPoint(1,(bssigmazvsorb->GetBinCenter(firstbin)+bssigmazvsorb->GetBinCenter(lastbin))/2.,vtxsigmazrunvalue);
	    gr.SetPointError(1,(bssigmazvsorb->GetBinCenter(lastbin)-bssigmazvsorb->GetBinCenter(firstbin))/2.,vtxsigmazrunerror);
	    gr.Draw("p");
	    grfit.SetMarkerStyle(24);grfit.SetMarkerColor(kBlue);grfit.SetLineColor(kBlue);
	    grfit.SetPoint(1,(bssigmazvsorb->GetBinCenter(firstbin)+bssigmazvsorb->GetBinCenter(lastbin))/2.,vtxsigmazrunfitvalue);
	    grfit.SetPointError(1,(bssigmazvsorb->GetBinCenter(lastbin)-bssigmazvsorb->GetBinCenter(firstbin))/2.,vtxsigmazrunfiterror);
	    grfit.Draw("p");
	  }
	}
	if(onlbssigmazvsorb) {
	  onlbssigmazvsorb->SetMarkerColor(kRed); 	  onlbssigmazvsorb->SetLineColor(kRed);
	  onlbssigmazvsorb->Draw("esame");
	}
	if(testbssigmazvsorb) {
	  testbssigmazvsorb->SetMarkerColor(kCyan); 	  testbssigmazvsorb->SetLineColor(kCyan);
	  testbssigmazvsorb->Draw("esame");
	}
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/pvtxsigmazvsorb_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete bssigmazvsorb;    delete onlbssigmazvsorb;    delete testbssigmazvsorb;
      }






      TProfile* nvtxvsorb = (TProfile*)castat.getObject("nvtxvsorbrun");
      if(nvtxvsorb) {
	nvtxvsorb->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/nvtxvsorb_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete nvtxvsorb;
      }
  
      TProfile* nvtxvsbx = (TProfile*)castat.getObject("nvtxvsbxrun");
      if(nvtxvsbx) {
	nvtxvsbx->SetLineColor(kRed);	nvtxvsbx->SetMarkerColor(kRed);	nvtxvsbx->SetMarkerStyle(20);	nvtxvsbx->SetMarkerSize(.5);
	nvtxvsbx->Draw();
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/nvtxvsbx_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	delete nvtxvsbx;
      }
  
      TH2D* nvtxvsbxvsorb = (TH2D*)castat.getObject("nvtxvsbxvsorbrun");
      if(nvtxvsbxvsorb) {
	
	nvtxvsbxvsorb->Draw("colz");
	std::string plotfilename;
	plotfilename += outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/nvtxvsbxvsorb_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());
	
	// slicing
	std::string cname;
	cname = "slice_run_";
	cname += runpath;
	new TCanvas(cname.c_str(),cname.c_str());
	bool first=true;
	int mcount=20;
	int ccount=1;
	for(unsigned int bx=0;bx<3564;++bx) {
	  char hname[300];
	  sprintf(hname,"bx_%d",bx);
	  TH1D* slice = nvtxvsbxvsorb->ProjectionY(hname,bx+1,bx+1);
	  //	  std::cout << "slice " << bx << " with pointer " << slice << std::endl;
	  if(slice) {
	    if(slice->GetEntries()) {
	      std::cout << "slice " << bx << " ready " << std::endl;
	      slice->SetMarkerStyle(mcount);
	      slice->SetMarkerColor(ccount);
	      slice->SetLineColor(ccount);
	      slice->SetMarkerSize(.4);
	      if(first) {slice->SetMaximum(4.); slice->Draw("e");}
	      else {slice->Draw("same");}
	      first=false;
	      ++mcount;
	      if(mcount==28) {mcount=20; ++ccount;}
	    }
	  }
	}
	//	std::string plotfilename;
	plotfilename = outtrunk;
	plotfilename += dirname;
	plotfilename += "/";
	plotfilename += labfull;
	plotfilename += "/nvtxvsorbsliced_";
	plotfilename += labfull;
	plotfilename += "_";
	plotfilename += dirname;
	plotfilename += "_";
	plotfilename += runpath;
	plotfilename += ".gif";
	gPad->Print(plotfilename.c_str());

	delete nvtxvsbxvsorb;
      }
    }

  }
  if(runs.size()) {
    std::string plotfilename;
    
    plotfilename = outtrunk;
    plotfilename += dirname;
    plotfilename += "/";
    plotfilename += labfull;
    plotfilename += "/vtxxsum_";
    plotfilename += labfull;
    plotfilename += "_";
    plotfilename += dirname;
    plotfilename += ".gif";

    TCanvas * cwidevtxx = new TCanvas(plotfilename.c_str(),plotfilename.c_str(),1500,500);

    vtxxmeansum->GetYaxis()->SetRangeUser(.05,.15);
    vtxxmeansum->GetYaxis()->SetTitle("x (cm)");
    vtxxmeansum->Draw();
    //    vtxxmeansum->SetLineColor(kRed);    vtxxmeansum->SetMarkerColor(kRed);
    //    vtxxmeansum->Draw("esame");
    gPad->Print(plotfilename.c_str());
    delete cwidevtxx;

    plotfilename = outtrunk;
    plotfilename += dirname;
    plotfilename += "/";
    plotfilename += labfull;
    plotfilename += "/vtxysum_";
    plotfilename += labfull;
    plotfilename += "_";
    plotfilename += dirname;
    plotfilename += ".gif";

    TCanvas * cwidevtxy = new TCanvas(plotfilename.c_str(),plotfilename.c_str(),1500,500);

    vtxymeansum->GetYaxis()->SetRangeUser(-0.05,.05);
    vtxymeansum->GetYaxis()->SetTitle("y (cm)");
    vtxymeansum->Draw();
    //    vtxymeansum->SetLineColor(kRed);    vtxymeansum->SetMarkerColor(kRed);
    //    vtxymeansum->Draw("esame");
    gPad->Print(plotfilename.c_str());
    delete cwidevtxy;

    plotfilename = outtrunk;
    plotfilename += dirname;
    plotfilename += "/";
    plotfilename += labfull;
    plotfilename += "/vtxzsum_";
    plotfilename += labfull;
    plotfilename += "_";
    plotfilename += dirname;
    plotfilename += ".gif";

    TCanvas * cwidevtxz = new TCanvas(plotfilename.c_str(),plotfilename.c_str(),1500,500);

    vtxzmeansum->GetYaxis()->SetRangeUser(-2.,2.);
    vtxzmeansum->GetYaxis()->SetTitle("z (cm)");
    vtxzmeansum->Draw();
    //    vtxzmeansum->SetLineColor(kRed);    vtxzmeansum->SetMarkerColor(kRed);
    //    vtxzmeansum->Draw("esame");
    gPad->Print(plotfilename.c_str());
    delete cwidevtxz;

    plotfilename = outtrunk;
    plotfilename += dirname;
    plotfilename += "/";
    plotfilename += labfull;
    plotfilename += "/vtxsigmazsum_";
    plotfilename += labfull;
    plotfilename += "_";
    plotfilename += dirname;
    plotfilename += ".gif";

    TCanvas * cwidevtxsigmaz = new TCanvas(plotfilename.c_str(),plotfilename.c_str(),1500,500);

    vtxzsigmasum->GetYaxis()->SetRangeUser(0.,15.);
    vtxzsigmasum->GetYaxis()->SetTitle("z (cm)");
    vtxzsigmasum->Draw();
    //    vtxzsigmasum->SetLineColor(kRed);    vtxzsigmasum->SetMarkerColor(kRed);
    //    vtxzsigmasum->Draw("esame");
    gPad->Print(plotfilename.c_str());
    delete cwidevtxsigmaz;

  }
  delete vtxxmeansum;
  delete vtxymeansum;
  delete vtxzmeansum;
  delete vtxzsigmasum;
}


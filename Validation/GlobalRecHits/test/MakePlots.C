///////////////////////////////////////////////////////////////////////////////
// Macro to produce histograms from the GlobalHitsProducer
//
// root -b -q MakePlots.C
///////////////////////////////////////////////////////////////////////////////
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TString.h"

void MakePlots(TString filename="GlobalRecHitsHistograms")
{
  gROOT->Reset();
  //http://root.cern.ch/root/html/src/TStyle.cxx.html#TStyle:SetOptStat
  gStyle->SetOptStat("emruo");

  TString srcname = filename+".root";

  // clear memory of file name
  delete gROOT->GetListOfFiles()->FindObject(srcname);

  // open source file
  TFile *srcfile = new TFile(srcname);
  
  // create canvas
  Int_t cWidth = 928, cHeight = 1218;
  //TCanvas *myCanvas = new TCanvas("globalhits","globalhits",cWidth,cHeight);
  TCanvas *myCanvas = new TCanvas("globalrechits","globalrechits");  
  //myCanvas->Size(21.59, 27.94);

  // open output ps file
  //TString filename = "GlobalHitsHistograms";
  TString psfile = filename+".ps";
  TString psfileopen = filename+".ps[";
  TString psfileclose = filename+".ps]";
  myCanvas->Print(psfileopen);

  // create label
  TLatex *label = new TLatex();
  label->SetNDC();
  TString labeltitle;

  // create attributes
  Int_t srccolor = kBlue;
  Int_t linewidth = 2;

  vector<int> histnames;



  vector<string> ecalhistname;
  ecalhistname.push_back("hEcalRes_EB");
  ecalhistname.push_back("hEcalRes_EE");
  ecalhistname.push_back("hEcalRes_ES");
  ecalhistname.push_back("hEcaln_EB");
  ecalhistname.push_back("hEcaln_EE");
  ecalhistname.push_back("hEcaln_ES");
  histnames.push_back(0);


  vector<string> hcalhistname;
  hcalhistname.push_back("hHcalRes_HB");
  hcalhistname.push_back("hHcalRes_HE");
  hcalhistname.push_back("hHcalRes_HF");
  hcalhistname.push_back("hHcalRes_HO");
  hcalhistname.push_back("hHcaln_HB");
  hcalhistname.push_back("hHcaln_HE");
  hcalhistname.push_back("hHcaln_HF");
  hcalhistname.push_back("hHcaln_HO");
  histnames.push_back(1);

  vector<string> pxlhistname;
  string PxlString[7] = {"BRL1", "BRL2", "BRL3", "FWD1n", "FWD1p", "FWD2n", "FWD2p"};
  string ResX[7], ResY[7], Count[7];
  for(int j = 0; j < 7; ++j)
    {
      ResX[j] = "hSiPixelResX_"+PxlString[j];
      pxlhistname.push_back(ResX[j]);
      pxlhistname.push_back("hSiPixelResY_"+PxlString[j]);
      pxlhistname.push_back("hSiPixeln_"+PxlString[j]);
    }

  histnames.push_back(2);
  vector<string> sihistname;
  string SiString[19] = {"TECW1", "TECW2", "TECW3", "TECW4", "TECW5", "TECW6", "TECW7", "TECW8", "TIBL1", "TIBL2", "TIBL3", "TIBL4", "TIDW1", "TIDW2", "TIDW3", "TOBL1", "TOBL2", "TOBL3", "TOBL4"};
  for(int i = 0; i < 19; ++i)
    {
      sihistname.push_back("hSiStripResX_"+SiString[i]);
      sihistname.push_back("hSiStripResY_"+SiString[i]);
      sihistname.push_back("hSiStripn_"+SiString[i]);
    }
  histnames.push_back(3);

  vector<string> cschistname;
  cschistname.push_back("CSCResRDPhi");
  cschistname.push_back("hCSCn");
  histnames.push_back(4);

  vector<string> dthistname;
  dthistname.push_back("hDtMuonn");
  dthistname.push_back("hDtMuonRes");
  histnames.push_back(5);

  vector<string> rpchistname;
  rpchistname.push_back("hRPCResX");
  rpchistname.push_back("hRPCn");
  histnames.push_back(6);

  //loop through histograms to prepare output
  for (Int_t i = 0; i < histnames.size(); ++i) {
    
    vector<string> names;
    
    // setup canvas depending on group of plots
    TCanvas *Canvas;

    if (i == 0) {
      names = ecalhistname;
      //Canvas = new TCanvas("ECalHits","ECalHits",cWidth,cHeight);
      Canvas = new TCanvas("ECalHits","ECalHits");
      //Canvas->Size(21.59, 27.94);
      Canvas->Divide(2,4);
      myCanvas = Canvas;
      myCanvas->cd(0);
      //label->DrawLatex(0.5,1.00,"ECal Information");
    }
    if (i == 1) {
      names = hcalhistname;
      //Canvas = new TCanvas("HCalHits","HCalHits",cWidth,cHeight);
      Canvas = new TCanvas("HCalHits","HCalHits");
      //Canvas->Size(21.59, 27.94);
      Canvas->Divide(2,4);
      myCanvas = Canvas;
      myCanvas->cd(0);
      //label->DrawLatex(0.5,1.00,"HCal Information");
    }
    if (i == 2) {
      names = pxlhistname;
      //Canvas = new TCanvas("PixelHits","PixelHits",cWidth,cHeight);
      Canvas = new TCanvas("PixelHits","PixelHits");
      //Canvas->Size(21.59, 27.94);
      Canvas->Divide(2,4);
      myCanvas = Canvas;
      myCanvas->cd(0);
      //label->DrawLatex(0.5,1.00,"Pixel Information");
    }
    if (i == 3) {
      names = sihistname;
      //Canvas = new TCanvas("StripHits","StripHits",cWidth,cHeight);
      Canvas = new TCanvas("StripHits","StripHits");
      //Canvas->Size(21.59, 27.94);
      Canvas->Divide(2,4);
      myCanvas = Canvas;
      myCanvas->cd(0);
      //label->DrawLatex(0.5,1.00,"Strip Information");
    }
    if (i == 4) {
      names = cschistname;
      //Canvas = new TCanvas("MuonCscHits","MuonCscHits",cWidth,cWidth);
      Canvas = new TCanvas("MuonCscHits","MuonCscHits");
      //Canvas->Size(21.59, 27.94);
      Canvas->Divide(2,2);
      myCanvas = Canvas;
      myCanvas->cd(0);
      //label->DrawLatex(0.5,1.00,"Muon CSC Information");
    }
    if (i == 5) {
      names = dthistname;
      //Canvas = new TCanvas("MuonDtHits","MuonDtHits",cWidth,cWidth);
      Canvas = new TCanvas("MuonDtHits","MuonDtHits");
      //Canvas->Size(21.59, 27.94);
      Canvas->Divide(2,2);
      myCanvas = Canvas;
      myCanvas->cd(0);
      //label->DrawLatex(0.5,1.00,"Muon DT Information");
    }
    if (i == 6) {
      names = rpchistname;
      //Canvas = new TCanvas("MuonRpcHits","MuonRpcHits",cWidth,cWidth);
      Canvas = new TCanvas("MuonRpcHits","MuonRpcHits");
      //Canvas->Size(21.59, 27.94);
      Canvas->Divide(2,3);
      myCanvas = Canvas;
      myCanvas->cd(0);
      //label->DrawLatex(0.5,1.00,"Muon RPC Information");
    }

    // loop through plots
    for (Int_t j = 0; j < names.size(); ++j) {

      //TH1F *sh = (TH1F*)srcfile->Get(names[j].c_str());
      TH1F *sh;

      // set axis info for the histograms
   
      if (i == 0) {
	  TString hpath = "DQMData/GlobalRecHitsV/ECals/"+names[j];
	  sh = (TH1F*)srcfile->Get(hpath);
	  cout << "i = 0" << "j =  " << j  << endl ; 
       
      }
      if (i == 1) {
	TString hpath = "DQMData/GlobalRecHitsV/HCals/"+names[j];
	sh = (TH1F*)srcfile->Get(hpath);
	cout << "i = 1" << endl;
		    
      }
      if (i == 2) {
	TString hpath = "DQMData/GlobalRecHitsV/SiPixels/"+names[j];
	sh = (TH1F*)srcfile->Get(hpath);
	 cout << "i = 2" << endl ;

      }
      if (i == 3) {
	TString hpath = "DQMData/GlobalRecHitsV/SiStrips/"+names[j];
	sh = (TH1F*)srcfile->Get(hpath);
	 cout << "i = 3" << endl ;
      }		    
      if (i == 4) {
	TString hpath = "DQMData/GlobalRecHitsV/Muons/"+names[j];
	sh = (TH1F*)srcfile->Get(hpath);
	 cout << "i = 3" << endl ;
      }
      if (i == 5) {
	TString hpath = "DQMData/GlobalRecHitsV/Muons/"+names[j];
	sh = (TH1F*)srcfile->Get(hpath);
	 cout << "i = 5" << endl ;
      }
      if (i == 6) {
	TString hpath = "DQMData/GlobalRecHitsV/Muons/"+names[j];
	sh = (TH1F*)srcfile->Get(hpath);
	 cout << "i = 6" << endl ;
      }
      cout << "HELP ME!!!! " << endl;
      
      sh->SetLineColor(srccolor);
      sh->SetLineWidth(linewidth);    

      // make plots
      myCanvas->cd(j+1);
      //gPad->SetLogy();
      sh->Draw();

    } // end loop through plots

    myCanvas->Print(psfile);

  } // end loop through histnames

  // close output ps file
  //myCanvas->Print(psfileclose);

  srcfile->Close();

  //convert to pdf
  TString cmnd;
  cmnd = "ps2pdf "+psfile+" "+filename+".pdf";
  gSystem->Exec(cmnd);
  cmnd = "rm "+psfile;
  gSystem->Exec(cmnd);  

  return;
}

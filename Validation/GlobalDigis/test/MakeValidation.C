///////////////////////////////////////////////////////////////////////////////
// Macro to compare histograms from the GlobalDigisProducer for validation
//
// root -b -q MakeValidation.C
///////////////////////////////////////////////////////////////////////////////
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TString.h"

void MakeValidation(TString sfilename = "GlobalDigisHistogramsAnalyze.root",
		    TString rfilename = "GlobalDigisHistogramsAnalyze.root",
		    TString filename = "GlobalDigisHistogramsCompare")
{
  gROOT->Reset();

  // zoom in on x-axis of plots
  bool zoomx = false;

  // create new histograms for comparison
  //gROOT->LoadMacro("MakeHistograms.C");
  //MakeHistograms();

  // setup names
  //TString sfilename = "GlobalDigisHistograms.root";
  //TString rfilename = "GlobalDigisHistograms-reference.root";

  // clear memory of file names
  delete gROOT->GetListOfFiles()->FindObject(sfilename);
  delete gROOT->GetListOfFiles()->FindObject(rfilename);

  // open reference file
  TFile *sfile = new TFile(sfilename);
  TFile *rfile = new TFile(rfilename);

  // create canvas
  Int_t cWidth = 928, cHeight = 1218;
  TCanvas *myCanvas = new TCanvas("globaldigis","globaldigis");

  // open output ps file
  //TString filename = "GlobalHitsHistogramsCompare";
  TString psfile = filename+".ps";
  TString psfileopen = filename+".ps[";
  TString psfileclose = filename+".ps]";
  myCanvas->Print(psfileopen);

  // create label
  TLatex *label = new TLatex();
  label->SetNDC();
  label->SetTextSize(0.03);
  label->SetTextAlign(22);
  TString labeltitle;

  // ceate text
  TText* te = new TText();
  te->SetTextSize(0.075);

  // create attributes
  Int_t rcolor = kBlue;
  Int_t rtype = kSolid;
  Int_t stype = kSolid;
  Int_t scolor = kBlack;
  Int_t rlinewidth = 6;
  Int_t slinewidth = 2;
  vector<Int_t> histnames;

  vector<string> eehistname;
  eehistname.push_back("hEcaln_EE");
  eehistname.push_back("hEcalAEE_EE");
  eehistname.push_back("hEcalSHE_EE");
  eehistname.push_back("hEcalMaxPos_EE");
  eehistname.push_back("hEcalMultvAEE_EE");
  eehistname.push_back("hEcalSHEvAEESHE_EE");
  histnames.push_back(0);

  vector<string> ebhistname;
  ebhistname.push_back("hEcaln_EB");
  ebhistname.push_back("hEcalAEE_EB");
  ebhistname.push_back("hEcalSHE_EB");
  ebhistname.push_back("hEcalMaxPos_EB");
  ebhistname.push_back("hEcalMultvAEE_EB");
  ebhistname.push_back("hEcalSHEvAEESHE_EB");
  histnames.push_back(1);

  vector<string> eshistname;
  eshistname.push_back("hEcaln_ES");
  eshistname.push_back("hEcalADC0_ES");
  eshistname.push_back("hEcalADC1_ES");
  eshistname.push_back("hEcalADC2_ES");
  histnames.push_back(2);

  vector<string> hbhistname;
  hbhistname.push_back("hHcaln_HB");
  hbhistname.push_back("hHcalAEE_HB");
  hbhistname.push_back("hHcalSHE_HB");
  hbhistname.push_back("hHcalAEESHE_HB");
  hbhistname.push_back("hHcalSHEvAEE_HE");
  histnames.push_back(3);

  vector<string> hehistname;
  hehistname.push_back("hHcaln_HE");
  hehistname.push_back("hHcalAEE_HE");
  hehistname.push_back("hHcalSHE_HE");
  hehistname.push_back("hHcalAEESHE_HE");
  hehistname.push_back("hHcalSHEvAEE_HE");
  histnames.push_back(4);

  vector<string> hohistname;
  hohistname.push_back("hHcaln_HO");
  hohistname.push_back("hHcalAEE_HO");
  hohistname.push_back("hHcalSHE_HO");
  hohistname.push_back("hHcalAEESHE_HO");
  hohistname.push_back("hHcalSHEvAEE_HO");
  histnames.push_back(5);

  vector<string> hfhistname;
  hfhistname.push_back("hHcaln_HF");
  hfhistname.push_back("hHcalAEE_HF");
  hfhistname.push_back("hHcalSHE_HF");
  hfhistname.push_back("hHcalAEESHE_HF");
  hfhistname.push_back("hHcalSHEvAEE_HF");
  histnames.push_back(6);

  vector<string> tibl1histname;
  tibl1histname.push_back("hSiStripn_TIBL1");
  tibl1histname.push_back("hSiStripADC_TIBL1");
  tibl1histname.push_back("hSiStripStripADC_TIBL1");
  histnames.push_back(7);

  vector<string> tibl2histname;
  tibl2histname.push_back("hSiStripn_TIBL2");
  tibl2histname.push_back("hSiStripADC_TIBL2");
  tibl2histname.push_back("hSiStripStripADC_TIBL2");
  histnames.push_back(8);

  vector<string> tibl3histname;
  tibl3histname.push_back("hSiStripn_TIBL3");
  tibl3histname.push_back("hSiStripADC_TIBL3");
  tibl3histname.push_back("hSiStripStripADC_TIBL3");
  histnames.push_back(9);

  vector<string> tibl4histname;
  tibl4histname.push_back("hSiStripn_TIBL4");
  tibl4histname.push_back("hSiStripADC_TIBL4");
  tibl4histname.push_back("hSiStripStripADC_TIBL4");
  histnames.push_back(10);

  vector<string> tobl1histname;
  tobl1histname.push_back("hSiStripn_TOBL1");
  tobl1histname.push_back("hSiStripADC_TOBL1");
  tobl1histname.push_back("hSiStripStripADC_TOBL1");
  histnames.push_back(11);

  vector<string> tobl2histname;
  tobl2histname.push_back("hSiStripn_TOBL2");
  tobl2histname.push_back("hSiStripADC_TOBL2");
  tobl2histname.push_back("hSiStripStripADC_TOBL2");
  histnames.push_back(12);

  vector<string> tobl3histname;
  tobl3histname.push_back("hSiStripn_TOBL3");
  tobl3histname.push_back("hSiStripADC_TOBL3");
  tobl3histname.push_back("hSiStripStripADC_TOBL3");
  histnames.push_back(13);

  vector<string> tobl4histname;
  tobl4histname.push_back("hSiStripn_TOBL4");
  tobl4histname.push_back("hSiStripADC_TOBL4");
  tobl4histname.push_back("hSiStripStripADC_TOBL4");
  histnames.push_back(14);

  vector<string> tidw1histname;
  tidw1histname.push_back("hSiStripn_TIDW1");
  tidw1histname.push_back("hSiStripADC_TIDW1");
  tidw1histname.push_back("hSiStripStripADC_TIDW1");
  histnames.push_back(15);

  vector<string> tidw2histname;
  tidw2histname.push_back("hSiStripn_TIDW2");
  tidw2histname.push_back("hSiStripADC_TIDW2");
  tidw2histname.push_back("hSiStripStripADC_TIDW2");
  histnames.push_back(16);

  vector<string> tidw3histname;
  tidw3histname.push_back("hSiStripn_TIDW3");
  tidw3histname.push_back("hSiStripADC_TIDW3");
  tidw3histname.push_back("hSiStripStripADC_TIDW3");
  histnames.push_back(17);

  vector<string> tecw1histname;
  tecw1histname.push_back("hSiStripn_TECW1");
  tecw1histname.push_back("hSiStripADC_TECW1");
  tecw1histname.push_back("hSiStripStripADC_TECW1");
  histnames.push_back(18);

  vector<string> tecw2histname;
  tecw2histname.push_back("hSiStripn_TECW2");
  tecw2histname.push_back("hSiStripADC_TECW2");
  tecw2histname.push_back("hSiStripStripADC_TECW2");
  histnames.push_back(19);

  vector<string> tecw3histname;
  tecw3histname.push_back("hSiStripn_TECW3");
  tecw3histname.push_back("hSiStripADC_TECW3");
  tecw3histname.push_back("hSiStripStripADC_TECW3");
  histnames.push_back(20);

  vector<string> tecw4histname;
  tecw4histname.push_back("hSiStripn_TECW4");
  tecw4histname.push_back("hSiStripADC_TECW4");
  tecw4histname.push_back("hSiStripStripADC_TECW4");
  histnames.push_back(21);

  vector<string> tecw5histname;
  tecw5histname.push_back("hSiStripn_TECW5");
  tecw5histname.push_back("hSiStripADC_TECW5");
  tecw5histname.push_back("hSiStripStripADC_TECW5");
  histnames.push_back(22);

  vector<string> tecw6histname;
  tecw6histname.push_back("hSiStripn_TECW6");
  tecw6histname.push_back("hSiStripADC_TECW6");
  tecw6histname.push_back("hSiStripStripADC_TECW6");
  histnames.push_back(23);

  vector<string> tecw7histname;
  tecw7histname.push_back("hSiStripn_TECW7");
  tecw7histname.push_back("hSiStripADC_TECW7");
  tecw7histname.push_back("hSiStripStripADC_TECW7");
  histnames.push_back(24);

  vector<string> tecw8histname;
  tecw8histname.push_back("hSiStripn_TECW8");
  tecw8histname.push_back("hSiStripADC_TECW8");
  tecw8histname.push_back("hSiStripStripADC_TECW8");
  histnames.push_back(25);

  vector<string> brl1histname;
  brl1histname.push_back("hSiPixeln_BRL1");
  brl1histname.push_back("hSiPixelADC_BRL1");
  brl1histname.push_back("hSiPixelRow_BRL1");
  brl1histname.push_back("hSiPixelColumn_BRL1");
  histnames.push_back(26);

  vector<string> brl2histname;
  brl2histname.push_back("hSiPixeln_BRL2");
  brl2histname.push_back("hSiPixelADC_BRL2");
  brl2histname.push_back("hSiPixelRow_BRL2");
  brl2histname.push_back("hSiPixelColumn_BRL2");
  histnames.push_back(27);

  vector<string> brl3histname;
  brl3histname.push_back("hSiPixeln_BRL3");
  brl3histname.push_back("hSiPixelADC_BRL3");
  brl3histname.push_back("hSiPixelRow_BRL3");
  brl3histname.push_back("hSiPixelColumn_BRL3");
  histnames.push_back(28);

  vector<string> fwd1phistname;
  fwd1phistname.push_back("hSiPixeln_FWD1p");
  fwd1phistname.push_back("hSiPixelADC_FWD1p");
  fwd1phistname.push_back("hSiPixelRow_FWD1p");
  fwd1phistname.push_back("hSiPixelColumn_FWD1p");
  histnames.push_back(29);

  vector<string> fwd1nhistname;
  fwd1nhistname.push_back("hSiPixeln_FWD1n");
  fwd1nhistname.push_back("hSiPixelADC_FWD1n");
  fwd1nhistname.push_back("hSiPixelRow_FWD1n");
  fwd1nhistname.push_back("hSiPixelColumn_FWD1n");
  histnames.push_back(30);

  vector<string> fwd2phistname;
  fwd2phistname.push_back("hSiPixeln_FWD2p");
  fwd2phistname.push_back("hSiPixelADC_FWD2p");
  fwd2phistname.push_back("hSiPixelRow_FWD2p");
  fwd2phistname.push_back("hSiPixelColumn_FWD2p");
  histnames.push_back(31);

  vector<string> fwd2nhistname;
  fwd2nhistname.push_back("hSiPixeln_FWD2n");
  fwd2nhistname.push_back("hSiPixelADC_FWD2n");
  fwd2nhistname.push_back("hSiPixelRow_FWD2n");
  fwd2nhistname.push_back("hSiPixelColumn_FWD2n");
  histnames.push_back(32);

  vector<string> mb1histname;
  mb1histname.push_back("hDtMuonn_MB1");
  mb1histname.push_back("hDtLayer_MB1");
  mb1histname.push_back("hDtMuonTime_MB1");
  mb1histname.push_back("hDtMuonTimevLayer_MB1");
  histnames.push_back(33);

  vector<string> mb2histname;
  mb2histname.push_back("hDtMuonn_MB2");
  mb2histname.push_back("hDtLayer_MB2");
  mb2histname.push_back("hDtMuonTime_MB2");
  mb2histname.push_back("hDtMuonTimevLayer_MB2");
  histnames.push_back(34);

  vector<string> mb3histname;
  mb3histname.push_back("hDtMuonn_MB3");
  mb3histname.push_back("hDtLayer_MB3");
  mb3histname.push_back("hDtMuonTime_MB3");
  mb3histname.push_back("hDtMuonTimevLayer_MB3");
  histnames.push_back(35);

  vector<string> mb4histname;
  mb4histname.push_back("hDtMuonn_MB4");
  mb4histname.push_back("hDtLayer_MB4");
  mb4histname.push_back("hDtMuonTime_MB4");
  mb4histname.push_back("hDtMuonTimevLayer_MB4");
  histnames.push_back(36);

  vector<string> cscstriphistname;
  cscstriphistname.push_back("hCSCStripn");
  cscstriphistname.push_back("hCSCStripADC");
  histnames.push_back(37);

  vector<string> cscwirehistname;
  cscwirehistname.push_back("hCSCWiren");
  cscwirehistname.push_back("hCSCWireTime");
  histnames.push_back(38);

  //loop through histograms to prepare output
  for (Int_t i = 0; i < histnames.size(); ++i) {

    Int_t page = histnames[i];

    vector<string> names;
    bool logy3 = kFALSE;
    bool logy2 = kFALSE;

    // setup canvas depending on group of plots
    TCanvas *Canvas;

    if (page == 0) {
      logy3 = kTRUE;
      names = eehistname;
      //Canvas = new TCanvas("eecal","eecal",cWidth,cHeight);
      Canvas = new TCanvas("eecal", "eecal");
      Canvas->Divide(2,3);
      myCanvas = Canvas;
      myCanvas->cd(0);
    }
    if (page == 1) {
      logy3 = kTRUE;
      names = ebhistname;
      Canvas = new TCanvas("ebcal","ebcal");
      Canvas->Divide(2,3);
      myCanvas = Canvas;
      myCanvas->cd(0);

    }
    if (page == 2) {
      names = eshistname;
      Canvas = new TCanvas("escal","escal");
      Canvas->Divide(2,2);
      myCanvas = Canvas;
      myCanvas->cd(0);

    }
    if (page == 3) {
      logy3 = kTRUE;
      names = hbhistname;
      Canvas = new TCanvas("hbcal","hbcal");
      Canvas->Divide(2,3);
      myCanvas = Canvas;
      myCanvas->cd(0);

    }
    if (page == 4) {
      logy3 = kTRUE;
      names = hehistname;
      Canvas = new TCanvas("hecal","hecal");
      Canvas->Divide(2,3);
      myCanvas = Canvas;
      myCanvas->cd(0);

    }
    if (page == 5) {
      logy3 = kTRUE;
      names = hohistname;
      Canvas = new TCanvas("hocal","hocal");
      Canvas->Divide(2,3);
      myCanvas = Canvas;
      myCanvas->cd(0);

    }
    if (page == 6) {
      logy3 = kTRUE;
      names = hfhistname;
      Canvas = new TCanvas("hfcal","hfcal");
      Canvas->Divide(2,3);
      myCanvas = Canvas;
      myCanvas->cd(0);

    }
    if (page == 7) {
      names = tibl1histname;
      Canvas = new TCanvas("tibl1","tibl1");
      Canvas->Divide(2,2);
      myCanvas = Canvas;
      myCanvas->cd(0);

    }
    if (page == 8) {
      names = tibl2histname;
      Canvas = new TCanvas("tibl2","tibl2");
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 9) {
      names = tibl3histname;
      Canvas = new TCanvas("tibl3","tibl3");
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 10) {
      names = tibl4histname;
      Canvas = new TCanvas("tibl4","tibl4");
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 11) {
      names = tobl1histname;
      Canvas = new TCanvas("tobl1","tobl1");
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 12) {
      names = tobl2histname;
      Canvas = new TCanvas("tobl2","tobl2" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 13) {
      names = tobl3histname;
      Canvas = new TCanvas("tobl3","tobl3" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 14) {
      names = tobl4histname;
      Canvas = new TCanvas("tobl4","tobl4" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 15) {
      names = tidw1histname;
      Canvas = new TCanvas("tidw1","tidw1" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 16) {
      names = tidw2histname;
      Canvas = new TCanvas("tidw2","tidw2" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 17) {
      names = tidw3histname;
      Canvas = new TCanvas("tidw3","tidw3" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 18) {
      names = tecw1histname;
      Canvas = new TCanvas("tecw1","tecw1" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 19) {
      names = tecw2histname;
      Canvas = new TCanvas("tecw2","tecw2" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 20) {
      names = tecw3histname;
      Canvas = new TCanvas("tecw3","tecw3" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 21) {
      names = tecw4histname;
      Canvas = new TCanvas("tecw4","tecw4" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 22) {
      names = tecw5histname;
      Canvas = new TCanvas("tecw5","tecw5" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 23) {
      names = tecw6histname;
      Canvas = new TCanvas("tecw6","tecw6" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 24) {
      names = tecw7histname;
      Canvas = new TCanvas("tecw7","tecw7" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 25) {
      names = tecw8histname;
      Canvas = new TCanvas("tecw8","tecw8" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 26) {
      logy2 = kTRUE;
      names = brl1histname;
      Canvas = new TCanvas("brl1","brl1" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 27) {
      logy2 = kTRUE;
      names = brl2histname;
      Canvas = new TCanvas("brl2","brl2" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 28) {
      logy2 = kTRUE;
      names = brl3histname;
      Canvas = new TCanvas("brl3","brl3" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 29) {
      logy2 = kTRUE;
      names = fwd1phistname;
      Canvas = new TCanvas("fwd1p","fwd1p" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 30) {
      logy2 = kTRUE;
      names = fwd1nhistname;
      Canvas = new TCanvas("fwd1n","fwd1n" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 31) {
      logy2 = kTRUE;
      names = fwd2phistname;
      Canvas = new TCanvas("fwd2p","fwd2p" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 32) {
      logy2 = kTRUE;
      names = fwd2nhistname;
      Canvas = new TCanvas("fwd2n","fwd2n" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 33) {
      names = mb1histname;
      Canvas = new TCanvas("mb1","mb1" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 34) {
      names = mb2histname;
      Canvas = new TCanvas("mb2","mb2" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 35) {
      names = mb3histname;
      Canvas = new TCanvas("mb3","mb3" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 36) {
      names = mb4histname;
      Canvas = new TCanvas("mb4","mb4" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 37) {
      names = cscstriphistname;
      Canvas = new TCanvas("cscstrip","cscstrip" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }
    if (page == 38) {
      names = cscwirehistname;
      Canvas = new TCanvas("cscwire","cscwire" );
      Canvas->Divide(2,2);
      myCanvas = Canvas; myCanvas->cd(0);
    }

    // loop through plots
    for (Int_t j = 0; j < names.size(); ++j) {

      // extract plot from both files
      //TH1F *sh = (TH1F*)sfile->Get(names[j].c_str());
      TH1F *sh; TH1F *rh;
      if(page >= 0 && page < 3)
	{
	  TString hpath = "DQMData/GlobalDigisV/ECals/"+names[j];
	}
      if(page >= 3 && page < 7)
	{
	  TString hpath = "DQMData/GlobalDigisV/HCals/"+names[j];
	}
      if(page >=7 && page < 26)
	{
	  TString hpath = "DQMData/GlobalDigisV/SiStrips/"+names[j];
	}
      if(page >= 26 && page < 33)
	{
	  TString hpath = "DQMData/GlobalDigisV/SiPixels/"+names[j];
	}
      if(page >= 33 && page < 39)
	{
	  TString hpath = "DQMData/GlobalDigisV/Muons/"+names[j];
	}
	sh = (TH1F*)sfile->Get(hpath);
	rh = (TH1F*)rfile->Get(hpath);

      

      if (!sh) cout << names[j].c_str() << "doesn't exist" << endl;
      sh->SetLineColor(scolor);
      sh->SetLineWidth(slinewidth);
      sh->SetLineStyle(stype);
      Double_t smax = sh->GetMaximum();
      //TH1F *rh = (TH1F*)rfile->Get(names[j].c_str());
      rh->SetLineColor(rcolor);
      rh->SetLineWidth(rlinewidth);
      rh->SetLineStyle(rtype);
      Double_t rmax = rh->GetMaximum();

      // find the probability value for the two plots being from the same
      // source
      //double pv = rh->Chi2Test(sh,"OU");
      double pv = rh->KolmogorovTest(sh);
      std::strstream buf;
      std::string value;
      buf << "PV=" << pv <<std::endl;
      buf >> value;

      // set maximum of y axis
      Double_t max = smax;
      if (rmax > smax) max = rmax;
      max *= 1.10;
      rh->SetMaximum(max);

      // zoom in on x axis
      if (zoomx) {
	Int_t nbins = rh->GetNbinsX();
	Int_t rlbin, slbin;
	Int_t rfbin = -1, sfbin = -1;
	for (Int_t k = 1; k < nbins; ++k) {
	  Int_t rbincont = rh->GetBinContent(k);
	  if (rbincont != 0) {
	    rlbin = k;
	    if (rfbin == -1) rfbin = k;
	  }
	  Int_t sbincont = sh->GetBinContent(k);
	  if (sbincont != 0) {
	    slbin = k;
	    if (sfbin == -1) sfbin = k;
	  }
	}
	Int_t binmin = rfbin, binmax = rlbin+1;
	if (sfbin < binmin) binmin = sfbin;
	if (slbin > binmax) binmax = slbin+1;
	rh->SetAxisRange(rh->GetBinLowEdge(binmin),rh->GetBinLowEdge(binmax));
      }

      // make plots
      myCanvas->cd(j+1);
      //if (j == 0) gPad->SetLogy();
      //if (logy2 && j == 1) gPad->SetLogy();
      //if (logy3 && j == 2) gPad->SetLogy();
      rh->Draw();
      sh->Draw("sames");

      te->DrawTextNDC(0.15,0.8, value.c_str());
      std::cout << "[OVAL] " << rh->GetName() 
		<< " PV = " << pv << std::endl;
    }
    myCanvas->Print(psfile);

  } 

  // close output ps file
  myCanvas->Print(psfileclose);

  // close root files
  rfile->Close();
  sfile->Close();

  TString cmnd;
  cmnd = "ps2pdf "+psfile+" "+filename+".pdf";
  gSystem->Exec(cmnd);
  cmnd = "rm "+psfile;
  gSystem->Exec(cmnd); 

  return;
}

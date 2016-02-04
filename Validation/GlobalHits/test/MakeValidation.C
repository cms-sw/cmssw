///////////////////////////////////////////////////////////////////////////////
// Macro to compare histograms from the GlobalHitsProducer for validation
//
// root -b -q MakeValidation.C
///////////////////////////////////////////////////////////////////////////////
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TString.h"

void MakeValidation(TString sfilename = "GlobalHitsHistograms.root",
		    TString rfilename = "GlobalHitsHistograms-reference.root",
		    TString filename = "GlobalHitsHistogramsCompare")
{
  gROOT->Reset();

  // zoom in on x-axis of plots
  bool zoomx = false;

  // create new histograms for comparison
  //gROOT->LoadMacro("MakeHistograms.C");
  //MakeHistograms();

  // setup names
  //TString sfilename = "GlobalHitsHistograms.root";
  //TString rfilename = "GlobalHitsHistograms-reference.root";

  // clear memory of file names
  delete gROOT->GetListOfFiles()->FindObject(sfilename);
  delete gROOT->GetListOfFiles()->FindObject(rfilename);

  // open reference file
  TFile *sfile = new TFile(sfilename);
  TFile *rfile = new TFile(rfilename);

  // create canvas
  Int_t cWidth = 928, cHeight = 1218;
  //TCanvas *myCanvas = new TCanvas("globalhits","globalhits",cWidth,cHeight);
  TCanvas *myCanvas = new TCanvas("globalhits","globalhits");  
  //myCanvas->Size(21.59, 27.94);

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
  Int_t rtype = kDotted;
  Int_t stype = kSolid;
  Int_t scolor = kRed;
  Int_t linewidth = 2;

  vector<Int_t> histnames;

  vector<string> mchistname;
  mchistname.push_back("hMCRGP1");
  mchistname.push_back("hMCRGP1");
  histnames.push_back(0);

  vector<string> vtxhistname;
  vtxhistname.push_back("hMCG4Vtx1");
  vtxhistname.push_back("hMCG4Vtx2");
  vtxhistname.push_back("hGeantVtxX1");
  vtxhistname.push_back("hGeantVtxX2");
  vtxhistname.push_back("hGeantVtxY1");
  vtxhistname.push_back("hGeantVtxY2");
  vtxhistname.push_back("hGeantVtxZ1");
  vtxhistname.push_back("hGeantVtxZ2");
  histnames.push_back(1);

  vector<string> trkhistname;
  trkhistname.push_back("hMCG4Trk1");
  trkhistname.push_back("hMCG4Trk2");
  trkhistname.push_back("hGeantTrkPt");
  trkhistname.push_back("hGeantTrkE");
  histnames.push_back(2);

  vector<string> ecalhistname;
  ecalhistname.push_back("hCaloEcal1");
  ecalhistname.push_back("hCaloEcal2");
  ecalhistname.push_back("hCaloEcalE1");
  ecalhistname.push_back("hCaloEcalE2");
  ecalhistname.push_back("hCaloEcalToF1");
  ecalhistname.push_back("hCaloEcalToF2");
  ecalhistname.push_back("hCaloEcalPhi");
  ecalhistname.push_back("hCaloEcalEta");
  histnames.push_back(3);

  vector<string> preshhistname;
  preshhistname.push_back("hCaloPreSh1");
  preshhistname.push_back("hCaloPreSh2");
  preshhistname.push_back("hCaloPreShE1");
  preshhistname.push_back("hCaloPreShE2");
  preshhistname.push_back("hCaloPreShToF1");
  preshhistname.push_back("hCaloPreShToF2");
  preshhistname.push_back("hCaloPreShPhi");
  preshhistname.push_back("hCaloPreShEta");
  histnames.push_back(4);

  vector<string> hcalhistname;
  hcalhistname.push_back("hCaloHcal1");
  hcalhistname.push_back("hCaloHcal2");
  hcalhistname.push_back("hCaloHcalE1");
  hcalhistname.push_back("hCaloHcalE2");
  hcalhistname.push_back("hCaloHcalToF1");
  hcalhistname.push_back("hCaloHcalToF2");
  hcalhistname.push_back("hCaloHcalPhi");
  hcalhistname.push_back("hCaloHcalEta");
  histnames.push_back(5);

  vector<string> pxlhistname;
  pxlhistname.push_back("hTrackerPx1");
  pxlhistname.push_back("hTrackerPx2");
  pxlhistname.push_back("hTrackerPxPhi");
  pxlhistname.push_back("hTrackerPxEta");
  pxlhistname.push_back("hTrackerPxBToF");
  pxlhistname.push_back("hTrackerPxBR");
  pxlhistname.push_back("hTrackerPxFToF");
  pxlhistname.push_back("hTrackerPxFZ");
  histnames.push_back(6);

  vector<string> sihistname;
  sihistname.push_back("hTrackerSi1");
  sihistname.push_back("hTrackerSi2");
  sihistname.push_back("hTrackerSiPhi");
  sihistname.push_back("hTrackerSiEta");
  sihistname.push_back("hTrackerSiBToF");
  sihistname.push_back("hTrackerSiBR");
  sihistname.push_back("hTrackerSiFToF");
  sihistname.push_back("hTrackerSiFZ");
  histnames.push_back(7);

  vector<string> muonhistname;
  muonhistname.push_back("hMuon1");
  muonhistname.push_back("hMuon2");
  muonhistname.push_back("hMuonPhi");
  muonhistname.push_back("hMuonEta");
  histnames.push_back(8);

  vector<string> cschistname;
  cschistname.push_back("hMuonCscToF1");
  cschistname.push_back("hMuonCscToF2");
  cschistname.push_back("hMuonCscZ");
  histnames.push_back(9);

  vector<string> dthistname;
  dthistname.push_back("hMuonDtToF1");
  dthistname.push_back("hMuonDtToF2");
  dthistname.push_back("hMuonDtR");
  histnames.push_back(10);

  vector<string> rpchistname;
  rpchistname.push_back("hMuonRpcFToF1");
  rpchistname.push_back("hMuonRpcFToF2");
  rpchistname.push_back("hMuonRpcFZ");
  rpchistname.push_back("hMuonRpcBToF1");
  rpchistname.push_back("hMuonRpcBToF2");
  rpchistname.push_back("hMuonRpcBR");
  histnames.push_back(11);

  //loop through histograms to prepare output
  for (Int_t i = 0; i < histnames.size(); ++i) {

    vector<string> names;

    // setup canvas depending on group of plots
    TCanvas *Canvas;
    if (i == 0) {
      names = mchistname;
      //Canvas = new TCanvas("MCRGP","MCRGP",cWidth,cHeight);
      Canvas = new TCanvas("MCRGP","MCRGP");
      //Canvas->Size(21.59, 27.94);
      Canvas->Divide(1,2);
      myCanvas = Canvas;
      myCanvas->cd(0);
      //label->DrawLatex(0.5,1.00,"Monte Carlo RawGenPart");
    }
    if (i == 1) {
      names = vtxhistname;
      //Canvas = new TCanvas("G4Vtx","G4Vtx",cWidth,cHeight);
      Canvas = new TCanvas("G4Vtx","G4Vtx");
      //Canvas->Size(21.59, 27.94);
      Canvas->Divide(2,4);
      myCanvas = Canvas;
      myCanvas->cd(0);
      //label->DrawLatex(0.5,1.00,"Geant4 Vertices");
    }
    if (i == 2) {
      names = trkhistname;
      //Canvas = new TCanvas("G4Trk","G4Trk",cWidth,cWidth);
      Canvas = new TCanvas("G4Trk","G4Trk");
      //Canvas->Size(21.59, 27.94);
      Canvas->Divide(2,2);
      myCanvas = Canvas;
      myCanvas->cd(0);
      //label->DrawLatex(0.5,1.00,"Geant4 Tracks");
    }
    if (i == 3) {
      names = ecalhistname;
      //Canvas = new TCanvas("ECalHits","ECalHits",cWidth,cHeight);
      Canvas = new TCanvas("ECalHits","ECalHits");
      //Canvas->Size(21.59, 27.94);
      Canvas->Divide(2,4);
      myCanvas = Canvas;
      myCanvas->cd(0);
      //label->DrawLatex(0.5,1.00,"ECal Information");
    }
    if (i == 4) {
      names = preshhistname;
      //Canvas = new TCanvas("PreShHits","PreShHits",cWidth,cHeight);
      Canvas = new TCanvas("PreShHits","PreShHits");
      //Canvas->Size(21.59, 27.94);
      Canvas->Divide(2,4);
      myCanvas = Canvas;
      myCanvas->cd(0);
      //label->DrawLatex(0.5,1.00,"PreShower Information");
    }
    if (i == 5) {
      names = hcalhistname;
      //Canvas = new TCanvas("HCalHits","HCalHits",cWidth,cHeight);
      Canvas = new TCanvas("HCalHits","HCalHits");
      //Canvas->Size(21.59, 27.94);
      Canvas->Divide(2,4);
      myCanvas = Canvas;
      myCanvas->cd(0);
      //label->DrawLatex(0.5,1.00,"HCal Information");
    }
    if (i == 6) {
      names = pxlhistname;
      //Canvas = new TCanvas("PixelHits","PixelHits",cWidth,cHeight);
      Canvas = new TCanvas("PixelHits","PixelHits");
      //Canvas->Size(21.59, 27.94);
      Canvas->Divide(2,4);
      myCanvas = Canvas;
      myCanvas->cd(0);
      //label->DrawLatex(0.5,1.00,"Pixel Information");
    }
    if (i == 7) {
      names = sihistname;
      //Canvas = new TCanvas("StripHits","StripHits",cWidth,cHeight);
      Canvas = new TCanvas("StripHits","StripHits");
      //Canvas->Size(21.59, 27.94);
      Canvas->Divide(2,4);
      myCanvas = Canvas;
      myCanvas->cd(0);
      //label->DrawLatex(0.5,1.00,"Strip Information");
    }
    if (i == 8) {
      names = muonhistname;
      //Canvas = new TCanvas("MuonHits","MuonHits",cWidth,cWidth);
      Canvas = new TCanvas("MuonHits","MuonHits");
      //Canvas->Size(21.59, 27.94);
      Canvas->Divide(2,2);
      myCanvas = Canvas;
      myCanvas->cd(0);
      //label->DrawLatex(0.5,1.00,"Muon Information");
    }
    if (i == 9) {
      names = cschistname;
      //Canvas = new TCanvas("MuonCscHits","MuonCscHits",cWidth,cWidth);
      Canvas = new TCanvas("MuonCscHits","MuonCscHits");
      //Canvas->Size(21.59, 27.94);
      Canvas->Divide(2,2);
      myCanvas = Canvas;
      myCanvas->cd(0);
      //label->DrawLatex(0.5,1.00,"Muon CSC Information");
    }
    if (i == 10) {
      names = dthistname;
      //Canvas = new TCanvas("MuonDtHits","MuonDtHits",cWidth,cWidth);
      Canvas = new TCanvas("MuonDtHits","MuonDtHits");
      //Canvas->Size(21.59, 27.94);
      Canvas->Divide(2,2);
      myCanvas = Canvas;
      myCanvas->cd(0);
      //label->DrawLatex(0.5,1.00,"Muon DT Information");
    }
    if (i == 11) {
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

      TH1F *sh;
      TH1F *rh;

      // set axis info for the histograms
      if (i == 0) {
	TString hpath = "DQMData/GlobalHitsV/MCGeant/"+names[j];
	sh = (TH1F*)sfile->Get(hpath);
	rh = (TH1F*)rfile->Get(hpath);
      }
      if (i == 1) {
	TString hpath = "DQMData/GlobalHitsV/MCGeant/"+names[j];
	sh = (TH1F*)sfile->Get(hpath);	
	rh = (TH1F*)rfile->Get(hpath);
      }
      if (i == 2) {
	TString hpath = "DQMData/GlobalHitsV/MCGeant/"+names[j];
	sh = (TH1F*)srcfile->Get(hpath);
	rh = (TH1F*)rfile->Get(hpath);
      }
      if (i == 3 || i == 4 || i == 5) {
	if (i == 3 || i == 4) {
	  TString hpath = "DQMData/GlobalHitsV/ECals/"+names[j];
	  sh = (TH1F*)sfile->Get(hpath);
	  rh = (TH1F*)rfile->Get(hpath);
	}
	if (i == 5) {
	  TString hpath = "DQMData/GlobalHitsV/HCals/"+names[j];
	  sh = (TH1F*)sfile->Get(hpath);
	  rh = (TH1F*)rfile->Get(hpath);
	}	
      }
      if (i == 6) {
	TString hpath = "DQMData/GlobalHitsV/SiPixels/"+names[j];
	sh = (TH1F*)sfile->Get(hpath);
	rh = (TH1F*)rfile->Get(hpath);
      }
      if (i == 7) {
	TString hpath = "DQMData/GlobalHitsV/SiStrips/"+names[j];
	sh = (TH1F*)sfile->Get(hpath);
	rh = (TH1F*)rfile->Get(hpath);
      }      
      if (i == 8) {
	TString hpath = "DQMData/GlobalHitsV/Muons/"+names[j];
	sh = (TH1F*)sfile->Get(hpath);
	rh = (TH1F*)rfile->Get(hpath);
      }
      if (i == 9) {
	TString hpath = "DQMData/GlobalHitsV/Muons/"+names[j];
	sh = (TH1F*)sfile->Get(hpath);
	rh = (TH1F*)rfile->Get(hpath);
      }
      if (i == 10) {
	TString hpath = "DQMData/GlobalHitsV/Muons/"+names[j];
	sh = (TH1F*)sfile->Get(hpath);
	rh = (TH1F*)rfile->Get(hpath);
      }
      if (i == 11) {
	TString hpath = "DQMData/GlobalHitsV/Muon/"+names[j];
	sh = (TH1F*)sfile->Get(hpath);
	rh = (TH1F*)rfile->Get(hpath);
      }

      // extract plot from both files
      //TH1F *sh = (TH1F*)sfile->Get(names[j].c_str());
      sh->SetLineColor(scolor);
      sh->SetLineWidth(linewidth);
      sh->SetLineStyle(stype);
      Double_t smax = sh->GetMaximum();
      //TH1F *rh = (TH1F*)rfile->Get(names[j].c_str());
      rh->SetLineColor(rcolor);
      rh->SetLineWidth(linewidth);
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
      //gPad->SetLogy();
      sh->Draw();
      rh->Draw("sames");

      te->DrawTextNDC(0.15,0.8, value.c_str());
      std::cout << "[OVAL] " << rh->GetName() 
		<< " PV = " << pv << std::endl;
    }
    myCanvas->Print(psfile);

  } // end loop through histnames

  // close output ps file
  myCanvas->Print(psfileclose);

  // close root files
  rfile->Close();
  sfile->Close();

  //convert to pdf
  TString cmnd;
  cmnd = "ps2pdf "+psfile+" "+filename+".pdf";
  gSystem->Exec(cmnd);
  cmnd = "rm "+psfile;
  gSystem->Exec(cmnd);  

  return;
}

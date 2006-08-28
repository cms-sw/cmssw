///////////////////////////////////////////////////////////////////////////////
// Macro to compare histograms from the GlobalHitsProducer for validation
//
// root -b -q MakeValidation.C
///////////////////////////////////////////////////////////////////////////////
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TString.h"

void MakeValidation()
{
  gROOT->Reset();

  // zoom in on x-axis of plots
  bool zoomx = false;

  // create new histograms for comparison
  gROOT->LoadMacro("MakeHistograms.C");
  MakeHistograms();

  // setup names
  TString sfilename = "GlobalHitsHistograms.root";
  TString rfilename = "GlobalHitsHistograms-reference.root";

  // clear memory of file names
  delete gROOT->GetListOfFiles()->FindObject(sfilename);
  delete gROOT->GetListOfFiles()->FindObject(rfilename);

  // open reference file
  TFile *sfile = new TFile(sfilename);
  TFile *rfile = new TFile(rfilename);

  // create canvas
  Int_t cWidth = 928, cHeight = 1218;
  TCanvas *myCanvas = new TCanvas("globalhits","globalhits",cWidth,cHeight);

  // open output ps file
  TString filename = "GlobalHitsHistogramsCompare";
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
  mchistname.push_back("hMCRGP_1003");
  mchistname.push_back("hMCRGP2_1013");
  histnames.push_back(0);

  vector<string> vtxhistname;
  vtxhistname.push_back("hMCG4Vtx_1001");
  vtxhistname.push_back("hMCG4Vtx2_1011");
  vtxhistname.push_back("hGeantVtxX_1101");
  vtxhistname.push_back("hGeantVtxY_1102");
  vtxhistname.push_back("hGeantVtxZ_1103");
  histnames.push_back(1);

  vector<string> trkhistname;
  trkhistname.push_back("hMCG4Trk_1002");
  trkhistname.push_back("hMCG4Trk2_1012");
  trkhistname.push_back("hGeantTrkPt_1201");
  trkhistname.push_back("hGeantTrkE_1202");
  histnames.push_back(2);

  vector<string> ecalhistname;
  ecalhistname.push_back("hCaloEcal_2101");
  ecalhistname.push_back("hCaloEcal2_2111");
  ecalhistname.push_back("hCaloEcalE_2102");
  ecalhistname.push_back("hCaloEcalE2_2112");
  ecalhistname.push_back("hCaloEcalToF_2103");
  ecalhistname.push_back("hCaloEcalToF2_2113");
  ecalhistname.push_back("hCaloEcalPhi_2104");
  ecalhistname.push_back("hCaloEcalEta_2105");
  histnames.push_back(3);

  vector<string> preshhistname;
  preshhistname.push_back("hCaloPreSh_2201");
  preshhistname.push_back("hCaloPreSh2_2211");
  preshhistname.push_back("hCaloPreShE_2202");
  preshhistname.push_back("hCaloPreShE2_2212");
  preshhistname.push_back("hCaloPreShToF_2203");
  preshhistname.push_back("hCaloPreShToF2_2213");
  preshhistname.push_back("hCaloPreShPhi_2204");
  preshhistname.push_back("hCaloPreShEta_2205");
  histnames.push_back(4);

  vector<string> hcalhistname;
  hcalhistname.push_back("hCaloHcal_2301");
  hcalhistname.push_back("hCaloHcal2_2311");
  hcalhistname.push_back("hCaloHcalE_2302");
  hcalhistname.push_back("hCaloHcalE2_2312");
  hcalhistname.push_back("hCaloHcalToF_2303");
  hcalhistname.push_back("hCaloHcalToF2_2313");
  hcalhistname.push_back("hCaloHcalPhi_2304");
  hcalhistname.push_back("hCaloHcalEta_2305");
  histnames.push_back(5);

  vector<string> pxlhistname;
  pxlhistname.push_back("hTrackerPx_3101");
  pxlhistname.push_back("hTrackerPx2_3111");
  pxlhistname.push_back("hTrackerPxPhi_3102");
  pxlhistname.push_back("hTrackerPxEta_3103");
  pxlhistname.push_back("hTrackerPxBToF_3104");
  pxlhistname.push_back("hTrackerPxBR_3106");
  pxlhistname.push_back("hTrackerPxFToF_3105");
  pxlhistname.push_back("hTrackerPxFZ_3107");
  histnames.push_back(6);

  vector<string> sihistname;
  sihistname.push_back("hTrackerSi_3201");
  sihistname.push_back("hTrackerSi2_3211");
  sihistname.push_back("hTrackerSiPhi_3202");
  sihistname.push_back("hTrackerSiEta_3203");
  sihistname.push_back("hTrackerSiBToF_3204");
  sihistname.push_back("hTrackerSiBR_3206");
  sihistname.push_back("hTrackerSiFToF_3205");
  sihistname.push_back("hTrackerSiFZ_3207");
  histnames.push_back(7);

  vector<string> muonhistname;
  muonhistname.push_back("hMuon_4001");
  muonhistname.push_back("hMuon2_4011");
  muonhistname.push_back("hMuonPhi_4002");
  muonhistname.push_back("hMuonEta_4003");
  histnames.push_back(8);

  vector<string> cschistname;
  cschistname.push_back("hMuonCscToF_4201");
  cschistname.push_back("hMuonCscToF2_4202");
  cschistname.push_back("hMuonCscZ_4203");
  histnames.push_back(9);

  vector<string> dthistname;
  dthistname.push_back("hMuonDtToF_4101");
  dthistname.push_back("hMuonDtToF2_4102");
  dthistname.push_back("hMuonDtR_4103");
  histnames.push_back(10);

  vector<string> rpchistname;
  rpchistname.push_back("hMuonRpcFToF_4304");
  rpchistname.push_back("hMuonRpcFToF2_4305");
  rpchistname.push_back("hMuonRpcFZ_4306");
  rpchistname.push_back("hMuonRpcBToF_4101");
  rpchistname.push_back("hMuonRpcBToF2_4102");
  rpchistname.push_back("hMuonRpcBR_4103");
  histnames.push_back(11);

  //loop through histograms to prepare output
  for (Int_t i = 0; i < histnames.size(); ++i) {

    vector<string> names;

    // setup canvas depending on group of plots
    TCanvas *Canvas;
    if (i == 0) {
      names = mchistname;
      Canvas = new TCanvas("MCRGP","MCRGP",cWidth,cHeight);
      Canvas->Divide(1,2);
      myCanvas = Canvas;
      myCanvas->cd(0);
      label->DrawLatex(0.5,1.00,"Monte Carlo RawGenPart");
    }
    if (i == 1) {
      names = vtxhistname;
      Canvas = new TCanvas("G4Vtx","G4Vtx",cWidth,cHeight);
      Canvas->Divide(2,3);
      myCanvas = Canvas;
      myCanvas->cd(0);
      label->DrawLatex(0.5,1.00,"Geant4 Vertices");
    }
    if (i == 2) {
      names = trkhistname;
      Canvas = new TCanvas("G4Trk","G4Trk",cWidth,cWidth);
      Canvas->Divide(2,2);
      myCanvas = Canvas;
      myCanvas->cd(0);
      label->DrawLatex(0.5,1.00,"Geant4 Tracks");
    }
    if (i == 3) {
      names = ecalhistname;
      Canvas = new TCanvas("ECalHits","ECalHits",cWidth,cHeight);
      Canvas->Divide(2,4);
      myCanvas = Canvas;
      myCanvas->cd(0);
      label->DrawLatex(0.5,1.00,"ECal Information");
    }
    if (i == 4) {
      names = preshhistname;
      Canvas = new TCanvas("PreShHits","PreShHits",cWidth,cHeight);
      Canvas->Divide(2,4);
      myCanvas = Canvas;
      myCanvas->cd(0);
      label->DrawLatex(0.5,1.00,"PreShower Information");
    }
    if (i == 5) {
      names = hcalhistname;
      Canvas = new TCanvas("HCalHits","HCalHits",cWidth,cHeight);
      Canvas->Divide(2,4);
      myCanvas = Canvas;
      myCanvas->cd(0);
      label->DrawLatex(0.5,1.00,"HCal Information");
    }
    if (i == 6) {
      names = pxlhistname;
      Canvas = new TCanvas("PixelHits","PixelHits",cWidth,cHeight);
      Canvas->Divide(2,4);
      myCanvas = Canvas;
      myCanvas->cd(0);
      label->DrawLatex(0.5,1.00,"Pixel Information");
    }
    if (i == 7) {
      names = sihistname;
      Canvas = new TCanvas("StripHits","StripHits",cWidth,cHeight);
      Canvas->Divide(2,4);
      myCanvas = Canvas;
      myCanvas->cd(0);
      label->DrawLatex(0.5,1.00,"Strip Information");
    }
    if (i == 8) {
      names = muonhistname;
      Canvas = new TCanvas("MuonHits","MuonHits",cWidth,cWidth);
      Canvas->Divide(2,2);
      myCanvas = Canvas;
      myCanvas->cd(0);
      label->DrawLatex(0.5,1.00,"Muon Information");
    }
    if (i == 9) {
      names = cschistname;
      Canvas = new TCanvas("MuonCscHits","MuonCscHits",cWidth,cWidth);
      Canvas->Divide(2,2);
      myCanvas = Canvas;
      myCanvas->cd(0);
      label->DrawLatex(0.5,1.00,"Muon CSC Information");
    }
    if (i == 10) {
      names = dthistname;
      Canvas = new TCanvas("MuonDtHits","MuonDtHits",cWidth,cWidth);
      Canvas->Divide(2,2);
      myCanvas = Canvas;
      myCanvas->cd(0);
      label->DrawLatex(0.5,1.00,"Muon DT Information");
    }
    if (i == 11) {
      names = rpchistname;
      Canvas = new TCanvas("MuonRpcHits","MuonRpcHits",cWidth,cWidth);
      Canvas->Divide(2,2);
      myCanvas = Canvas;
      myCanvas->cd(0);
      label->DrawLatex(0.5,1.00,"Muon RPC Information");
    }

    // loop through plots
    for (Int_t j = 0; j < names.size(); ++j) {

      // extract plot from both files
      TH1F *sh = (TH1F*)sfile->Get(names[j].c_str());
      sh->SetLineColor(scolor);
      sh->SetLineWidth(linewidth);
      sh->SetLineStyle(stype);
      Double_t smax = sh->GetMaximum();
      TH1F *rh = (TH1F*)rfile->Get(names[j].c_str());
      rh->SetLineColor(rcolor);
      rh->SetLineWidth(linewidth);
      rh->SetLineStyle(rtype);
      Double_t rmax = rh->GetMaximum();

      // find the probability value for the two plots being from the same
      // source
      double pv = rh->Chi2Test(sh,"OU");
      //double pv = rh->KolmogorovTest(sh);
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

  return;
}

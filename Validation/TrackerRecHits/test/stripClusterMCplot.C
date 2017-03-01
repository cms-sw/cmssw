//
//  Plots from StripClusterMCanalysis
//

/*
Generate plots produced exhibiting properties of Si strip clusters and 
their associated simTracks and simHits.
 */

void stripClusterMCplot(const char* ntupleFile="clusNtuple.root") {

  gROOT->Reset();

  // use the 'plain' style for plots (white backgrounds, etc)
  cout << "...using style 'Plain'\n";

  gROOT->SetStyle("Plain");

  // use bold lines and markers
  gStyle->SetMarkerStyle(8);  // non-scalable dot
  gStyle->SetHistLineWidth(2);
  gStyle->SetLineStyleString(2,"[12 12]"); // postscript dashes

  // For the fit/function:
  gStyle->SetOptFit(1);
  gStyle->SetFitFormat("5.4g");
  gStyle->SetFuncColor(4);
  gStyle->SetFuncStyle(1);
  gStyle->SetFuncWidth(2);

  //..Get rid of X error bars
  gStyle->SetErrorX(0.001);

  // do not display any of the standard histogram decorations
  gStyle->SetOptTitle(0);

  // put tick marks on top and RHS of plots
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetHistLineWidth(2);

  //  ===============================================================

  double Pi = 2*acos(0);

  const TCut oneSimHit = "NsimHits == 1";
  const TCut twoSimHit = "NsimHits == 2";
  const TCut threeSimHit = "NsimHits == 3";
  const TCut fourSimHit = "NsimHits == 4";
  const TCut oneTp = "Ntp == 1";
  const TCut twoTp = "Ntp == 2";
  const TCut threeTp = "Ntp == 3";
  const TCut fourTp = "Ntp == 4";

  // // Following are for the old process ID scheme
  // const TCut primary = "firstProcess == 2";
  // const TCut primarytwo = primary && "secondProcess == 2";
  // const TCut primarythree = primarytwo && "thirdProcess == 2";
  // const TCut primaryfour = primarythree && "fourthProcess == 2";
  // const TCut primDec = "firstProcess == 2 || firstProcess == 4";
  // const TCut primDectwo = primDec && "(secondProcess == 2 || secondProcess == 4)";
  // const TCut primDecthree = primDectwo && "(thirdProcess == 2 || thirdProcess == 4)";
  // const TCut primDecfour = primDecthree && "(fourthProcess == 2 || fourthProcess == 4)";

//
//  Particle interaction process codes for the new scheme are found in
// SimG4Core/Physics/src/G4ProcessTypeEnumerator.cc
// See also https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideMCTruth
//
  const TCut primary = "firstProcess == 0";
  const TCut primarytwo = primary && "secondProcess == 0";
  const TCut primarythree = primarytwo && "thirdProcess == 0";
  const TCut primaryfour = primarythree && "fourthProcess == 0";
  const TCut primDec = "firstProcess >= 201 && firstProcess <= 203";
  const TCut primDectwo = primDec && "(secondProcess >= 201 && secondProcess <= 203)";
  const TCut primDecthree = primDectwo && "(thirdProcess >= 201 && thirdProcess <= 203)";
  const TCut primDecfour = primDecthree && "(fourthProcess >= 201 && fourthProcess <= 203)";

  const TCut chgPi = "abs(firstPID) == 211";
  const TCut chgPitwo = chgPi && "abs(secondPID) == 211";
  const TCut chgPithree = chgPitwo && "abs(thirdPID) == 211";
  const TCut chgPifour = chgPithree && "abs(fourthPID) == 211";
  const TCut chgNotEp = "abs(firstPID) == 13 || abs(firstPID) == 211 || abs(firstPID) == 321";
  const TCut chgNotEptwo = chgNotEp && "(abs(secondPID) == 13 || abs(secondPID) == 211 || abs(secondPID) == 321)";
  const TCut chgNotEpthree = chgNotEptwo && "(abs(thirdPID) == 13 || abs(thirdPID) == 211 || abs(thirdPID) == 321)";
  const TCut chgNotEpfour = chgNotEpthree && "(abs(fourthPID) == 13 || abs(fourthPID) == 211 || abs(fourthPID) == 321)";
  const TCut chgNotE = "abs(firstPID) == 13 || abs(firstPID) == 211 || abs(firstPID) == 321 || abs(firstPID) == 2212";
  const TCut chgNotEtwo = chgNotE && "(abs(secondPID) == 13 || abs(secondPID) == 211 || abs(secondPID) == 321 || abs(secondPID) == 2212)";
  const TCut chgNotEthree = chgNotEtwo && "(abs(thirdPID) == 13 || abs(thirdPID) == 211 || abs(thirdPID) == 321 || abs(thirdPID) == 2212)";
  const TCut chgNotEfour = chgNotEthree && "(abs(fourthPID) == 13 || abs(fourthPID) == 211 || abs(fourthPID) == 321 || abs(fourthPID) == 2212)";
  const TCut chg = "abs(firstPID) == 11 || abs(firstPID) == 13 || abs(firstPID) == 211 || abs(firstPID) == 321 || abs(firstPID) == 2212";
  const TCut chgtwo = chg && "(abs(secondPID) == 11 || abs(secondPID) == 13 || abs(secondPID) == 211 || abs(secondPID) == 321 || abs(secondPID) == 2212)";
  const TCut chgthree = chg && "(abs(thirdPID) == 11 || abs(thirdPID) == 13 || abs(thirdPID) == 211 || abs(thirdPID) == 321 || abs(thirdPID) == 2212)";
  const TCut chgtwofour = chg && "(abs(fourthPID) == 11 || abs(fourthPID) == 13 || abs(fourthPID) == 211 || abs(fourthPID) == 321 || abs(fourthPID) == 2212)";

  const TCut Pmin = "firstPmag > 1";
  const TCut Pmintwo = Pmin && "secondPmag > 1";
  const TCut Pminthree = Pmintwo && "thirdPmag > 1";
  const TCut Pminfour = Pminthree && "fourthPmag > 1";
  const TCut saturated = "sat == 1";
  const TCut notEMnoise = "abs(firstPID) != 11 && firstPID != 22 && firstPID != 0";
  const TCut sym = "abs(firstTkChg-secondTkChg)/(firstTkChg+secondTkChg) < 0.9";

  void drawStats(TH1* hist, double& top);
  void plot_Ncross(TTree* tp);
  void plot_process(TTree* tp, const TCut& cut);
  void plot_process2(TTree* tp, const TCut& cut);
  void plot_PID(TTree* tree);
  void plot_PIDzoom(TTree* tree);
  void plot_Pmag(TTree* tree, const TCut& cut);
  void plot_charge(TTree* tree, const TCut& cut);
  void plot_Eloss(TTree* tree, const TCut& cut);
  void plot_chargeEloss(TTree* tree, const TCut& cut);
  void plot_width(TTree* tree, const TCut& cut);
  void plot_path(TTree* tree, const TCut& cut);
  void plot_path1(TTree* tree, const TCut& cut);
  void plot_NormCharge(TTree* tree, const TString& MIPdEdx, const TCut& cut1, const TCut& cut2, const TCut& cut3);
  void plot_NormEloss(TTree* tree, const TString& MIPdEdx, const TCut& cut1, const TCut& cut2, const TCut& cut3);
  void plot_chgTotPath(TTree* tree, const TString& MIPdEdx, const TCut& cut0, const TCut& cut1, const TCut& cut2, const TCut& cut3,
		       const char* title0, const char* title1, const char* title2, const char* title3 );
  void plot_ChgTnrm_v_Snrm(TTree* tree, const TString& MIPdEdx, const TCut& cut1, const TCut& cut2, const TCut& cut3);
  void plot_NormChgSpl(TTree* tree, const TString& MIPdEdx, const TCut& cut);
  void plot_NormChgNtp(TTree* tree, const TString& MIPdEdx, const TCut& cut);
  void plot_NrmChgWid(TTree* tree, const TString& MIPdEdx);
  void plot_ChgPath(TTree* tree, const TCut& cut);
  void plot_ChgAsym(TTree* tree, const TString& MIPdEdx, const TCut& cut);
  void plot_Asym(TTree* tree, const TCut& cut);

  TFile tfile(ntupleFile);

  TTree* clTree;
  if (tfile.Get("makeNtuple/ClusterNtuple")) 
    clTree = (TTree*) tfile.Get("makeNtuple/ClusterNtuple");
  else if (tfile.Get("ClusterNtuplizer/ClusterNtuple"))
    clTree = (TTree*) tfile.Get("ClusterNtuplizer/ClusterNtuple");
  else if (tfile.Get("StripClusterMCanalysis/ClusterNtuple"))
    clTree = (TTree*) tfile.Get("StripClusterMCanalysis/ClusterNtuple");
  else {
    cout << "No ntuple found in this file and directory" << endl;
    return;
  }

  // clTree->Print();

  TDirectory *rootdir = gDirectory->GetDirectory("Rint:"); // Without these 2 lines
  rootdir->cd();                     // the histo disappears when the macro exits.

// 298 ADC counts per mm for MIP:
#define MIPdEdxVal 3.36e-4
  char scharge[80];  sprintf(scharge, "%10.3e", MIPdEdxVal);
  TString MIPdEdx(scharge);
  cout << "MIPdEdx = " << MIPdEdx << endl;

  gStyle->SetOptStat("ormen");  plot_Ncross(clTree);
  gStyle->SetOptStat("orme");  plot_width(clTree, oneSimHit&&!saturated&&primary&&notEMnoise&&Pmin);
  gStyle->SetOptStat("euo");  plot_process(clTree, oneSimHit);
//   gStyle->SetOptStat(0);  plot_process2(clTree, twoTp);
  gStyle->SetOptStat(0);  plot_PID(clTree);
  gStyle->SetOptStat(0);  plot_PIDzoom(clTree);
  gStyle->SetOptStat("ormen");  plot_Pmag(clTree, oneSimHit&&!saturated&&primary&&chgNotE&&Pmin);
  gStyle->SetOptStat("ormen");  plot_charge(clTree, oneSimHit);
  gStyle->SetOptStat("ormen");  plot_Eloss(clTree, oneSimHit);
  gStyle->SetOptStat("ormen");  plot_chargeEloss(clTree, oneSimHit);
  gStyle->SetOptStat("oen");  plot_path(clTree, oneSimHit&&!saturated&&primary&&chgNotE&&Pmin);
  gStyle->SetOptStat("oen");  plot_path1(clTree, oneSimHit&&!saturated&&primary&&chgNotE&&Pmin);
  gStyle->SetOptStat("oen");  plot_NormCharge(clTree, MIPdEdx, oneSimHit&&!saturated&&primary&&chgNotE, twoTp&&!saturated&&sym, twoTp&&twoSimHit&&!saturated&&primarytwo&&chgNotEtwo&&sym);
  gStyle->SetOptStat("oen");  plot_NormEloss(clTree, (const TString)"350.", oneSimHit&&!saturated&&primary&&chgNotE, twoTp&&!saturated&&sym, twoTp&&twoSimHit&&!saturated&&primarytwo&&chgNotEtwo&&sym);

// //   gStyle->SetOptStat("oen");  plot_chgTotPath(clTree, MIPdEdx, oneSimHit&&!saturated, oneSimHit&&!saturated&&primary&&chgNotE, twoSimHit&&!saturated, twoTp&&twoSimHit&&!saturated&&primarytwo&&chgNotEtwo, "One hit", "One primary", "Two hits", "Two primaries");
// //   gStyle->SetOptStat("oen");  plot_chgTotPath(clTree, MIPdEdx, oneSimHit&&!saturated, twoSimHit&&!saturated, threeSimHit&&!saturated, fourSimHit&&!saturated, "One hit", "Two hits", "Three hits", "Four hits");
// //   gStyle->SetOptStat("oen");  plot_chgTotPath(clTree, MIPdEdx, oneSimHit&&!saturated&&primary&&chgNotE, twoTp&&twoSimHit&&!saturated&&primarytwo&&chgNotEtwo, threeTp&&threeSimHit&&!saturated&&primarythree&&chgNotEthree, fourTp&&fourSimHit&&!saturated&&primaryfour&&chgNotEfour, "One primary", "Two primaries", "Three primaries", "Four primaries");

  gStyle->SetOptStat("oen");  plot_ChgTnrm_v_Snrm(clTree, MIPdEdx, oneSimHit&&!saturated&&primary&&chgNotE, twoTp&&!saturated&&sym, twoTp&&twoSimHit&&!saturated&&primarytwo&&chgNotEtwo&&sym);
//   //  With no truth info for black and green curves:
//   gStyle->SetOptStat("oen");  plot_ChgTnrm_v_Snrm(clTree, MIPdEdx, !saturated, twoTp&&!saturated&&sym, twoTp&&twoSimHit&&!saturated&&primarytwo&&chgNotEtwo&&sym);
// //   gStyle->SetOptStat("orme");  plot_NormChgSpl(clTree, MIPdEdx, split);
//   gStyle->SetOptStat(0);  plot_NormChgNtp(clTree, MIPdEdx, !saturated);
//   gStyle->SetOptStat(0);  plot_NrmChgWid(clTree, MIPdEdx);
//   gStyle->SetOptStat("oen");  plot_ChgPath(clTree, twoTp&&twoSimHit&&!saturated&&primarytwo&&chgNotEtwo&&sym);
  gStyle->SetOptStat("oen");  plot_ChgAsym(clTree, MIPdEdx, twoTp&&twoSimHit&&!saturated&&primarytwo&&chgNotEtwo);
  gStyle->SetOptStat("oen");  plot_Asym(clTree, twoTp&&twoSimHit&&!saturated&&primarytwo&&chgNotEtwo);

//   clTree->draw("secondPmag:firstPmag", "twoTp&&twoSimHit", "box");
//   clTree->draw("secondPmag:firstPmag");
//   clTree->Draw("charge:firstTkChg+secondTkChg",
// 	      "Ntp==2&&NsimHits==2&&abs(firstPID)==211&&abs(secondPID)==211&&firstProcess==2&&secondProcess==2&&sat==0&&subDet==3","box");
//   clTree->Draw("secondProcess:firstProcess","Ntp==2&&NsimHits==2&&sat==0&&subDet==3","box");
//   clTree->Draw("abs(firstTkChg-secondTkChg)/(firstTkChg+secondTkChg)", "Ntp==2&&NsimHits==2&&abs(firstPID)==211&&abs(secondPID)==211&&firstProcess==2&&secondProcess==2&&sat==0&&subDet==3");
//   clTree->Draw("abs(firstTkChg-secondTkChg)/(firstTkChg+secondTkChg):firstTkChg+secondTkChg", "Ntp==2&&NsimHits==2&&abs(firstPID)==211&&abs(secondPID)==211&&firstProcess==2&&secondProcess==2&&sat==0&&subDet==3","box");
//   clTree->Draw("(1-2*tkFlip)*(firstTkChg-secondTkChg)/(firstTkChg+secondTkChg)", "Ntp==2&&NsimHits==2&&abs(firstPID)==211&&abs(secondPID)==211&&firstProcess==2&&secondProcess==2&&sat==0&&subDet==3");

}  // ---------------------------------------------------------------

void drawStats(TH1* hist, double& top) {
  TPaveStats* stats = (TPaveStats*)hist->FindObject("stats");
  if (stats) {
    stats->SetLineColor(hist->GetLineColor());
    stats->SetTextColor(hist->GetLineColor());
    double height = stats->GetY2NDC()-stats->GetY1NDC();
    if (top != 0.0) {
      stats->SetY2NDC(top);
      stats->SetY1NDC(top - height);
    }
    top = stats->GetY1NDC() - 0.005;
    stats->Draw();
  }
}  // ---------------------------------------------------------------

void plot_Ncross(TTree* tree) {

  TCanvas *canv1 = new TCanvas("canv1", "canv1", 700, 500);

  TH1F* hNtp = new TH1F("hNtp", "No. of crossing simTracks", 10, 0, 10);
  tree->Project("hNtp", "Ntp");
  hNtp->GetXaxis()->SetTitle("No. of track crossings");
  hNtp->SetLineColor(kBlue);
  hNtp->Draw();

  TH1F* hNhit = new TH1F("hNhit", "No. of SimHits", 10, 0, 10);
  tree->Project("hNhit", "NsimHits");
  hNhit->GetXaxis()->SetTitle("No. of SimHits");
  hNhit->SetLineColor(kRed);
  hNhit->Draw("sames");

  canv1->Update();  // without this the "stats" pointers are null
  double top = 0.0;
  drawStats(hNtp, top);
  drawStats(hNhit, top);
  canv1->Update();

  TLegend* Rtlegend = new TLegend(.32, .65, .72, .80, "");
  Rtlegend->AddEntry(hNtp, "simTracks", "L");
  Rtlegend->AddEntry(hNhit, "SimHits", "L");
  Rtlegend->Draw();
  canv1->Update();

  canv1->SaveAs("clusNtp_Ncross.pdf");

}  // ---------------------------------------------------------------

void plot_process(TTree* tree, const TCut& cut) {

  TCanvas *canv2 = new TCanvas("canv2", "canv2", 700, 500);

  TH1F* hProcess = new TH1F("hProcess", "Process that produced hit", 210, 0, 210);
  tree->Project("hProcess", "firstProcess", cut);
  hProcess->GetXaxis()->SetTitle("Process code");
  hProcess->SetLineColor(kBlue);
  hProcess->Draw();

  canv2->Update();  // without this the "stats" pointers are null

  canv2->SaveAs("clusNtp_process.pdf");

}  // ---------------------------------------------------------------

/*
void plot_process(TTree* tree, const TCut& cut) {

  TCanvas *canv2 = new TCanvas("canv2", "canv2", 700, 500);

  TH1F* hProcess = new TH1F("hProcess", "Process that produced hit", 20, 0, 20);
  tree->Project("hProcess", "firstProcess", cut);
//   hProcess->GetXaxis()->SetTitle("Process that produced simHit");
  hProcess->GetXaxis()->SetTitle("");
  hProcess->SetLineColor(kBlue);
  hProcess->GetXaxis()->SetBinLabel(1, "Undefined");
  hProcess->GetXaxis()->SetBinLabel(2, "Unknown");
  hProcess->GetXaxis()->SetBinLabel(3, "Primary");
  hProcess->GetXaxis()->SetBinLabel(4, "Hadronic");
  hProcess->GetXaxis()->SetBinLabel(5, "Decay");
  hProcess->GetXaxis()->SetBinLabel(6, "Compton");
  hProcess->GetXaxis()->SetBinLabel(7, "Annihilation");
  hProcess->GetXaxis()->SetBinLabel(8, "EIoni");
  hProcess->GetXaxis()->SetBinLabel(9, "HIoni");
  hProcess->GetXaxis()->SetBinLabel(10, "MuIoni");
  hProcess->GetXaxis()->SetBinLabel(11, "Photon");
  hProcess->GetXaxis()->SetBinLabel(12, "MuPairProd");
  hProcess->GetXaxis()->SetBinLabel(13, "Conversions");
  hProcess->GetXaxis()->SetBinLabel(14, "EBrem");
  hProcess->GetXaxis()->SetBinLabel(15, "SynchrotronRadiation");
  hProcess->GetXaxis()->SetBinLabel(16, "MuBrem");
  hProcess->GetXaxis()->SetBinLabel(17, "MuNucl");
  hProcess->Draw();

//   canv2->SetBottomMargin(.2);
//   gStyle->SetTitleOffset(1.1, "x");

  canv2->Update();  // without this the "stats" pointers are null

  canv2->SaveAs("clusNtp_process.pdf");

}  // ---------------------------------------------------------------

void plot_process2(TTree* tree, const TCut& cut) {

  TCanvas *canvProc2 = new TCanvas("canvProc2", "canvProc2", 700, 500);

  TH2F* hProcess2 = new TH2F("hProcess2", "Processes that produced hits", 210, 0, 210, 210, 0, 210);
  tree->Project("hProcess2", "secondProcess:firstProcess", cut);
//   hProcess2->GetXaxis()->SetTitle("Process that produced simHit");
  hProcess2->SetLineColor(kBlue);
  hProcess2->GetXaxis()->SetTitle("");
  hProcess2->GetXaxis()->SetBinLabel(1, "Undefined");
  hProcess2->GetXaxis()->SetBinLabel(2, "Unknown");
  hProcess2->GetXaxis()->SetBinLabel(3, "Primary");
  hProcess2->GetXaxis()->SetBinLabel(4, "Hadronic");
  hProcess2->GetXaxis()->SetBinLabel(5, "Decay");
  hProcess2->GetXaxis()->SetBinLabel(6, "Compton");
  hProcess2->GetXaxis()->SetBinLabel(7, "Annihilation");
  hProcess2->GetXaxis()->SetBinLabel(8, "EIoni");
  hProcess2->GetXaxis()->SetBinLabel(9, "HIoni");
  hProcess2->GetXaxis()->SetBinLabel(10, "MuIoni");
  hProcess2->GetXaxis()->SetBinLabel(11, "Photon");
  hProcess2->GetXaxis()->SetBinLabel(12, "MuPairProd");
  hProcess2->GetXaxis()->SetBinLabel(13, "Conversions");
  hProcess2->GetXaxis()->SetBinLabel(14, "EBrem");
  hProcess2->GetXaxis()->SetBinLabel(15, "SynchrotronRadiation");
  hProcess2->GetXaxis()->SetBinLabel(16, "MuBrem");
  hProcess2->GetXaxis()->SetBinLabel(17, "MuNucl");
  hProcess2->GetYaxis()->SetTitle("");
  hProcess2->GetYaxis()->SetBinLabel(1, "Undefined");
  hProcess2->GetYaxis()->SetBinLabel(2, "Unknown");
  hProcess2->GetYaxis()->SetBinLabel(3, "Primary");
  hProcess2->GetYaxis()->SetBinLabel(4, "Hadronic");
  hProcess2->GetYaxis()->SetBinLabel(5, "Decay");
  hProcess2->GetYaxis()->SetBinLabel(6, "Compton");
  hProcess2->GetYaxis()->SetBinLabel(7, "Annihilation");
  hProcess2->GetYaxis()->SetBinLabel(8, "EIoni");
  hProcess2->GetYaxis()->SetBinLabel(9, "HIoni");
  hProcess2->GetYaxis()->SetBinLabel(10, "MuIoni");
  hProcess2->GetYaxis()->SetBinLabel(11, "Photon");
  hProcess2->GetYaxis()->SetBinLabel(12, "MuPairProd");
  hProcess2->GetYaxis()->SetBinLabel(13, "Conversions");
  hProcess2->GetYaxis()->SetBinLabel(14, "EBrem");
  hProcess2->GetYaxis()->SetBinLabel(15, "SynchrotronRadiation");
  hProcess2->GetYaxis()->SetBinLabel(16, "MuBrem");
  hProcess2->GetYaxis()->SetBinLabel(17, "MuNucl");
  hProcess2->Draw("box");

//   canvProc2->SetBottomMargin(.2);
//   gStyle->SetTitleOffset(1.1, "x");

  canvProc2->Update();  // without this the "stats" pointers are null

  canvProc2->SaveAs("clusNtp_process2.pdf");

}  
*/

// ---------------------------------------------------------------

void plot_PID(TTree* tree) {

  TCanvas *canvPID = new TCanvas("canvPID", "canvPID", 700, 500);

  TH1F* hPID = new TH1F("hPID", "Particle ID", 600, -3000, 3000);
  tree->Project("hPID", "firstPID");
  hPID->GetXaxis()->SetTitle("Particle ID");
  hPID->Draw();

  canvPID->Update();  // without this the "stats" pointers are null

  canvPID->SaveAs("clusNtp_PID.pdf");

}  // ---------------------------------------------------------------

void plot_PIDzoom(TTree* tree) {

  TCanvas *canvPIDzoom = new TCanvas("canvPIDzoom", "canvPIDzoom", 700, 500);

  TH1F* hPIDzoom = new TH1F("hPIDzoom", "Particle ID", 800, -400, 400);
  tree->Project("hPIDzoom", "firstPID");
  hPIDzoom->GetXaxis()->SetTitle("Particle ID");
  hPIDzoom->Draw();

  canvPIDzoom->Update();  // without this the "stats" pointers are null

  canvPIDzoom->SaveAs("clusNtp_PIDzoom.pdf");

}  // ---------------------------------------------------------------

void plot_Pmag(TTree* tree, const TCut& cut) {

  TCanvas *canvPmag = new TCanvas("canvPmag", "canvPmag", 700, 500);

  TH1F* hPmag = new TH1F("hPmag", "Particle momentum", 200, 0, 1000);
  tree->Project("hPmag", "firstPmag");
  hPmag->GetXaxis()->SetTitle("Particle momentum");
  hPmag->Draw();

  canvPmag->SetLogy(1);
  canvPmag->Update();  // without this the "stats" pointers are null
  double top = 0.0;
  drawStats(hPmag, top);

  canvPmag->SaveAs("clusNtp_Pmag.pdf");

}  // ---------------------------------------------------------------

void plot_charge(TTree* tree, const TCut& cut) {

  TCanvas *canv3 = new TCanvas("canv3", "canv3", 1100, 800);
  canv3->Divide(2,2);

//  enum SubDetector { UNKNOWN=0, TIB=3, TID=4, TOB=5, TEC=6 };
  TH1F* hChgTIB = new TH1F("hChgTIB", "Charge read out", 200, 0, 1000);
  tree->Project("hChgTIB", "charge", cut&&"subDet==3");
  hChgTIB->GetXaxis()->SetTitle("Charge read out");
  hChgTIB->SetLineColor(kBlue);
  canv3->cd(1);
  hChgTIB->Draw();

  TH1F* hChgTID = new TH1F("hChgTID", "Charge read out", 200, 0, 1000);
  tree->Project("hChgTID", "charge", cut&&"subDet==4");
  hChgTID->GetXaxis()->SetTitle("Charge read out");
  hChgTID->SetLineColor(kBlue);
  canv3->cd(2);
  hChgTID->Draw();

  TH1F* hChgTOB = new TH1F("hChgTOB", "Charge read out", 200, 0, 1000);
  tree->Project("hChgTOB", "charge", cut&&"subDet==5");
  hChgTOB->GetXaxis()->SetTitle("Charge read out");
  hChgTOB->SetLineColor(kBlue);
  canv3->cd(3);
  hChgTOB->Draw();

  TH1F* hChgTEC = new TH1F("hChgTEC", "Charge read out", 200, 0, 1000);
  tree->Project("hChgTEC", "charge", cut&&"subDet==6");
  hChgTEC->GetXaxis()->SetTitle("Charge read out");
  hChgTEC->SetLineColor(kBlue);
  canv3->cd(4);
  hChgTEC->Draw();

  canv3->Update();  // without this the "stats" pointers are null

  canv3->SaveAs("clusNtp_charge.pdf");

}  // ---------------------------------------------------------------

void plot_Eloss(TTree* tree, const TCut& cut) {

  TCanvas *canv4 = new TCanvas("canv4", "canv4", 1100, 800);
  canv4->Divide(2,2);

//  enum SubDetector { UNKNOWN=0, TIB=3, TID=4, TOB=5, TEC=6 };
  TH1F* hElossTIB = new TH1F("hElossTIB", "energy loss", 200, 0, 0.002);
  tree->Project("hElossTIB", "Eloss", cut&&"subDet==3");
  hElossTIB->GetXaxis()->SetTitle("energy loss");
  hElossTIB->SetLineColor(kBlue);
  canv4->cd(1);
  hElossTIB->Draw();

  TH1F* hElossTID = new TH1F("hElossTID", "energy loss", 200, 0, 0.002);
  tree->Project("hElossTID", "Eloss", cut&&"subDet==4");
  hElossTID->GetXaxis()->SetTitle("energy loss");
  hElossTID->SetLineColor(kBlue);
  canv4->cd(2);
  hElossTID->Draw();

  TH1F* hElossTOB = new TH1F("hElossTOB", "energy loss", 200, 0, 0.002);
  tree->Project("hElossTOB", "Eloss", cut&&"subDet==5");
  hElossTOB->GetXaxis()->SetTitle("energy loss");
  hElossTOB->SetLineColor(kBlue);
  canv4->cd(3);
  hElossTOB->Draw();

  TH1F* hElossTEC = new TH1F("hElossTEC", "energy loss", 200, 0, 0.002);
  tree->Project("hElossTEC", "Eloss", cut&&"subDet==6");
  hElossTEC->GetXaxis()->SetTitle("energy loss");
  hElossTEC->SetLineColor(kBlue);
  canv4->cd(4);
  hElossTEC->Draw();

  canv4->Update();  // without this the "stats" pointers are null

  canv4->SaveAs("clusNtp_Eloss.pdf");

}  // ---------------------------------------------------------------

void plot_chargeEloss(TTree* tree, const TCut& cut) {

  TCanvas *canvChgEloss = new TCanvas("canvChgEloss", "canvChgEloss", 1100, 800);
  canvChgEloss->Divide(2,2);

//  enum SubDetector { UNKNOWN=0, TIB=3, TID=4, TOB=5, TEC=6 };
  TH2F* hChgElossTIB = new TH2F("hChgElossTIB", "Charge read out vs Eloss", 200, 0, 0.002, 200, 0, 1000);
  tree->Project("hChgElossTIB", "charge:Eloss", cut&&"subDet==3");
  hChgElossTIB->GetXaxis()->SetTitle("Energy loss");
  hChgElossTIB->GetYaxis()->SetTitle("Charge read out");
  canvChgEloss->cd(1);
  hChgElossTIB->Draw("colz");

  TH2F* hChgElossTID = new TH2F("hChgElossTID", "Charge read out vs Eloss", 200, 0, 0.002, 200, 0, 1000);
  tree->Project("hChgElossTID", "charge:Eloss", cut&&"subDet==4");
  hChgElossTID->GetXaxis()->SetTitle("Energy loss");
  hChgElossTID->GetYaxis()->SetTitle("Charge read out");
  canvChgEloss->cd(2);
  hChgElossTID->Draw("colz");

  TH2F* hChgElossTOB = new TH2F("hChgElossTOB", "Charge read out vs Eloss", 200, 0, 0.002, 200, 0, 1000);
  tree->Project("hChgElossTOB", "charge:Eloss", cut&&"subDet==5");
  hChgElossTOB->GetXaxis()->SetTitle("Energy loss");
  hChgElossTOB->GetYaxis()->SetTitle("Charge read out");
  canvChgEloss->cd(3);
  hChgElossTOB->Draw("colz");

  TH2F* hChgElossTEC = new TH2F("hChgElossTEC", "Charge read out vs Eloss", 200, 0, 0.002, 200, 0, 1000);
  tree->Project("hChgElossTEC", "charge:Eloss", cut&&"subDet==6");
  hChgElossTEC->GetXaxis()->SetTitle("Energy loss");
  hChgElossTEC->GetYaxis()->SetTitle("Charge read out");
  canvChgEloss->cd(4);
  hChgElossTEC->Draw("colz");

  canvChgEloss->Update();  // without this the "stats" pointers are null

  canvChgEloss->SaveAs("clusNtp_chargeEloss.pdf");

}  // ---------------------------------------------------------------

void plot_width(TTree* tree, const TCut& cut) {

  TCanvas *canv5 = new TCanvas("canv5", "canv5", 700, 500);

  TH1F* hwidth = new TH1F("hwidth", "Cluster width", 20, 0, 20);
  tree->Project("hwidth", "width");
  hwidth->GetXaxis()->SetTitle("Cluster width (strips)");
  hwidth->SetLineColor(kRed);
  hwidth->Draw();

  TH1F* hwidOne = new TH1F("hwidOne", "Cluster width", 20, 0, 20);
  tree->Project("hwidOne", "width", cut);
  hwidOne->GetXaxis()->SetTitle("Cluster width (strips)");
  hwidOne->SetLineColor(kBlue);
  hwidOne->Draw("sames");

  canv5->Update();  // without this the "stats" pointers are null
  double top = 0.0;
  drawStats(hwidth, top);
  drawStats(hwidOne, top);

  TLegend* Rtlegend = new TLegend(.34, .65, .76, .80, "");
  Rtlegend->AddEntry(hwidth, "all clusters", "L");
  Rtlegend->AddEntry(hwidOne, "1 crossing track", "L");
  canv5->cd(1);
  Rtlegend->Draw();
  canv5->Update();

  canv5->SaveAs("clusNtp_width.pdf");

}  // ---------------------------------------------------------------

void plot_path(TTree* tree, const TCut& cut) {

  TCanvas *canvPath = new TCanvas("canvPath", "canvPath", 1100, 800);
  canvPath->Divide(2,2);
  gStyle->SetPalette(1);

  // TIB
  TH2F* hPathTIBMI = new TH2F("hPathTIBMI", "Path length, MC truth vs straight line", 280, 280, 560, 280, 280, 560);
  tree->Project("hPathTIBMI", "10000*firstPathLength:10000*pathLstraight", cut&&"subDet==3");
  hPathTIBMI->GetXaxis()->SetTitle("straight path (microns)");
  hPathTIBMI->GetYaxis()->SetTitle("MC path (microns)");
  hPathTIBMI->SetLineColor(8);
  hPathTIBMI->SetMaximum(80);
  canvPath->cd(1);
  hPathTIBMI->Draw("colz");

  // TID
  TH2F* hPathTIDMI = new TH2F("hPathTIDMI", "Path length, MC truth vs straight line", 240, 280, 400, 240, 280, 400);
  tree->Project("hPathTIDMI", "10000*firstPathLength:10000*pathLstraight", cut&&"subDet==4");
  hPathTIDMI->GetXaxis()->SetTitle("straight path (microns)");
  hPathTIDMI->GetYaxis()->SetTitle("MC path (microns)");
  hPathTIDMI->SetLineColor(8);
  hPathTIDMI->SetMaximum(20);
  canvPath->cd(2);
  hPathTIDMI->Draw("colz");

  // TOB
  TH2F* hPathTOBMI = new TH2F("hPathTOBMI", "Path length, MC truth vs straight line", 340, 460, 800, 170, 460, 800);
  tree->Project("hPathTOBMI", "10000*firstPathLength:10000*pathLstraight", cut&&"subDet==5");
  hPathTOBMI->GetXaxis()->SetTitle("straight path (microns)");
  hPathTOBMI->GetYaxis()->SetTitle("MC path (microns)");
  hPathTOBMI->SetLineColor(8);
  hPathTOBMI->SetMaximum(80);
  canvPath->cd(3);
  hPathTOBMI->Draw("colz");

  // TEC
  TH2F* hPathTECMI = new TH2F("hPathTECMI", "Path length, MC truth vs straight line", 320, 280, 600, 320, 280, 600);
  tree->Project("hPathTECMI", "10000*firstPathLength:10000*pathLstraight", cut&&"subDet==6");
  hPathTECMI->GetXaxis()->SetTitle("straight path (microns)");
  hPathTECMI->GetYaxis()->SetTitle("MC path (microns)");
  hPathTECMI->SetLineColor(8);
  hPathTECMI->SetMaximum(20);
  canvPath->cd(4);
  hPathTECMI->Draw("colz");

  canvPath->SaveAs("clusNtp_path.pdf");

}  // ---------------------------------------------------------------

void plot_path1(TTree* tree, const TCut& cut) {

  TCanvas *canvPath1 = new TCanvas("canvPath1", "canvPath1", 1100, 800);
  canvPath1->Divide(2,2);
  gStyle->SetPalette(1);

  // TIB
  TH1F* hPath1TIBMI = new TH1F("hPath1TIBMI", "Path length, straight line", 280, 280, 560);
  tree->Project("hPath1TIBMI", "10000*pathLstraight", cut&&"subDet==3");
  hPath1TIBMI->GetXaxis()->SetTitle("path length (microns)");
  hPath1TIBMI->SetLineColor(kBlue);
  TH1F* hPath2TIBMI = new TH1F("hPath2TIBMI", "Path length, MC truth", 280, 280, 560);
  tree->Project("hPath2TIBMI", "10000*firstPathLength", cut&&"subDet==3");
  hPath2TIBMI->SetLineColor(kRed);
  canvPath1->cd(1);
  hPath1TIBMI->Draw();
  hPath2TIBMI->Draw("sames");

  canvPath1->Update();  // without this the "stats" pointers are null
  double top = 0.0;
  drawStats(hPath1TIBMI, top);
  drawStats(hPath2TIBMI, top);
  canvPath1->Update();

  // TID
  TH1F* hPath1TIDMI = new TH1F("hPath1TIDMI", "Path length, straight line", 240, 280, 400);
  tree->Project("hPath1TIDMI", "10000*pathLstraight", cut&&"subDet==4");
  hPath1TIDMI->GetXaxis()->SetTitle("path length (microns)");
  hPath1TIDMI->SetLineColor(kBlue);
  TH1F* hPath2TIDMI = new TH1F("hPath2TIDMI", "Path length, MC truth", 240, 280, 400);
  tree->Project("hPath2TIDMI", "10000*firstPathLength", cut&&"subDet==4");
  hPath2TIDMI->SetLineColor(kRed);
  canvPath1->cd(2);
  hPath1TIDMI->Draw();
  hPath2TIDMI->Draw("sames");

  canvPath1->Update();  // without this the "stats" pointers are null
  top = 0.0;
  drawStats(hPath1TIDMI, top);
  drawStats(hPath2TIDMI, top);
  canvPath1->Update();

  // TOB
  TH1F* hPath1TOBMI = new TH1F("hPath1TOBMI", "Path length, straight line", 340, 460, 800);
  tree->Project("hPath1TOBMI", "10000*pathLstraight", cut&&"subDet==5");
  hPath1TOBMI->GetXaxis()->SetTitle("path length (microns)");
  hPath1TOBMI->SetLineColor(kBlue);
  TH1F* hPath2TOBMI = new TH1F("hPath2TOBMI", "Path length, MC truth", 340, 460, 800);
  tree->Project("hPath2TOBMI", "10000*firstPathLength", cut&&"subDet==5");
  hPath2TOBMI->SetLineColor(kRed);
  canvPath1->cd(3);
  hPath1TOBMI->Draw();
  hPath2TOBMI->Draw("sames");

  canvPath1->Update();  // without this the "stats" pointers are null
  top = 0.0;
  drawStats(hPath1TOBMI, top);
  drawStats(hPath2TOBMI, top);
  canvPath1->Update();

  // TEC
  TH1F* hPath1TECMI = new TH1F("hPath1TECMI", "Path length, straight line", 320, 280, 600);
  tree->Project("hPath1TECMI", "10000*pathLstraight", cut&&"subDet==6");
  hPath1TECMI->GetXaxis()->SetTitle("path length (microns)");
  hPath1TECMI->SetLineColor(kBlue);
  TH1F* hPath2TECMI = new TH1F("hPath2TECMI", "Path length, MC truth", 320, 280, 600);
  tree->Project("hPath2TECMI", "10000*firstPathLength", cut&&"subDet==6");
  hPath2TECMI->SetLineColor(kRed);
  canvPath1->cd(4);
  hPath1TECMI->Draw();
  hPath2TECMI->Draw("sames");

  canvPath1->Update();  // without this the "stats" pointers are null
  top = 0.0;
  drawStats(hPath1TECMI, top);
  drawStats(hPath2TECMI, top);
  canvPath1->Update();

  TLegend* Rtlegend = new TLegend(.32, .65, .72, .80, "");
  Rtlegend->AddEntry(hPath1TOBMI, "Straight-track", "L");
  Rtlegend->AddEntry(hPath2TOBMI, "MC", "L");
  canvPath1->cd(1);
  Rtlegend->Draw();
  canvPath1->Update();

  canvPath1->SaveAs("clusNtp_path1.pdf");

}  // ---------------------------------------------------------------

void plot_NormCharge(TTree* tree, const TString& MIPdEdx, const TCut& cut1, const TCut& cut2, const TCut& cut3) {

  TCanvas *canv6 = new TCanvas("canv6", "canv6", 1100, 800);
  canv6->Divide(2,2);

  TH1F* hrelChgTIB = new TH1F("hrelChgTIB", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTIB", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet == 3");
  hrelChgTIB->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTIB->SetMinimum(10);
  canv6->cd(1);
  hrelChgTIB->Draw();

  TH1F* hrelChgTIBMI = new TH1F("hrelChgTIBMI", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTIBMI", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut1&&"subDet==3");
  hrelChgTIBMI->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTIBMI->SetLineColor(8);
//   hrelChgTIBMI->SetLineStyle(kDashed);
  hrelChgTIBMI->Draw("sames");

  TH1F* hrelChgTIB2h = new TH1F("hrelChgTIB2h", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTIB2h", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut2&&"subDet == 3");
  hrelChgTIB2h->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTIB2h->SetLineColor(kBlue);
  hrelChgTIB2h->Draw("sames");

  TH1F* hrelChgTIB2hcl = new TH1F("hrelChgTIB2hcl", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTIB2hcl", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut3&&"subDet == 3");
  hrelChgTIB2hcl->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTIB2hcl->SetMinimum(10);
  hrelChgTIB2hcl->SetLineColor(kRed);
  hrelChgTIB2hcl->Draw("sames");

  gPad->SetLogy(1);

  canv6->Update();  // without this the "stats" pointers are null
  double top = 0.0;
  drawStats(hrelChgTIB, top);
  drawStats(hrelChgTIBMI, top);
  drawStats(hrelChgTIB2h, top);
  drawStats(hrelChgTIB2hcl, top);
  canv6->Update();

  TH1F* hrelChgTID = new TH1F("hrelChgTID", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTID", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet == 4");
  hrelChgTID->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTID->SetMinimum(1);
  canv6->cd(2);
  hrelChgTID->Draw();

  TH1F* hrelChgTIDMI = new TH1F("hrelChgTIDMI", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTIDMI", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut1&&"subDet==4");
  hrelChgTIDMI->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTIDMI->SetLineColor(8);
  hrelChgTIDMI->Draw("sames");

  TH1F* hrelChgTID2h = new TH1F("hrelChgTID2h", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTID2h", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut2&&"subDet==4");
  hrelChgTID2h->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTID2h->SetLineColor(kBlue);
  hrelChgTID2h->Draw("sames");

  TH1F* hrelChgTID2hcl = new TH1F("hrelChgTID2hcl", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTID2hcl", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut3&&"subDet==4");
  hrelChgTID2hcl->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTID2hcl->SetLineColor(kRed);
  hrelChgTID2hcl->Draw("sames");

  gPad->SetLogy(1);

  canv6->Update();  // without this the "stats" pointers are null
  top = 0.0;
  drawStats(hrelChgTID, top);
  drawStats(hrelChgTIDMI, top);
  drawStats(hrelChgTID2h, top);
  drawStats(hrelChgTID2hcl, top);
  canv6->Update();

  TH1F* hrelChgTOB = new TH1F("hrelChgTOB", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTOB", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet == 5");
  hrelChgTOB->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTOB->SetMinimum(10);
  canv6->cd(3);
  hrelChgTOB->Draw();

  TH1F* hrelChgTOBMI = new TH1F("hrelChgTOBMI", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTOBMI", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut1&&"subDet==5");
  hrelChgTOBMI->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTOBMI->SetLineColor(8);
  hrelChgTOBMI->Draw("sames");

  TH1F* hrelChgTOB2h = new TH1F("hrelChgTOB2h", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTOB2h", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut2&&"subDet==5");
  hrelChgTOB2h->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTOB2h->SetLineColor(kBlue);
  hrelChgTOB2h->Draw("sames");

  TH1F* hrelChgTOB2hcl = new TH1F("hrelChgTOB2hcl", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTOB2hcl", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut3&&"subDet==5");
  hrelChgTOB2hcl->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTOB2hcl->SetLineColor(kRed);
  hrelChgTOB2hcl->Draw("sames");

  gPad->SetLogy(1);

  canv6->Update();  // without this the "stats" pointers are null
  top = 0.0;
  drawStats(hrelChgTOB, top);
  drawStats(hrelChgTOBMI, top);
  drawStats(hrelChgTOB2h, top);
  drawStats(hrelChgTOB2hcl, top);
  canv6->Update();

  TH1F* hrelChgTEC = new TH1F("hrelChgTEC", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTEC", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet == 6");
  hrelChgTEC->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTEC->SetMinimum(1);
  canv6->cd(4);
  hrelChgTEC->Draw();

  TH1F* hrelChgTECMI = new TH1F("hrelChgTECMI", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTECMI", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut1&&"subDet==6");
  hrelChgTECMI->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTECMI->SetLineColor(8);
  hrelChgTECMI->Draw("sames");

  TH1F* hrelChgTEC2h = new TH1F("hrelChgTEC2h", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTEC2h", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut2&&"subDet==6");
  hrelChgTEC2h->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTEC2h->SetLineColor(kBlue);
  hrelChgTEC2h->Draw("sames");

  TH1F* hrelChgTEC2hcl = new TH1F("hrelChgTEC2hcl", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTEC2hcl", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut3&&"subDet==6");
  hrelChgTEC2hcl->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTEC2hcl->SetLineColor(kRed);
  hrelChgTEC2hcl->Draw("sames");

  gPad->SetLogy(1);

  canv6->Update();  // without this the "stats" pointers are null
  top = 0.0;
  drawStats(hrelChgTEC, top);
  drawStats(hrelChgTECMI, top);
  drawStats(hrelChgTEC2h, top);
  drawStats(hrelChgTEC2hcl, top);
  canv6->Update();

  TLegend* Rtlegend = new TLegend(.36, .67, .78, .87, "");
  Rtlegend->AddEntry(hrelChgTIB, "all clusters", "L");
  Rtlegend->AddEntry(hrelChgTIBMI, "1 crossing primary", "L");
  Rtlegend->AddEntry(hrelChgTIB2h, "2 crossing tracks", "L");
  Rtlegend->AddEntry(hrelChgTIB2hcl, "2 crossing primaries", "L");
  canv6->cd(1);
  Rtlegend->Draw();
  canv6->Update();

  canv6->SaveAs("clusNtp_relCharge.pdf");

}  // ---------------------------------------------------------------

void plot_NormEloss(TTree* tree, const TString& MIPdEdx, const TCut& cut1, const TCut& cut2, const TCut& cut3) {

  TCanvas *canvNormEloss = new TCanvas("canvNormEloss", "canvNormEloss", 1100, 800);
  canvNormEloss->Divide(2,2);

  TH1F* hrelElossTIB = new TH1F("hrelElossTIB", "Relative Eloss", 100, 0, 10);
  tree->Project("hrelElossTIB", MIPdEdx+"*Eloss/max(1.e-6,firstPathLength)", "subDet == 3");
  hrelElossTIB->GetXaxis()->SetTitle("Eloss/(path length), MIP");
  hrelElossTIB->SetMinimum(10);
  canvNormEloss->cd(1);
  hrelElossTIB->Draw();

  TH1F* hrelElossTIBMI = new TH1F("hrelElossTIBMI", "Relative Eloss", 100, 0, 10);
  tree->Project("hrelElossTIBMI", MIPdEdx+"*Eloss/max(1.e-6,firstPathLength)", cut1&&"subDet==3");
  hrelElossTIBMI->GetXaxis()->SetTitle("Eloss/(path length), MIP");
  hrelElossTIBMI->SetLineColor(8);
  hrelElossTIBMI->Draw("sames");

  TH1F* hrelElossTIB2h = new TH1F("hrelElossTIB2h", "Relative Eloss", 100, 0, 10);
  tree->Project("hrelElossTIB2h", MIPdEdx+"*Eloss/max(1.e-6,firstPathLength)", cut2&&"subDet == 3");
  hrelElossTIB2h->GetXaxis()->SetTitle("Eloss/(path length), MIP");
  hrelElossTIB2h->SetLineColor(kBlue);
  hrelElossTIB2h->Draw("sames");

  TH1F* hrelElossTIB2hcl = new TH1F("hrelElossTIB2hcl", "Relative Eloss", 100, 0, 10);
  tree->Project("hrelElossTIB2hcl", MIPdEdx+"*Eloss/max(1.e-6,firstPathLength)", cut3&&"subDet == 3");
  hrelElossTIB2hcl->GetXaxis()->SetTitle("Eloss/(path length), MIP");
  hrelElossTIB2hcl->SetMinimum(10);
  hrelElossTIB2hcl->SetLineColor(kRed);
  hrelElossTIB2hcl->Draw("sames");

  gPad->SetLogy(1);

  canvNormEloss->Update();  // without this the "stats" pointers are null
  double top = 0.0;
  drawStats(hrelElossTIB, top);
  drawStats(hrelElossTIBMI, top);
  drawStats(hrelElossTIB2h, top);
  drawStats(hrelElossTIB2hcl, top);
  canvNormEloss->Update();

  TH1F* hrelElossTID = new TH1F("hrelElossTID", "Relative Eloss", 100, 0, 10);
  tree->Project("hrelElossTID", MIPdEdx+"*Eloss/max(1.e-6,firstPathLength)", "subDet == 4");
  hrelElossTID->GetXaxis()->SetTitle("Eloss/(path length), MIP");
  hrelElossTID->SetMinimum(1);
  canvNormEloss->cd(2);
  hrelElossTID->Draw();

  TH1F* hrelElossTIDMI = new TH1F("hrelElossTIDMI", "Relative Eloss", 100, 0, 10);
  tree->Project("hrelElossTIDMI", MIPdEdx+"*Eloss/max(1.e-6,firstPathLength)", cut1&&"subDet==4");
  hrelElossTIDMI->GetXaxis()->SetTitle("Eloss/(path length), MIP");
  hrelElossTIDMI->SetLineColor(8);
  hrelElossTIDMI->Draw("sames");

  TH1F* hrelElossTID2h = new TH1F("hrelElossTID2h", "Relative Eloss", 100, 0, 10);
  tree->Project("hrelElossTID2h", MIPdEdx+"*Eloss/max(1.e-6,firstPathLength)", cut2&&"subDet==4");
  hrelElossTID2h->GetXaxis()->SetTitle("Eloss/(path length), MIP");
  hrelElossTID2h->SetLineColor(kBlue);
  hrelElossTID2h->Draw("sames");

  TH1F* hrelElossTID2hcl = new TH1F("hrelElossTID2hcl", "Relative Eloss", 100, 0, 10);
  tree->Project("hrelElossTID2hcl", MIPdEdx+"*Eloss/max(1.e-6,firstPathLength)", cut3&&"subDet==4");
  hrelElossTID2hcl->GetXaxis()->SetTitle("Eloss/(path length), MIP");
  hrelElossTID2hcl->SetLineColor(kRed);
  hrelElossTID2hcl->Draw("sames");

  gPad->SetLogy(1);

  canvNormEloss->Update();  // without this the "stats" pointers are null
  top = 0.0;
  drawStats(hrelElossTID, top);
  drawStats(hrelElossTIDMI, top);
  drawStats(hrelElossTID2h, top);
  drawStats(hrelElossTID2hcl, top);
  canvNormEloss->Update();

  TH1F* hrelElossTOB = new TH1F("hrelElossTOB", "Relative Eloss", 100, 0, 10);
  tree->Project("hrelElossTOB", MIPdEdx+"*Eloss/max(1.e-6,firstPathLength)", "subDet == 5");
  hrelElossTOB->GetXaxis()->SetTitle("Eloss/(path length), MIP");
  hrelElossTOB->SetMinimum(10);
  canvNormEloss->cd(3);
  hrelElossTOB->Draw();

  TH1F* hrelElossTOBMI = new TH1F("hrelElossTOBMI", "Relative Eloss", 100, 0, 10);
  tree->Project("hrelElossTOBMI", MIPdEdx+"*Eloss/max(1.e-6,firstPathLength)", cut1&&"subDet==5");
  hrelElossTOBMI->GetXaxis()->SetTitle("Eloss/(path length), MIP");
  hrelElossTOBMI->SetLineColor(8);
  hrelElossTOBMI->Draw("sames");

  TH1F* hrelElossTOB2h = new TH1F("hrelElossTOB2h", "Relative Eloss", 100, 0, 10);
  tree->Project("hrelElossTOB2h", MIPdEdx+"*Eloss/max(1.e-6,firstPathLength)", cut2&&"subDet==5");
  hrelElossTOB2h->GetXaxis()->SetTitle("Eloss/(path length), MIP");
  hrelElossTOB2h->SetLineColor(kBlue);
  hrelElossTOB2h->Draw("sames");

  TH1F* hrelElossTOB2hcl = new TH1F("hrelElossTOB2hcl", "Relative Eloss", 100, 0, 10);
  tree->Project("hrelElossTOB2hcl", MIPdEdx+"*Eloss/max(1.e-6,firstPathLength)", cut3&&"subDet==5");
  hrelElossTOB2hcl->GetXaxis()->SetTitle("Eloss/(path length), MIP");
  hrelElossTOB2hcl->SetLineColor(kRed);
  hrelElossTOB2hcl->Draw("sames");

  gPad->SetLogy(1);

  canvNormEloss->Update();  // without this the "stats" pointers are null
  top = 0.0;
  drawStats(hrelElossTOB, top);
  drawStats(hrelElossTOBMI, top);
  drawStats(hrelElossTOB2h, top);
  drawStats(hrelElossTOB2hcl, top);
  canvNormEloss->Update();

  TH1F* hrelElossTEC = new TH1F("hrelElossTEC", "Relative Eloss", 100, 0, 10);
  tree->Project("hrelElossTEC", MIPdEdx+"*Eloss/max(1.e-6,firstPathLength)", "subDet == 6");
  hrelElossTEC->GetXaxis()->SetTitle("Eloss/(path length), MIP");
  hrelElossTEC->SetMinimum(1);
  canvNormEloss->cd(4);
  hrelElossTEC->Draw();

  TH1F* hrelElossTECMI = new TH1F("hrelElossTECMI", "Relative Eloss", 100, 0, 10);
  tree->Project("hrelElossTECMI", MIPdEdx+"*Eloss/max(1.e-6,firstPathLength)", cut1&&"subDet==6");
  hrelElossTECMI->GetXaxis()->SetTitle("Eloss/(path length), MIP");
  hrelElossTECMI->SetLineColor(8);
  hrelElossTECMI->Draw("sames");

  TH1F* hrelElossTEC2h = new TH1F("hrelElossTEC2h", "Relative Eloss", 100, 0, 10);
  tree->Project("hrelElossTEC2h", MIPdEdx+"*Eloss/max(1.e-6,firstPathLength)", cut2&&"subDet==6");
  hrelElossTEC2h->GetXaxis()->SetTitle("Eloss/(path length), MIP");
  hrelElossTEC2h->SetLineColor(kBlue);
  hrelElossTEC2h->Draw("sames");

  TH1F* hrelElossTEC2hcl = new TH1F("hrelElossTEC2hcl", "Relative Eloss", 100, 0, 10);
  tree->Project("hrelElossTEC2hcl", MIPdEdx+"*Eloss/max(1.e-6,firstPathLength)", cut3&&"subDet==6");
  hrelElossTEC2hcl->GetXaxis()->SetTitle("Eloss/(path length), MIP");
  hrelElossTEC2hcl->SetLineColor(kRed);
  hrelElossTEC2hcl->Draw("sames");

  gPad->SetLogy(1);

  canvNormEloss->Update();  // without this the "stats" pointers are null
  top = 0.0;
  drawStats(hrelElossTEC, top);
  drawStats(hrelElossTECMI, top);
  drawStats(hrelElossTEC2h, top);
  drawStats(hrelElossTEC2hcl, top);
  canvNormEloss->Update();

  TLegend* Rtlegend = new TLegend(.36, .67, .78, .87, "");
  Rtlegend->AddEntry(hrelElossTIB, "all clusters", "L");
  Rtlegend->AddEntry(hrelElossTIBMI, "1 crossing primary", "L");
  Rtlegend->AddEntry(hrelElossTIB2h, "2 crossing tracks", "L");
  Rtlegend->AddEntry(hrelElossTIB2hcl, "2 crossing primaries", "L");
  canvNormEloss->cd(1);
  Rtlegend->Draw();
  canvNormEloss->Update();

  canvNormEloss->SaveAs("clusNtp_relEloss.pdf");

}  // ---------------------------------------------------------------

void plot_chgTotPath(TTree* tree, const TString& MIPdEdx, const TCut& cut0, const TCut& cut1, const TCut& cut2, const TCut& cut3,
		     const char* title0, const char* title1, const char* title2, const char* title3 ) {

  TCanvas *canvChgTotPath = new TCanvas("canvChgTotPath", "canvChgTotPath", 1100, 800);
  canvChgTotPath->Divide(2,2);

  TH1F* hchgTPaTIB = new TH1F("hchgTPaTIB", "Relative charge", 100, 0, 10);
  tree->Project("hchgTPaTIB", MIPdEdx+"*charge/max(1.e-6,allHtPathLength)", "subDet == 3");
  hchgTPaTIB->GetXaxis()->SetTitle("Charge/(total path length), MIP");
  hchgTPaTIB->SetMinimum(10);
  canvChgTotPath->cd(1);
  hchgTPaTIB->Draw();

  TH1F* hchgTPaTIBMI = new TH1F("hchgTPaTIBMI", "Relative charge", 100, 0, 10);
  tree->Project("hchgTPaTIBMI", MIPdEdx+"*charge/max(1.e-6,allHtPathLength)", cut1&&"subDet==3");
  hchgTPaTIBMI->GetXaxis()->SetTitle("Charge/(total path length), MIP");
  hchgTPaTIBMI->SetLineColor(8);
  hchgTPaTIBMI->Draw("sames");

  TH1F* hchgTPaTIB2h = new TH1F("hchgTPaTIB2h", "Relative charge", 100, 0, 10);
  tree->Project("hchgTPaTIB2h", MIPdEdx+"*charge/max(1.e-6,allHtPathLength)", cut2&&"subDet == 3");
  hchgTPaTIB2h->GetXaxis()->SetTitle("Charge/(total path length), MIP");
  hchgTPaTIB2h->SetLineColor(kBlue);
  hchgTPaTIB2h->Draw("sames");

  TH1F* hchgTPaTIB2hcl = new TH1F("hchgTPaTIB2hcl", "Relative charge", 100, 0, 10);
  tree->Project("hchgTPaTIB2hcl", MIPdEdx+"*charge/max(1.e-6,allHtPathLength)", cut3&&"subDet == 3");
  hchgTPaTIB2hcl->GetXaxis()->SetTitle("Charge/(total path length), MIP");
  hchgTPaTIB2hcl->SetMinimum(10);
  hchgTPaTIB2hcl->SetLineColor(kRed);
  hchgTPaTIB2hcl->Draw("sames");

  gPad->SetLogy(1);

  canvChgTotPath->Update();  // without this the "stats" pointers are null
  double top = 0.0;
  drawStats(hchgTPaTIB, top);
  drawStats(hchgTPaTIBMI, top);
  drawStats(hchgTPaTIB2h, top);
  drawStats(hchgTPaTIB2hcl, top);
  canvChgTotPath->Update();

  TH1F* hchgTPaTID = new TH1F("hchgTPaTID", "Relative charge", 100, 0, 10);
  tree->Project("hchgTPaTID", MIPdEdx+"*charge/max(1.e-6,allHtPathLength)", "subDet == 4");
  hchgTPaTID->GetXaxis()->SetTitle("Charge/(total path length), MIP");
  hchgTPaTID->SetMinimum(1);
  canvChgTotPath->cd(2);
  hchgTPaTID->Draw();

  TH1F* hchgTPaTIDMI = new TH1F("hchgTPaTIDMI", "Relative charge", 100, 0, 10);
  tree->Project("hchgTPaTIDMI", MIPdEdx+"*charge/max(1.e-6,allHtPathLength)", cut1&&"subDet==4");
  hchgTPaTIDMI->GetXaxis()->SetTitle("Charge/(total path length), MIP");
  hchgTPaTIDMI->SetLineColor(8);
  hchgTPaTIDMI->Draw("sames");

  TH1F* hchgTPaTID2h = new TH1F("hchgTPaTID2h", "Relative charge", 100, 0, 10);
  tree->Project("hchgTPaTID2h", MIPdEdx+"*charge/max(1.e-6,allHtPathLength)", cut2&&"subDet==4");
  hchgTPaTID2h->GetXaxis()->SetTitle("Charge/(total path length), MIP");
  hchgTPaTID2h->SetLineColor(kBlue);
  hchgTPaTID2h->Draw("sames");

  TH1F* hchgTPaTID2hcl = new TH1F("hchgTPaTID2hcl", "Relative charge", 100, 0, 10);
  tree->Project("hchgTPaTID2hcl", MIPdEdx+"*charge/max(1.e-6,allHtPathLength)", cut3&&"subDet==4");
  hchgTPaTID2hcl->GetXaxis()->SetTitle("Charge/(total path length), MIP");
  hchgTPaTID2hcl->SetLineColor(kRed);
  hchgTPaTID2hcl->Draw("sames");

  gPad->SetLogy(1);

  canvChgTotPath->Update();  // without this the "stats" pointers are null
  top = 0.0;
  drawStats(hchgTPaTID, top);
  drawStats(hchgTPaTIDMI, top);
  drawStats(hchgTPaTID2h, top);
  drawStats(hchgTPaTID2hcl, top);
  canvChgTotPath->Update();

  TH1F* hchgTPaTOB = new TH1F("hchgTPaTOB", "Relative charge", 100, 0, 10);
  tree->Project("hchgTPaTOB", MIPdEdx+"*charge/max(1.e-6,allHtPathLength)", "subDet == 5");
  hchgTPaTOB->GetXaxis()->SetTitle("Charge/(total path length), MIP");
  hchgTPaTOB->SetMinimum(10);
  canvChgTotPath->cd(3);
  hchgTPaTOB->Draw();

  TH1F* hchgTPaTOBMI = new TH1F("hchgTPaTOBMI", "Relative charge", 100, 0, 10);
  tree->Project("hchgTPaTOBMI", MIPdEdx+"*charge/max(1.e-6,allHtPathLength)", cut1&&"subDet==5");
  hchgTPaTOBMI->GetXaxis()->SetTitle("Charge/(total path length), MIP");
  hchgTPaTOBMI->SetLineColor(8);
  hchgTPaTOBMI->Draw("sames");

  TH1F* hchgTPaTOB2h = new TH1F("hchgTPaTOB2h", "Relative charge", 100, 0, 10);
  tree->Project("hchgTPaTOB2h", MIPdEdx+"*charge/max(1.e-6,allHtPathLength)", cut2&&"subDet==5");
  hchgTPaTOB2h->GetXaxis()->SetTitle("Charge/(total path length), MIP");
  hchgTPaTOB2h->SetLineColor(kBlue);
  hchgTPaTOB2h->Draw("sames");

  TH1F* hchgTPaTOB2hcl = new TH1F("hchgTPaTOB2hcl", "Relative charge", 100, 0, 10);
  tree->Project("hchgTPaTOB2hcl", MIPdEdx+"*charge/max(1.e-6,allHtPathLength)", cut3&&"subDet==5");
  hchgTPaTOB2hcl->GetXaxis()->SetTitle("Charge/(total path length), MIP");
  hchgTPaTOB2hcl->SetLineColor(kRed);
  hchgTPaTOB2hcl->Draw("sames");

  gPad->SetLogy(1);

  canvChgTotPath->Update();  // without this the "stats" pointers are null
  top = 0.0;
  drawStats(hchgTPaTOB, top);
  drawStats(hchgTPaTOBMI, top);
  drawStats(hchgTPaTOB2h, top);
  drawStats(hchgTPaTOB2hcl, top);
  canvChgTotPath->Update();

  TH1F* hchgTPaTEC = new TH1F("hchgTPaTEC", "Relative charge", 100, 0, 10);
  tree->Project("hchgTPaTEC", MIPdEdx+"*charge/max(1.e-6,allHtPathLength)", "subDet == 6");
  hchgTPaTEC->GetXaxis()->SetTitle("Charge/(total path length), MIP");
  hchgTPaTEC->SetMinimum(1);
  canvChgTotPath->cd(4);
  hchgTPaTEC->Draw();

  TH1F* hchgTPaTECMI = new TH1F("hchgTPaTECMI", "Relative charge", 100, 0, 10);
  tree->Project("hchgTPaTECMI", MIPdEdx+"*charge/max(1.e-6,allHtPathLength)", cut1&&"subDet==6");
  hchgTPaTECMI->GetXaxis()->SetTitle("Charge/(total path length), MIP");
  hchgTPaTECMI->SetLineColor(8);
  hchgTPaTECMI->Draw("sames");

  TH1F* hchgTPaTEC2h = new TH1F("hchgTPaTEC2h", "Relative charge", 100, 0, 10);
  tree->Project("hchgTPaTEC2h", MIPdEdx+"*charge/max(1.e-6,allHtPathLength)", cut2&&"subDet==6");
  hchgTPaTEC2h->GetXaxis()->SetTitle("Charge/(total path length), MIP");
  hchgTPaTEC2h->SetLineColor(kBlue);
  hchgTPaTEC2h->Draw("sames");

  TH1F* hchgTPaTEC2hcl = new TH1F("hchgTPaTEC2hcl", "Relative charge", 100, 0, 10);
  tree->Project("hchgTPaTEC2hcl", MIPdEdx+"*charge/max(1.e-6,allHtPathLength)", cut3&&"subDet==6");
  hchgTPaTEC2hcl->GetXaxis()->SetTitle("Charge/(total path length), MIP");
  hchgTPaTEC2hcl->SetLineColor(kRed);
  hchgTPaTEC2hcl->Draw("sames");

  gPad->SetLogy(1);

  canvChgTotPath->Update();  // without this the "stats" pointers are null
  top = 0.0;
  drawStats(hchgTPaTEC, top);
  drawStats(hchgTPaTECMI, top);
  drawStats(hchgTPaTEC2h, top);
  drawStats(hchgTPaTEC2hcl, top);
  canvChgTotPath->Update();

  TLegend* Rtlegend = new TLegend(.36, .67, .78, .87, "");
  Rtlegend->AddEntry(hchgTPaTIB, title0, "L");
  Rtlegend->AddEntry(hchgTPaTIBMI, title1, "L");
  Rtlegend->AddEntry(hchgTPaTIB2h, title2, "L");
  Rtlegend->AddEntry(hchgTPaTIB2hcl, title3, "L");
  canvChgTotPath->cd(1);
  Rtlegend->Draw();
  canvChgTotPath->Update();

  canvChgTotPath->SaveAs("clusNtp_chgTotPath.pdf");

}  // ---------------------------------------------------------------

void plot_NormChgSpl(TTree* tree, const TString& MIPdEdx, const TCut& cut) {

  TCanvas *canv7 = new TCanvas("canv7", "canv7", 700, 500);

  TH1F* hrelChg = new TH1F("hrelChg", "Relative charge", 100, 0, 10);
  tree->Project("hrelChg", MIPdEdx+"*charge/max(1.e-6,Eloss)");
  hrelChg->GetXaxis()->SetTitle("Charge/(expected charge)");
  hrelChg->SetLineColor(kBlue);
  hrelChg->Draw();

  TH1F* hrelChgSpl = new TH1F("hrelChgSpl", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgSpl", MIPdEdx+"*charge/max(1.e-6,Eloss)", cut);
  hrelChgSpl->GetXaxis()->SetTitle("Charge/(expected charge)");
  hrelChgSpl->SetLineColor(kRed);
  hrelChgSpl->Draw("sames");

  canv7->Update();  // without this the "stats" pointers are null
  double top = 0.0;
  drawStats(hrelChg, top);
  drawStats(hrelChgSpl, top);
  canv7->Update();

  TLegend* Rtlegend = new TLegend(.35, .65, .75, .80, "");
  Rtlegend->AddEntry(hrelChg, "All clusters", "L");
  Rtlegend->AddEntry(hrelChgSpl, "2-simHit clusters", "L");
  Rtlegend->Draw();
  canv7->Update();

  canv7->SaveAs("clusNtp_relChgSplit.pdf");

}  // ---------------------------------------------------------------


void plot_NormChgNtp(TTree* tree, const TString& MIPdEdx, const TCut& cut) {

  TCanvas *canvNchgNtp = new TCanvas("canvNchgNtp", "canvNchgNtp", 700, 900);
  canvNchgNtp->Divide(1,2);

  TH2F* hNchgNtp = new TH2F("hNchgNtp", "Ntp vs Relative charge", 25, 0, 5, 10, 0, 10);
  tree->Project("hNchgNtp", "Ntp:"+MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut);
  hNchgNtp->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hNchgNtp->GetYaxis()->SetTitle("No. tracks");
  hNchgNtp->SetLineColor(kBlue);
  canvNchgNtp->cd(1);  hNchgNtp->Draw("box");

  TH2F* hNchgNht = new TH2F("hNchgNht", "Nhits vs Relative charge", 25, 0, 5, 10, 0, 10);
  tree->Project("hNchgNht", "NsimHits:"+MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut);
  hNchgNht->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hNchgNht->GetYaxis()->SetTitle("No. hits");
  hNchgNht->SetLineColor(kBlue);
  canvNchgNtp->cd(2);  hNchgNht->Draw("box");

  canvNchgNtp->Update();

  canvNchgNtp->SaveAs("clusNtp_NchgNtp.pdf");

}  // ---------------------------------------------------------------

void plot_NrmChgWid(TTree* tree, const TString& MIPdEdx) {

  TCanvas *canvNchgWid = new TCanvas("canvNchgWid", "canvNchgWid", 1100, 800);
  canvNchgWid->Divide(2,2);

  // TIB

  TH1F* hChgWidTIB3 = new TH1F("hChgWidTIB3", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTIB3", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==3 && width == 3");
  hChgWidTIB3->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hChgWidTIB3->SetLineColor(kBlack);
  canvNchgWid->cd(1);
  hChgWidTIB3->Draw();

  TH1F* hChgWidTIB1 = new TH1F("hChgWidTIB1", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTIB1", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==3 && width == 1");
  hChgWidTIB1->SetLineColor(kRed);
  hChgWidTIB1->Draw("same");

  TH1F* hChgWidTIB2 = new TH1F("hChgWidTIB2", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTIB2", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==3 && width == 2");
  hChgWidTIB2->SetLineColor(kBlue);
  hChgWidTIB2->Draw("same");

  TH1F* hChgWidTIB4 = new TH1F("hChgWidTIB4", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTIB4", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==3 && width == 4");
  hChgWidTIB4->SetLineColor(8);
  hChgWidTIB4->Draw("same");

  TH1F* hChgWidTIB5 = new TH1F("hChgWidTIB5", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTIB5", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==3 && width == 5");
  hChgWidTIB5->SetLineColor(7);
  hChgWidTIB5->Draw("same");

  TH1F* hChgWidTIB6 = new TH1F("hChgWidTIB6", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTIB6", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==3 && width >= 6 && width <= 8");
  hChgWidTIB6->SetLineColor(28);
  hChgWidTIB6->Draw("same");

  TH1F* hChgWidTIB8 = new TH1F("hChgWidTIB8", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTIB8", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==3 && width > 8");
  hChgWidTIB8->SetLineColor(6);
  hChgWidTIB8->Draw("same");

  TLegend* CWlegend = new TLegend(.54, .43, .76, .83, "");
  CWlegend->AddEntry(hChgWidTIB1, "width = 1", "L");
  CWlegend->AddEntry(hChgWidTIB2, "width = 2", "L");
  CWlegend->AddEntry(hChgWidTIB3, "width = 3", "L");
  CWlegend->AddEntry(hChgWidTIB4, "width = 4", "L");
  CWlegend->AddEntry(hChgWidTIB5, "width = 5", "L");
  CWlegend->AddEntry(hChgWidTIB6, "width = 6-8", "L");
  CWlegend->AddEntry(hChgWidTIB8, "width > 8", "L");
  canvNchgWid->cd(1);
  CWlegend->Draw();
  canvNchgWid->Update();

  canvNchgWid->cd(2);
  hChgWidTIB3->Draw();
  hChgWidTIB1->Draw("same");
  hChgWidTIB2->Draw("same");
  hChgWidTIB4->Draw("same");
  hChgWidTIB5->Draw("same");
  hChgWidTIB6->Draw("same");
  hChgWidTIB8->Draw("same");
  gPad->SetLogy(1);
  canvNchgWid->Update();

  // TOB

  TH1F* hChgWidTOB3 = new TH1F("hChgWidTOB3", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTOB3", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==5 && width == 3");
  hChgWidTOB3->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hChgWidTOB3->SetLineColor(kBlack);
  canvNchgWid->cd(3);
  hChgWidTOB3->Draw();

  TH1F* hChgWidTOB1 = new TH1F("hChgWidTOB1", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTOB1", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==5 && width == 1");
  hChgWidTOB1->SetLineColor(kRed);
  hChgWidTOB1->Draw("same");

  TH1F* hChgWidTOB2 = new TH1F("hChgWidTOB2", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTOB2", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==5 && width == 2");
  hChgWidTOB2->SetLineColor(kBlue);
  hChgWidTOB2->Draw("same");

  TH1F* hChgWidTOB4 = new TH1F("hChgWidTOB4", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTOB4", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==5 && width == 4");
  hChgWidTOB4->SetLineColor(8);
  hChgWidTOB4->Draw("same");

  TH1F* hChgWidTOB5 = new TH1F("hChgWidTOB5", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTOB5", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==5 && width == 5");
  hChgWidTOB5->SetLineColor(7);
  hChgWidTOB5->Draw("same");

  TH1F* hChgWidTOB6 = new TH1F("hChgWidTOB6", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTOB6", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==5 && width >= 6 && width <= 8");
  hChgWidTOB6->SetLineColor(28);
  hChgWidTOB6->Draw("same");

  TH1F* hChgWidTOB8 = new TH1F("hChgWidTOB8", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTOB8", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==5 && width > 8");
  hChgWidTOB8->SetLineColor(6);
  hChgWidTOB8->Draw("same");

  canvNchgWid->cd(4);
  hChgWidTOB3->Draw();
  hChgWidTOB1->Draw("same");
  hChgWidTOB2->Draw("same");
  hChgWidTOB4->Draw("same");
  hChgWidTOB5->Draw("same");
  hChgWidTOB6->Draw("same");
  hChgWidTOB8->Draw("same");
  gPad->SetLogy(1);
  canvNchgWid->Update();

  TCanvas *canvNchgWidEC = new TCanvas("canvNchgWidEC", "canvNchgWidEC", 1100, 800);
  canvNchgWidEC->Divide(2,2);

  // TID

  TH1F* hChgWidTID3 = new TH1F("hChgWidTID3", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTID3", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==4 && width == 3");
  hChgWidTID3->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hChgWidTID3->SetLineColor(kBlack);
  canvNchgWidEC->cd(1);
  hChgWidTID3->Draw();

  TH1F* hChgWidTID1 = new TH1F("hChgWidTID1", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTID1", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==4 && width == 1");
  hChgWidTID1->SetLineColor(kRed);
  hChgWidTID1->Draw("same");

  TH1F* hChgWidTID2 = new TH1F("hChgWidTID2", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTID2", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==4 && width == 2");
  hChgWidTID2->SetLineColor(kBlue);
  hChgWidTID2->Draw("same");

  TH1F* hChgWidTID4 = new TH1F("hChgWidTID4", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTID4", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==4 && width == 4");
  hChgWidTID4->SetLineColor(8);
  hChgWidTID4->Draw("same");

  TH1F* hChgWidTID5 = new TH1F("hChgWidTID5", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTID5", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==4 && width == 5");
  hChgWidTID5->SetLineColor(7);
  hChgWidTID5->Draw("same");

  TH1F* hChgWidTID6 = new TH1F("hChgWidTID6", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTID6", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==4 && width >= 6 && width <= 8");
  hChgWidTID6->SetLineColor(28);
  hChgWidTID6->Draw("same");

  TH1F* hChgWidTID8 = new TH1F("hChgWidTID8", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTID8", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==4 && width > 8");
  hChgWidTID8->SetLineColor(6);
  hChgWidTID8->Draw("same");

  canvNchgWidEC->cd(2);
  hChgWidTID3->Draw();
  hChgWidTID1->Draw("same");
  hChgWidTID2->Draw("same");
  hChgWidTID4->Draw("same");
  hChgWidTID5->Draw("same");
  hChgWidTID6->Draw("same");
  hChgWidTID8->Draw("same");
  gPad->SetLogy(1);
  canvNchgWidEC->Update();

  canvNchgWidEC->cd(1);
  CWlegend->Draw();
  canvNchgWidEC->Update();

  // TEC

  TH1F* hChgWidTEC3 = new TH1F("hChgWidTEC3", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTEC3", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==6 && width == 3");
  hChgWidTEC3->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hChgWidTEC3->SetLineColor(kBlack);
  canvNchgWidEC->cd(3);
  hChgWidTEC3->Draw();

  TH1F* hChgWidTEC1 = new TH1F("hChgWidTEC1", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTEC1", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==6 && width == 1");
  hChgWidTEC1->SetLineColor(kRed);
  hChgWidTEC1->Draw("same");

  TH1F* hChgWidTEC2 = new TH1F("hChgWidTEC2", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTEC2", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==6 && width == 2");
  hChgWidTEC2->SetLineColor(kBlue);
  hChgWidTEC2->Draw("same");

  TH1F* hChgWidTEC4 = new TH1F("hChgWidTEC4", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTEC4", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==6 && width == 4");
  hChgWidTEC4->SetLineColor(8);
  hChgWidTEC4->Draw("same");

  TH1F* hChgWidTEC5 = new TH1F("hChgWidTEC5", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTEC5", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==6 && width == 5");
  hChgWidTEC5->SetLineColor(7);
  hChgWidTEC5->Draw("same");

  TH1F* hChgWidTEC6 = new TH1F("hChgWidTEC6", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTEC6", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==6 && width >= 6 && width <= 8");
  hChgWidTEC6->SetLineColor(28);
  hChgWidTEC6->Draw("same");

  TH1F* hChgWidTEC8 = new TH1F("hChgWidTEC8", "Relative charge", 100, 0, 10);
  tree->Project("hChgWidTEC8", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", "subDet==6 && width > 8");
  hChgWidTEC8->SetLineColor(6);
  hChgWidTEC8->Draw("same");

  canvNchgWidEC->cd(4);
  hChgWidTEC3->Draw();
  hChgWidTEC1->Draw("same");
  hChgWidTEC2->Draw("same");
  hChgWidTEC4->Draw("same");
  hChgWidTEC5->Draw("same");
  hChgWidTEC6->Draw("same");
  hChgWidTEC8->Draw("same");
  gPad->SetLogy(1);
  canvNchgWidEC->Update();

  canvNchgWid->SaveAs("clusNtp_relChgWid.pdf");
  canvNchgWidEC->SaveAs("clusNtp_relChgWidEC.pdf");

}  // ---------------------------------------------------------------

void plot_ChgTnrm_v_Snrm(TTree* tree, const TString& MIPdEdx, const TCut& cut1, const TCut& cut2, const TCut& cut3) {

  TCanvas *canvChgTnrm_v_Snrm = new TCanvas("canvChgTnrm_v_Snrm", "canvChgTnrm_v_Snrm", 1100, 800);
  canvChgTnrm_v_Snrm->Divide(2,2);

  // TIB
  TH1F* hrelChgTIBMItrue = new TH1F("hrelChgTIBMItrue", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTIBMItrue", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut1&&"subDet==3");
  hrelChgTIBMItrue->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTIBMItrue->SetMinimum(10);
  hrelChgTIBMItrue->SetLineColor(8);
  canvChgTnrm_v_Snrm->cd(1);
  hrelChgTIBMItrue->Draw();

  TH1F* hrelChgTIBMIstr = new TH1F("hrelChgTIBMIstr", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTIBMIstr", MIPdEdx+"*charge/max(1.e-6,pathLstraight)", cut1&&"subDet==3");
  hrelChgTIBMIstr->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTIBMIstr->SetLineColor(kBlack);
  hrelChgTIBMIstr->SetLineStyle(kDashed);
  hrelChgTIBMIstr->Draw("sames");

  TH1F* hrelChgTIB2MItrue = new TH1F("hrelChgTIB2MItrue", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTIB2MItrue", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut3&&"subDet==3");
  hrelChgTIB2MItrue->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTIB2MItrue->SetLineColor(kRed);
  hrelChgTIB2MItrue->Draw("sames");

  TH1F* hrelChgTIB2MIstr = new TH1F("hrelChgTIB2MIstr", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTIB2MIstr", MIPdEdx+"*charge/max(1.e-6,pathLstraight)", cut3&&"subDet==3");
  hrelChgTIB2MIstr->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTIB2MIstr->SetLineColor(kBlue);
  hrelChgTIB2MIstr->SetLineStyle(kDashed);
  hrelChgTIB2MIstr->Draw("sames");

  gPad->SetLogy(1);

  canvChgTnrm_v_Snrm->Update();  // without this the "stats" pointers are null
  double top = 0.0;
  drawStats(hrelChgTIBMItrue, top);
  drawStats(hrelChgTIBMIstr, top);
  drawStats(hrelChgTIB2MItrue, top);
  drawStats(hrelChgTIB2MIstr, top);
  canvChgTnrm_v_Snrm->Update();

  // TID
  TH1F* hrelChgTIDMItrue = new TH1F("hrelChgTIDMItrue", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTIDMItrue", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut1&&"subDet==4");
  hrelChgTIDMItrue->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTIDMItrue->SetMinimum(1);
  hrelChgTIDMItrue->SetLineColor(8);
  canvChgTnrm_v_Snrm->cd(2);
  hrelChgTIDMItrue->Draw();

  TH1F* hrelChgTIDMIstr = new TH1F("hrelChgTIDMIstr", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTIDMIstr", MIPdEdx+"*charge/max(1.e-6,pathLstraight)", cut1&&"subDet==4");
  hrelChgTIDMIstr->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTIDMIstr->SetLineColor(kBlack);
  hrelChgTIDMIstr->SetLineStyle(kDashed);
  hrelChgTIDMIstr->Draw("sames");

  TH1F* hrelChgTID2MItrue = new TH1F("hrelChgTID2MItrue", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTID2MItrue", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut3&&"subDet==4");
  hrelChgTID2MItrue->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTID2MItrue->SetLineColor(kRed);
  hrelChgTID2MItrue->Draw("sames");

  TH1F* hrelChgTID2MIstr = new TH1F("hrelChgTID2MIstr", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTID2MIstr", MIPdEdx+"*charge/max(1.e-6,pathLstraight)", cut3&&"subDet==4");
  hrelChgTID2MIstr->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTID2MIstr->SetLineColor(kBlue);
  hrelChgTID2MIstr->SetLineStyle(kDashed);
  hrelChgTID2MIstr->Draw("sames");

  gPad->SetLogy(1);

  canvChgTnrm_v_Snrm->Update();  // without this the "stats" pointers are null
  top = 0.0;
  drawStats(hrelChgTIDMItrue, top);
  drawStats(hrelChgTIDMIstr, top);
  drawStats(hrelChgTID2MItrue, top);
  drawStats(hrelChgTID2MIstr, top);
  canvChgTnrm_v_Snrm->Update();

  // TOB
  TH1F* hrelChgTOBMItrue = new TH1F("hrelChgTOBMItrue", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTOBMItrue", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut1&&"subDet==5");
  hrelChgTOBMItrue->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTOBMItrue->SetMinimum(10);
  hrelChgTOBMItrue->SetLineColor(8);
  canvChgTnrm_v_Snrm->cd(3);
  hrelChgTOBMItrue->Draw();

  TH1F* hrelChgTOBMIstr = new TH1F("hrelChgTOBMIstr", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTOBMIstr", MIPdEdx+"*charge/max(1.e-6,pathLstraight)", cut1&&"subDet==5");
  hrelChgTOBMIstr->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTOBMIstr->SetLineColor(kBlack);
  hrelChgTOBMIstr->SetLineStyle(kDashed);
  hrelChgTOBMIstr->Draw("sames");

  TH1F* hrelChgTOB2MItrue = new TH1F("hrelChgTOB2MItrue", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTOB2MItrue", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut3&&"subDet==5");
  hrelChgTOB2MItrue->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTOB2MItrue->SetLineColor(kRed);
  hrelChgTOB2MItrue->Draw("sames");

  TH1F* hrelChgTOB2MIstr = new TH1F("hrelChgTOB2MIstr", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTOB2MIstr", MIPdEdx+"*charge/max(1.e-6,pathLstraight)", cut3&&"subDet==5");
  hrelChgTOB2MIstr->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTOB2MIstr->SetLineColor(kBlue);
  hrelChgTOB2MIstr->SetLineStyle(kDashed);
  hrelChgTOB2MIstr->Draw("sames");

  gPad->SetLogy(1);

  canvChgTnrm_v_Snrm->Update();  // without this the "stats" pointers are null
  top = 0.0;
  drawStats(hrelChgTOBMItrue, top);
  drawStats(hrelChgTOBMIstr, top);
  drawStats(hrelChgTOB2MItrue, top);
  drawStats(hrelChgTOB2MIstr, top);
  canvChgTnrm_v_Snrm->Update();

  // TEC
  TH1F* hrelChgTECMItrue = new TH1F("hrelChgTECMItrue", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTECMItrue", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut1&&"subDet==6");
  hrelChgTECMItrue->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTECMItrue->SetMinimum(1);
  hrelChgTECMItrue->SetLineColor(8);
  canvChgTnrm_v_Snrm->cd(4);
  hrelChgTECMItrue->Draw();

  TH1F* hrelChgTECMIstr = new TH1F("hrelChgTECMIstr", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTECMIstr", MIPdEdx+"*charge/max(1.e-6,pathLstraight)", cut1&&"subDet==6");
  hrelChgTECMIstr->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTECMIstr->SetLineColor(kBlack);
  hrelChgTECMIstr->SetLineStyle(kDashed);
  hrelChgTECMIstr->Draw("sames");

  TH1F* hrelChgTEC2MItrue = new TH1F("hrelChgTEC2MItrue", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTEC2MItrue", MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut3&&"subDet==6");
  hrelChgTEC2MItrue->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTEC2MItrue->SetLineColor(kRed);
  hrelChgTEC2MItrue->Draw("sames");

  TH1F* hrelChgTEC2MIstr = new TH1F("hrelChgTEC2MIstr", "Relative charge", 100, 0, 10);
  tree->Project("hrelChgTEC2MIstr", MIPdEdx+"*charge/max(1.e-6,pathLstraight)", cut3&&"subDet==6");
  hrelChgTEC2MIstr->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hrelChgTEC2MIstr->SetLineColor(kBlue);
  hrelChgTEC2MIstr->SetLineStyle(kDashed);
  hrelChgTEC2MIstr->Draw("sames");

  gPad->SetLogy(1);

  canvChgTnrm_v_Snrm->Update();  // without this the "stats" pointers are null
  top = 0.0;
  drawStats(hrelChgTECMItrue, top);
  drawStats(hrelChgTECMIstr, top);
  drawStats(hrelChgTEC2MItrue, top);
  drawStats(hrelChgTEC2MIstr, top);
  canvChgTnrm_v_Snrm->Update();

  TLegend* Rtlegend = new TLegend(.30, .67, .74, .87, "");
  Rtlegend->AddEntry(hrelChgTIBMItrue, "1 crossing primary, true path", "L");
  Rtlegend->AddEntry(hrelChgTIBMIstr, "1 crossing primary, straight path", "L");
  Rtlegend->AddEntry(hrelChgTIB2MItrue, "2 crossing primaries, true path", "L");
  Rtlegend->AddEntry(hrelChgTIB2MIstr, "2 crossing primaries, straight path", "L");
  canvChgTnrm_v_Snrm->cd(1);
  Rtlegend->Draw();
  canvChgTnrm_v_Snrm->Update();

  canvChgTnrm_v_Snrm->SaveAs("clusNtp_ChgTnrm_v_Snrm.pdf");

}  // ---------------------------------------------------------------


void plot_ChgPath(TTree* tree, const TCut& cut) {

  TCanvas *canvChgPath = new TCanvas("canvChgPath", "canvChgPath", 1100, 800);
  canvChgPath->Divide(2,2);
  gStyle->SetPalette(1);

  // TIB
  TH2F* hChgPathTIBMI = new TH2F("hChgPathTIBMI", "Charge vs Path length", 140, 280, 420, 110, 0, 550);
  tree->Project("hChgPathTIBMI", "charge:10000*secondPathLength", cut&&"subDet==3");
  hChgPathTIBMI->GetXaxis()->SetTitle("MC path (microns)");
  hChgPathTIBMI->GetYaxis()->SetTitle("Raw charge");
  hChgPathTIBMI->SetLineColor(8);
  canvChgPath->cd(1);
  hChgPathTIBMI->Draw("colz");

  // TID
  TH2F* hChgPathTIDMI = new TH2F("hChgPathTIDMI", "Charge vs Path length", 240, 280, 400, 110, 0, 550);
  tree->Project("hChgPathTIDMI", "charge:10000*secondPathLength", cut&&"subDet==4");
  hChgPathTIDMI->GetXaxis()->SetTitle("MC path (microns)");
  hChgPathTIDMI->GetYaxis()->SetTitle("Raw charge");
  hChgPathTIDMI->SetLineColor(8);
  canvChgPath->cd(2);
  hChgPathTIDMI->Draw("colz");

  // TOB
  TH2F* hChgPathTOBMI = new TH2F("hChgPathTOBMI", "Charge vs Path length", 170, 460, 630, 110, 0, 550);
  tree->Project("hChgPathTOBMI", "charge:10000*secondPathLength", cut&&"subDet==5");
  hChgPathTOBMI->GetXaxis()->SetTitle("MC path (microns)");
  hChgPathTOBMI->GetYaxis()->SetTitle("Raw charge");
  hChgPathTOBMI->SetLineColor(8);
  canvChgPath->cd(3);
  hChgPathTOBMI->Draw("colz");

  // TEC
  TH2F* hChgPathTECMI = new TH2F("hChgPathTECMI", "Charge vs Path length", 320, 280, 600, 110, 0, 550);
  tree->Project("hChgPathTECMI", "charge:10000*secondPathLength", cut&&"subDet==6");
  hChgPathTECMI->GetXaxis()->SetTitle("MC path (microns)");
  hChgPathTECMI->GetYaxis()->SetTitle("Raw charge");
  hChgPathTECMI->SetLineColor(8);
  canvChgPath->cd(4);
  hChgPathTECMI->Draw("colz");

  canvChgPath->SaveAs("clusNtp_chgPath.pdf");

}  // ---------------------------------------------------------------

void plot_ChgAsym(TTree* tree, const TString& MIPdEdx, const TCut& cut) {

  TCanvas *canvChgAsym = new TCanvas("canvChgAsym", "canvChgAsym", 1100, 800);
  canvChgAsym->Divide(2,2);
  gStyle->SetPalette(1);

  // TIB
  TH2F* hChgAsymTIB = new TH2F("hChgAsymTIB", "Charge asymmetry vs norm. charge", 100, 0, 10, 100, -1, 1);
  tree->Project("hChgAsymTIB", "(1-2*tkFlip)*(firstTkChg-secondTkChg)/(firstTkChg+secondTkChg):"+MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut&&"subDet==3");
  hChgAsymTIB->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hChgAsymTIB->GetYaxis()->SetTitle("2-track charge asymmetry");
  hChgAsymTIB->SetLineColor(8);
  hChgAsymTIB->SetMaximum(80);
  canvChgAsym->cd(1);
  hChgAsymTIB->Draw("colz");

  // TID
  TH2F* hChgAsymTID = new TH2F("hChgAsymTID", "Charge asymmetry vs norm. charge", 100, 0, 10, 100, -1, 1);
  tree->Project("hChgAsymTID", "(1-2*tkFlip)*(firstTkChg-secondTkChg)/(firstTkChg+secondTkChg):"+MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut&&"subDet==4");
  hChgAsymTID->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hChgAsymTID->GetYaxis()->SetTitle("2-track charge asymmetry");
  hChgAsymTID->SetLineColor(8);
  hChgAsymTID->SetMaximum(20);
  canvChgAsym->cd(2);
  hChgAsymTID->Draw("colz");

  // TOB
  TH2F* hChgAsymTOB = new TH2F("hChgAsymTOB", "Charge asymmetry vs norm. charge", 100, 0, 10, 100, -1, 1);
  tree->Project("hChgAsymTOB", "(1-2*tkFlip)*(firstTkChg-secondTkChg)/(firstTkChg+secondTkChg):"+MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut&&"subDet==5");
  hChgAsymTOB->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hChgAsymTOB->GetYaxis()->SetTitle("2-track charge asymmetry");
  hChgAsymTOB->SetLineColor(8);
  hChgAsymTOB->SetMaximum(80);
  canvChgAsym->cd(3);
  hChgAsymTOB->Draw("colz");

  // TEC
  TH2F* hChgAsymTEC = new TH2F("hChgAsymTEC", "Charge asymmetry vs norm. charge", 100, 0, 10, 100, -1, 1);
  tree->Project("hChgAsymTEC", "(1-2*tkFlip)*(firstTkChg-secondTkChg)/(firstTkChg+secondTkChg):"+MIPdEdx+"*charge/max(1.e-6,firstPathLength)", cut&&"subDet==6");
  hChgAsymTEC->GetXaxis()->SetTitle("Charge/(path length), MIP");
  hChgAsymTEC->GetYaxis()->SetTitle("2-track charge asymmetry");
  hChgAsymTEC->SetLineColor(8);
  hChgAsymTEC->SetMaximum(20);
  canvChgAsym->cd(4);
  hChgAsymTEC->Draw("colz");

  canvChgAsym->SaveAs("clusNtp_chgAsym.pdf");

}  // ---------------------------------------------------------------

void plot_Asym(TTree* tree, const TCut& cut) {

  TCanvas *canvAsym = new TCanvas("canvAsym", "canvAsym", 1100, 800);
  canvAsym->Divide(2,2);
  gStyle->SetPalette(1);

  // TIB
  TH1F* hAsymTIB = new TH1F("hAsymTIB", "Charge asymmetry vs norm. charge", 115, -1.15, 1.15);
  tree->Project("hAsymTIB", "(1-2*tkFlip)*(firstTkChg-secondTkChg)/(firstTkChg+secondTkChg)", cut&&"subDet==3");
  hAsymTIB->GetXaxis()->SetTitle("2-track charge asymmetry");
  canvAsym->cd(1);
  hAsymTIB->Draw();
  canvAsym->Update();

  // TID
  TH1F* hAsymTID = new TH1F("hAsymTID", "Charge asymmetry", 100, -1.15, 1.15);
  tree->Project("hAsymTID", "(1-2*tkFlip)*(firstTkChg-secondTkChg)/(firstTkChg+secondTkChg)", cut&&"subDet==4");
  hAsymTID->GetXaxis()->SetTitle("2-track charge asymmetry");
  canvAsym->cd(2);
  hAsymTID->Draw();
  canvAsym->Update();

  // TOB
  TH1F* hAsymTOB = new TH1F("hAsymTOB", "Charge asymmetry", 100, -1.15, 1.15);
  tree->Project("hAsymTOB", "(1-2*tkFlip)*(firstTkChg-secondTkChg)/(firstTkChg+secondTkChg)", cut&&"subDet==5");
  hAsymTOB->GetXaxis()->SetTitle("2-track charge asymmetry");
  canvAsym->cd(3);
  hAsymTOB->Draw();
  canvAsym->Update();

  // TEC
  TH1F* hAsymTEC = new TH1F("hAsymTEC", "Charge asymmetry", 100, -1.15, 1.15);
  tree->Project("hAsymTEC", "(1-2*tkFlip)*(firstTkChg-secondTkChg)/(firstTkChg+secondTkChg)", cut&&"subDet==6");
  hAsymTEC->GetXaxis()->SetTitle("2-track charge asymmetry");
  canvAsym->cd(4);
  hAsymTEC->Draw();
  canvAsym->Update();

  canvAsym->SaveAs("clusNtp_asym.pdf");

}  // ---------------------------------------------------------------

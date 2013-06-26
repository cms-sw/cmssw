void V0ValHistoPublisher(char* newFile="NEW_FILE", char* refFile="REF_FILE") {

  gROOT->Reset();
  gROOT->SetBatch();

  // Many of these style statements will likely require tweaking
  //  to work with the V0 histograms
  gROOT->SetStyle("Plain");
  gStyle->SetPadGridX(kTRUE);
  gStyle->SetPadGridY(kTRUE);
  gStyle->SetPadRightMargin(0.07);
  gStyle->SetPadLeftMargin(0.13);
  gStyle->SetOptTitle(1);
  //gStyle->SetTitleXSize(0.07); 
  //gStyle->SetTitleXOffset(0.6); 
  //tyle->SetTitleYSize(0.3);
  //gStyle->SetLabelSize(0.6) 
  //gStyle->SetTextSize(0.5);
  char* refLabel("REF_LABEL, REF_RELEASE REFSELECTION");
  char* newLabel("NEW_LABEL, NEW_RELEASE NEWSELECTION");
  //char* refLabel2("");

  TFile* infile1 = new TFile(refFile);
  infile1->cd("DQMData/Vertexing/V0V/EffFakes");
  TDirectory* refdir = gDirectory;
  TList* hList1 = refdir->GetListOfKeys();
  infile1->cd("DQMData/Vertexing/V0V/Other");
  TDirectory* refdir_1 = gDirectory;
  TList* hList1_1 = refdir_1->GetListOfKeys();

  TFile* infile2 = new TFile(newFile);
  infile2->cd("DQMData/Vertexing/V0V/EffFakes");
  TDirectory* newdir = gDirectory;
  TList* hList2 = newdir->GetListOfKeys();
  infile2->cd("DQMData/Vertexing/V0V/Other");
  TDirectory* newdir_1 = gDirectory;
  TList* hList2_1 = newdir_1->GetListOfKeys();

  TCanvas* canvas;

  // Kshort plots
  TH1F *ksEffEta, *ksTkEffEta;
  TH1F *ksEffPt, *ksTkEffPt;
  TH1F *ksEffR, *ksTkEffR;

  TH1F *ksNewEffEta, *ksNewTkEffEta;
  TH1F *ksNewEffPt, *ksNewTkEffPt;
  TH1F *ksNewEffR, *ksNewTkEffR;

  TH1F *ksFakeEta, *ksTkFakeEta;
  TH1F *ksFakePt, *ksTkFakePt;
  TH1F *ksFakeR, *ksTkFakeR;

  TH1F *ksNewFakeEta, *ksNewTkFakeEta;
  TH1F *ksNewFakePt, *ksNewTkFakePt;
  TH1F *ksNewFakeR, *ksNewTkFakeR;

  TH1F *ksNewFakeMass, *ksNewGoodMass;
  TH1F *ksNewMassAll;
  TH1F *ksFakeMass, *ksGoodMass;
  TH1F* ksMassAll;

  // K0s Efficiency plots from reference release
  //refdir->GetObject(hList1->At(0)->GetName(), ksEffEta);
  refdir->GetObject(hList1->FindObject("K0sEffVsEta")->GetName(), ksEffEta);
  ksEffEta->GetYaxis()->SetRangeUser(0, 1.1);
  
  //refdir->GetObject(hList1->At(6)->GetName(), ksTkEffEta);
  refdir->GetObject(hList1->FindObject("K0sTkEffVsEta")->GetName(), ksTkEffEta);
  ksTkEffEta->GetYaxis()->SetRangeUser(0, 1.1);

  //refdir->GetObject(hList1->At(1)->GetName(), ksEffPt);
  refdir->GetObject(hList1->FindObject("K0sEffVsPt")->GetName(), ksEffPt);
  ksEffPt->GetYaxis()->SetRangeUser(0, 1.1);
  ksEffPt->GetYaxis()->SetTitle("K^{0}_{S} Efficiency vs p_{T}");
  ksEffPt->GetYaxis()->SetTitleSize(0.05);
  ksEffPt->GetYaxis()->SetTitleOffset(1.2);
  ksEffPt->SetTitle("");

  //refdir->GetObject(hList1->At(7)->GetName(), ksTkEffPt);
  refdir->GetObject(hList1->FindObject("K0sTkEffVsPt")->GetName(), ksTkEffPt);
  ksTkEffPt->GetYaxis()->SetRangeUser(0, 1.1);
  ksTkEffPt->GetYaxis()->SetTitle("K^{0}_{S} Tracking Efficiency vs p_{T}");
  ksTkEffPt->GetYaxis()->SetTitleSize(0.05);
  ksTkEffPt->GetYaxis()->SetTitleOffset(1.2);
  ksTkEffPt->SetTitle("");

  //refdir->GetObject(hList1->At(2)->GetName(), ksEffR);
  refdir->GetObject(hList1->FindObject("K0sEffVsR")->GetName(), ksEffR);
  ksEffR->GetYaxis()->SetRangeUser(0, 1.1);

  //refdir->GetObject(hList1->At(8)->GetName(), ksTkEffR);
  refdir->GetObject(hList1->FindObject("K0sTkEffVsR")->GetName(), ksTkEffR);
  ksTkEffR->GetYaxis()->SetRangeUser(0, 1.1);

  // K0S mass plots from ref release
  refdir_1->GetObject(hList1_1->FindObject("ksMassFake")->GetName(), ksFakeMass);
  ksFakeMass->SetXTitle(//"Mass of fake Ks, ref release");
			refLabel);
  refdir_1->GetObject(hList1_1->FindObject("ksMassGood")->GetName(), ksGoodMass);
  ksGoodMass->SetXTitle(//"Mass of good Ks, ref release");
			refLabel);
  refdir_1->GetObject(hList1_1->FindObject("ksMassAll")->GetName(), ksMassAll);
  ksMassAll->SetXTitle(//"Mass of all found Ks, ref release");
		       refLabel);


  // K0s efficiency plots from new release
  //newdir->GetObject(hList1->At(0)->GetName(), ksNewEffEta);
  newdir->GetObject(hList2->FindObject("K0sEffVsEta")->GetName(), ksNewEffEta);
  ksNewEffEta->GetYaxis()->SetRangeUser(0, 1.1);

  //newdir->GetObject(hList1->At(6)->GetName(), ksNewTkEffEta);
  newdir->GetObject(hList2->FindObject("K0sTkEffVsEta")->GetName(), ksNewTkEffEta);
  ksNewTkEffEta->GetYaxis()->SetRangeUser(0, 1.1);

  //newdir->GetObject(hList1->At(1)->GetName(), ksNewEffPt);
  newdir->GetObject(hList2->FindObject("K0sEffVsPt")->GetName(), ksNewEffPt);
  ksNewEffPt->GetYaxis()->SetRangeUser(0, 1.1);
  ksNewEffPt->GetYaxis()->SetRangeUser(0, 1.1);
  ksNewEffPt->GetYaxis()->SetTitle("K^{0}_{S} Efficiency vs p_{T}");
  ksNewEffPt->GetYaxis()->SetTitleSize(0.05);
  ksNewEffPt->GetYaxis()->SetTitleOffset(1.2);
  ksNewEffPt->SetTitle("");

  //newdir->GetObject(hList1->At(7)->GetName(), ksNewTkEffPt);
  newdir->GetObject(hList2->FindObject("K0sTkEffVsPt")->GetName(), ksNewTkEffPt);
  ksNewTkEffPt->GetYaxis()->SetRangeUser(0, 1.1);
  ksNewTkEffPt->GetYaxis()->SetTitle("K^{0}_{S} Tracking Efficiency vs p_{T}");
  ksNewTkEffPt->GetYaxis()->SetTitleSize(0.05);
  ksNewTkEffPt->GetYaxis()->SetTitleOffset(1.2);
  ksNewTkEffPt->SetTitle("");

  //newdir->GetObject(hList1->At(2)->GetName(), ksNewEffR);
  newdir->GetObject(hList2->FindObject("K0sEffVsR")->GetName(), ksNewEffR);
  ksNewEffR->GetYaxis()->SetRangeUser(0, 1.1);

  //newdir->GetObject(hList1->At(8)->GetName(), ksNewTkEffR);
  newdir->GetObject(hList2->FindObject("K0sTkEffVsR")->GetName(), ksNewTkEffR);
  ksNewTkEffR->GetYaxis()->SetRangeUser(0, 1.1);

  // K0S mass plots from new release
  newdir_1->GetObject(hList2_1->FindObject("ksMassFake")->GetName(), ksNewFakeMass);
  ksNewFakeMass->SetXTitle(//"Mass of fake Ks, new release");
			   newLabel);
  newdir_1->GetObject(hList2_1->FindObject("ksMassGood")->GetName(), ksNewGoodMass);
  ksNewGoodMass->SetXTitle(//"Mass of good Ks, new release");
			   newLabel);
  newdir_1->GetObject(hList2_1->FindObject("ksMassAll")->GetName(), ksNewMassAll);
  ksNewMassAll->SetXTitle(//"Mass of all found Ks, new release");
			  newLabel);


  // K0s fake rate plots from reference release
  //refdir->GetObject(hList1->At(3)->GetName(), ksFakeEta);
  refdir->GetObject(hList1->FindObject("K0sFakeVsEta")->GetName(), ksFakeEta);
  ksFakeEta->GetYaxis()->SetRangeUser(0, 1.1);

  //refdir->GetObject(hList1->At(9)->GetName(), ksTkFakeEta);
  refdir->GetObject(hList1->FindObject("K0sTkFakeVsEta")->GetName(), ksTkFakeEta);
  ksTkFakeEta->GetYaxis()->SetRangeUser(0, 1.1);

  //refdir->GetObject(hList1->At(4)->GetName(), ksFakePt);
  refdir->GetObject(hList1->FindObject("K0sFakeVsPt")->GetName(), ksFakePt);
  ksFakePt->GetYaxis()->SetRangeUser(0, 1.1);
  ksFakePt->GetYaxis()->SetRangeUser(0, 1.1);
  ksFakePt->GetYaxis()->SetTitle("K^{0}_{S} Fake Rate vs p_{T}");
  ksFakePt->GetYaxis()->SetTitleSize(0.05);
  ksFakePt->GetYaxis()->SetTitleOffset(1.2);
  ksFakePt->SetTitle("");

  //refdir->GetObject(hList1->At(10)->GetName(), ksTkFakePt);
  refdir->GetObject(hList1->FindObject("K0sTkFakeVsPt")->GetName(), ksTkFakePt);
  ksTkFakePt->GetYaxis()->SetRangeUser(0, 1.1);
  ksTkFakePt->GetYaxis()->SetTitle("K^{0}_{S} Tracking Fake Rate vs p_{T}");
  ksTkFakePt->GetYaxis()->SetTitleSize(0.05);
  ksTkFakePt->GetYaxis()->SetTitleOffset(1.2);
  ksTkFakePt->SetTitle("");

  //refdir->GetObject(hList1->At(5)->GetName(), ksFakeR);
  refdir->GetObject(hList1->FindObject("K0sFakeVsR")->GetName(), ksFakeR);
  ksFakeR->GetYaxis()->SetRangeUser(0, 1.1);

  //refdir->GetObject(hList1->At(11)->GetName(), ksTkFakeR);
  refdir->GetObject(hList1->FindObject("K0sTkFakeVsR")->GetName(), ksTkFakeR);
  ksTkFakeR->GetYaxis()->SetRangeUser(0, 1.1);



  // Kshort plots from new release we're testing
  //newdir->GetObject(hList1->At(3)->GetName(), ksNewFakeEta);
  newdir->GetObject(hList2->FindObject("K0sFakeVsEta")->GetName(), ksNewFakeEta);
  ksNewFakeEta->GetYaxis()->SetRangeUser(0, 1.1);

  //newdir->GetObject(hList1->At(9)->GetName(), ksNewTkFakeEta);
  newdir->GetObject(hList2->FindObject("K0sTkFakeVsEta")->GetName(), ksNewTkFakeEta);
  ksNewTkFakeEta->GetYaxis()->SetRangeUser(0, 1.1);

  //newdir->GetObject(hList1->At(4)->GetName(), ksNewFakePt);
  newdir->GetObject(hList2->FindObject("K0sFakeVsPt")->GetName(), ksNewFakePt);
  ksNewFakePt->GetYaxis()->SetRangeUser(0, 1.1);
  ksNewFakePt->SetTitle("");

  //newdir->GetObject(hList1->At(10)->GetName(), ksNewTkFakePt);
  newdir->GetObject(hList2->FindObject("K0sTkFakeVsPt")->GetName(), ksNewTkFakePt);
  ksNewTkFakePt->GetYaxis()->SetRangeUser(0, 1.1);
  ksNewTkFakePt->SetTitle("");

  //newdir->GetObject(hList1->At(5)->GetName(), ksNewFakeR);
  newdir->GetObject(hList2->FindObject("K0sFakeVsR")->GetName(), ksNewFakeR);
  ksNewFakeR->GetYaxis()->SetRangeUser(0, 1.1);

  //newdir->GetObject(hList1->At(11)->GetName(), ksNewTkFakeR);
  newdir->GetObject(hList2->FindObject("K0sTkFakeVsR")->GetName(), ksNewTkFakeR);
  ksNewTkFakeR->GetYaxis()->SetRangeUser(0, 1.1);


  canvas = new TCanvas("Kshorts", "K^{0}_{S} Efficiency", 
		       1000, 1400);

  ksEffEta->SetMarkerStyle(20);
  ksEffEta->SetMarkerColor(2);
  ksEffEta->SetMarkerSize(0.7);
  ksEffEta->SetLineColor(2);
  ksNewEffEta->SetMarkerStyle(20);
  ksNewEffEta->SetMarkerColor(4);
  ksNewEffEta->SetMarkerSize(0.7);
  ksNewEffEta->SetLineColor(4);

  ksTkEffEta->SetMarkerStyle(20);
  ksTkEffEta->SetMarkerColor(2);
  ksTkEffEta->SetMarkerSize(0.7);
  ksTkEffEta->SetLineColor(2);
  ksNewTkEffEta->SetMarkerStyle(20);
  ksNewTkEffEta->SetMarkerColor(4);
  ksNewTkEffEta->SetMarkerSize(0.7);
  ksNewTkEffEta->SetLineColor(4);

  ksEffPt->SetMarkerStyle(20);
  ksEffPt->SetMarkerColor(2);
  ksEffPt->SetMarkerSize(0.7);
  ksEffPt->SetLineColor(2);
  ksNewEffPt->SetMarkerStyle(20);
  ksNewEffPt->SetMarkerColor(4);
  ksNewEffPt->SetMarkerSize(0.7);
  ksNewEffPt->SetLineColor(4);

  ksTkEffPt->SetMarkerStyle(20);
  ksTkEffPt->SetMarkerColor(2);
  ksTkEffPt->SetMarkerSize(0.7);
  ksTkEffPt->SetLineColor(2);
  ksNewTkEffPt->SetMarkerStyle(20);
  ksNewTkEffPt->SetMarkerColor(4);
  ksNewTkEffPt->SetMarkerSize(0.7);
  ksNewTkEffPt->SetLineColor(4);

  ksEffR->SetMarkerStyle(20);
  ksEffR->SetMarkerColor(2);
  ksEffR->SetMarkerSize(0.7);
  ksEffR->SetLineColor(2);
  ksNewEffR->SetMarkerStyle(20);
  ksNewEffR->SetMarkerColor(4);
  ksNewEffR->SetMarkerSize(0.7);
  ksNewEffR->SetLineColor(4);

  ksTkEffR->SetMarkerStyle(20);
  ksTkEffR->SetMarkerColor(2);
  ksTkEffR->SetMarkerSize(0.7);
  ksTkEffR->SetLineColor(2);
  ksNewTkEffR->SetMarkerStyle(20);
  ksNewTkEffR->SetMarkerColor(4);
  ksNewTkEffR->SetMarkerSize(0.7);
  ksNewTkEffR->SetLineColor(4);

  canvas->Divide(2,3);

  canvas->cd(1);
  setStats(ksEffEta, ksNewEffEta, -1, 0, false);
  ksEffEta->Draw();
  ksNewEffEta->Draw("sames");

  canvas->cd(2);
  setStats(ksTkEffEta, ksNewTkEffEta, -1, 0, false);
  ksTkEffEta->Draw();
  ksNewTkEffEta->Draw("sames");

  canvas->cd(3);
  setStats(ksEffPt, ksNewEffPt, -1, 0, false);
  ksEffPt->Draw();
  ksNewEffPt->Draw("sames");

  canvas->cd(4);
  setStats(ksTkEffPt, ksNewTkEffPt, 0.6, 0.65, false);
  ksTkEffPt->Draw();
  ksNewTkEffPt->Draw("sames");

  canvas->cd(5);
  setStats(ksEffR, ksNewEffR, -1, 0, false);
  ksEffR->Draw();
  ksNewEffR->Draw("sames");

  canvas->cd(6);
  setStats(ksTkEffR, ksNewTkEffR, 0.6, 0.65, false);
  ksTkEffR->Draw();
  ksNewTkEffR->Draw("sames");

  canvas->cd();
  leg1 = new TLegend(0.20, 0.64, 0.80, 0.69);
  leg1->SetTextSize(0.012);
  leg1->SetLineColor(1);
  leg1->SetLineWidth(1);
  leg1->SetLineStyle(1);
  leg1->SetFillColor(0);
  leg1->SetBorderSize(3);
  leg1->AddEntry(ksEffEta, refLabel, "LPF");
  leg1->AddEntry(ksNewEffEta, newLabel, "LPF");
  leg1->Draw();

  canvas->Print("K0sEff.png");
  canvas->Print("K0sEff.pdf");

  delete leg1;
  //delete canvas;

  ksmcanvas = new TCanvas("KsMass", "K^{0}_{S} mass plots",
			  1000, 1400);

  ksFakeMass->SetLineColor(2);
  ksFakeMass->SetMarkerStyle(20);
  ksFakeMass->SetMarkerColor(2);
  ksNewFakeMass->SetLineColor(4);
  ksNewFakeMass->SetMarkerStyle(20);
  ksNewFakeMass->SetMarkerColor(4);

  ksGoodMass->SetLineColor(2);
  ksGoodMass->SetMarkerStyle(20);
  ksGoodMass->SetMarkerColor(2);
  ksNewGoodMass->SetLineColor(4);
  ksNewGoodMass->SetMarkerStyle(20);
  ksNewGoodMass->SetMarkerColor(4);

  ksMassAll->SetLineColor(2);
  ksMassAll->SetMarkerStyle(20);
  ksMassAll->SetMarkerColor(2);
  ksNewMassAll->SetLineColor(4);
  ksNewMassAll->SetMarkerStyle(20);
  ksNewMassAll->SetMarkerColor(4);

  ksmcanvas->Divide(2,3);
  ksmcanvas->cd(1);
  ksMassAll->Draw();

  ksmcanvas->cd(2);
  ksNewMassAll->Draw();

  ksmcanvas->cd(3);
  ksGoodMass->Draw();

  ksmcanvas->cd(4);
  ksNewGoodMass->Draw();

  ksmcanvas->cd(5);
  ksFakeMass->Draw();

  ksmcanvas->cd(6);
  ksNewFakeMass->Draw();

  ksmcanvas->Print("KsMass.png");
  ksmcanvas->Print("KsMass.pdf");


  canvas = new TCanvas("Kshorts", "K^{0}_{S} Fake Rate", 
		       1000, 1400);

  ksFakeEta->SetMarkerStyle(20);
  ksFakeEta->SetMarkerColor(2);
  ksFakeEta->SetMarkerSize(0.5);
  ksFakeEta->SetLineColor(2);
  ksNewFakeEta->SetMarkerStyle(20);
  ksNewFakeEta->SetMarkerColor(4);
  ksNewFakeEta->SetMarkerSize(0.5);
  ksNewFakeEta->SetLineColor(4);

  ksTkFakeEta->SetMarkerStyle(20);
  ksTkFakeEta->SetMarkerColor(2);
  ksTkFakeEta->SetMarkerSize(0.5);
  ksTkFakeEta->SetLineColor(2);
  ksNewTkFakeEta->SetMarkerStyle(20);
  ksNewTkFakeEta->SetMarkerColor(4);
  ksNewTkFakeEta->SetMarkerSize(0.5);
  ksNewTkFakeEta->SetLineColor(4);

  ksFakePt->SetMarkerStyle(20);
  ksFakePt->SetMarkerColor(2);
  ksFakePt->SetMarkerSize(0.5);
  ksFakePt->SetLineColor(2);
  ksNewFakePt->SetMarkerStyle(20);
  ksNewFakePt->SetMarkerColor(4);
  ksNewFakePt->SetMarkerSize(0.5);
  ksNewFakePt->SetLineColor(4);

  ksTkFakePt->SetMarkerStyle(20);
  ksTkFakePt->SetMarkerColor(2);
  ksTkFakePt->SetMarkerSize(0.5);
  ksTkFakePt->SetLineColor(2);
  ksNewTkFakePt->SetMarkerStyle(20);
  ksNewTkFakePt->SetMarkerColor(4);
  ksNewTkFakePt->SetMarkerSize(0.5);
  ksNewTkFakePt->SetLineColor(4);

  ksFakeR->SetMarkerStyle(20);
  ksFakeR->SetMarkerColor(2);
  ksFakeR->SetMarkerSize(0.5);
  ksFakeR->SetLineColor(2);
  ksNewFakeR->SetMarkerStyle(20);
  ksNewFakeR->SetMarkerColor(4);
  ksNewFakeR->SetMarkerSize(0.5);
  ksNewFakeR->SetLineColor(4);

  ksTkFakeR->SetMarkerStyle(20);
  ksTkFakeR->SetMarkerColor(2);
  ksTkFakeR->SetMarkerSize(0.5);
  ksTkFakeR->SetLineColor(2);
  ksNewTkFakeR->SetMarkerStyle(20);
  ksNewTkFakeR->SetMarkerColor(4);
  ksNewTkFakeR->SetMarkerSize(0.5);
  ksNewTkFakeR->SetLineColor(4);

  canvas->Divide(2,3);

  canvas->cd(1);
  setStats(ksFakeEta, ksNewFakeEta, -1, 0, false);
  ksFakeEta->Draw();
  ksNewFakeEta->Draw("sames");

  canvas->cd(2);
  setStats(ksTkFakeEta, ksNewTkFakeEta, -1, 0, false);
  ksTkFakeEta->Draw();
  ksNewTkFakeEta->Draw("sames");

  canvas->cd(3);
  setStats(ksFakePt, ksNewFakePt, -1, 0, false);
  ksFakePt->Draw();
  ksNewFakePt->Draw("sames");

  canvas->cd(4);
  setStats(ksTkFakePt, ksNewTkFakePt, 0.6, 0.65, false);
  ksTkFakePt->Draw();
  ksNewTkFakePt->Draw("sames");

  canvas->cd(5);
  setStats(ksFakeR, ksNewFakeR, -1, 0, false);
  ksFakeR->Draw();
  ksNewFakeR->Draw("sames");

  canvas->cd(6);
  setStats(ksTkFakeR, ksNewTkFakeR, 0.6, 0.65, false);
  ksTkFakeR->Draw();
  ksNewTkFakeR->Draw("sames");

  canvas->cd();
  leg2 = new TLegend(0.20, 0.64, 0.80, 0.69);
  leg2->SetTextSize(0.012);
  leg2->SetLineColor(1);
  leg2->SetLineWidth(1);
  leg2->SetLineStyle(1);
  leg2->SetFillColor(0);
  leg2->SetBorderSize(3);
  leg2->AddEntry(ksFakeEta, refLabel, "LPF");
  leg2->AddEntry(ksNewFakeEta, newLabel, "LPF");
  leg2->Draw();

  canvas->Print("K0sFake.png");
  canvas->Print("K0sFake.pdf");

  delete leg2;

  cout << "Plotting Lambdas" << endl;
  // Lambda plots
  TH1F *lamEffEta, *lamTkEffEta;
  TH1F *lamEffPt, *lamTkEffPt;
  TH1F *lamEffR, *lamTkEffR;

  TH1F *lamNewEffEta, *lamNewTkEffEta;
  TH1F *lamNewEffPt, *lamNewTkEffPt;
  TH1F *lamNewEffR, *lamNewTkEffR;

  TH1F *lamFakeEta, *lamTkFakeEta;
  TH1F *lamFakePt, *lamTkFakePt;
  TH1F *lamFakeR, *lamTkFakeR;

  TH1F *lamNewFakeEta, *lamNewTkFakeEta;
  TH1F *lamNewFakePt, *lamNewTkFakePt;
  TH1F *lamNewFakeR, *lamNewTkFakeR;

  TH1F *lamNewFakeMass, *lamNewGoodMass;
  TH1F *lamNewMassAll;
  TH1F *lamFakeMass, *lamGoodMass;
  TH1F *lamMassAll;

  // Lambda Efficiency plots from reference release
  //refdir->GetObject(hList1->At(12)->GetName(), lamEffEta);
  refdir->GetObject(hList1->FindObject("LamEffVsEta")->GetName(), lamEffEta);
  lamEffEta->GetYaxis()->SetRangeUser(0, 1.1);

  //refdir->GetObject(hList1->At(18)->GetName(), lamTkEffEta);
  refdir->GetObject(hList1->FindObject("LamTkEffVsEta")->GetName(), lamTkEffEta);
  lamTkEffEta->GetYaxis()->SetRangeUser(0, 1.1);

  //refdir->GetObject(hList1->At(13)->GetName(), lamEffPt);
  refdir->GetObject(hList1->FindObject("LamEffVsPt")->GetName(), lamEffPt);
  lamEffPt->GetYaxis()->SetRangeUser(0, 1.1);
  lamEffPt->GetYaxis()->SetTitle("#Lambda^{0} Efficiency vs p_{T}");
  lamEffPt->GetYaxis()->SetTitleSize(0.05);
  lamEffPt->GetYaxis()->SetTitleOffset(1.2);
  lamEffPt->SetTitle("");

  //refdir->GetObject(hList1->At(19)->GetName(), lamTkEffPt);
  refdir->GetObject(hList1->FindObject("LamTkEffVsPt")->GetName(), lamTkEffPt);
  lamTkEffPt->GetYaxis()->SetRangeUser(0, 1.1);
  lamTkEffPt->GetYaxis()->SetTitle("#Lambda^{0} Tracking Efficiency vs p_{T}");
  lamTkEffPt->GetYaxis()->SetTitleSize(0.05);
  lamTkEffPt->GetYaxis()->SetTitleOffset(1.2);
  lamTkEffPt->SetTitle("");

  //refdir->GetObject(hList1->At(14)->GetName(), lamEffR);
  refdir->GetObject(hList1->FindObject("LamEffVsR")->GetName(), lamEffR);
  lamEffR->GetYaxis()->SetRangeUser(0, 1.1);

  //refdir->GetObject(hList1->At(20)->GetName(), lamTkEffR);
  refdir->GetObject(hList1->FindObject("LamTkEffVsR")->GetName(), lamTkEffR);
  lamTkEffR->GetYaxis()->SetRangeUser(0, 1.1);

  // Lambda mass plots from ref release
  refdir_1->GetObject(hList1_1->FindObject("lamMassFake")->GetName(), lamFakeMass);
  lamFakeMass->SetXTitle(//"Mass of fake #Lambda, ref release");
			 refLabel);
  refdir_1->GetObject(hList1_1->FindObject("lamMassGood")->GetName(), lamGoodMass);
  lamGoodMass->SetXTitle(//"Mass of good #Lambda, ref release");
			 refLabel);
  refdir_1->GetObject(hList1_1->FindObject("lamMassAll")->GetName(), lamMassAll);
  lamMassAll->SetXTitle(//"Mass of all found #Lambda, ref release");
			refLabel);


  // Lambda efficiency plots from new release
  //newdir->GetObject(hList1->At(12)->GetName(), lamNewEffEta);
  newdir->GetObject(hList2->FindObject("LamEffVsEta")->GetName(), lamNewEffEta);
  lamNewEffEta->GetYaxis()->SetRangeUser(0, 1.1);

  //newdir->GetObject(hList1->At(18)->GetName(), lamNewTkEffEta);
  newdir->GetObject(hList2->FindObject("LamTkEffVsEta")->GetName(), lamNewTkEffEta);
  lamNewTkEffEta->GetYaxis()->SetRangeUser(0, 1.1);

  //newdir->GetObject(hList1->At(13)->GetName(), lamNewEffPt);
  newdir->GetObject(hList2->FindObject("LamEffVsPt")->GetName(), lamNewEffPt);
  lamNewEffPt->GetYaxis()->SetRangeUser(0, 1.1);
  lamNewEffPt->GetYaxis()->SetRangeUser(0, 1.1);
  lamNewEffPt->GetYaxis()->SetTitle("#Lambda^{0} Efficiency vs p_{T}");
  lamNewEffPt->GetYaxis()->SetTitleSize(0.05);
  lamNewEffPt->GetYaxis()->SetTitleOffset(1.2);
  lamNewEffPt->SetTitle("");

  //newdir->GetObject(hList1->At(19)->GetName(), lamNewTkEffPt);
  newdir->GetObject(hList1->FindObject("LamTkEffVsPt")->GetName(), lamNewTkEffPt);
  lamNewTkEffPt->GetYaxis()->SetRangeUser(0, 1.1);
  lamNewTkEffPt->GetYaxis()->SetTitle("#Lambda^{0} Tracking Efficiency vs p_{T}");
  lamNewTkEffPt->GetYaxis()->SetTitleSize(0.05);
  lamNewTkEffPt->GetYaxis()->SetTitleOffset(1.2);
  lamNewTkEffPt->SetTitle("");

  //newdir->GetObject(hList1->At(14)->GetName(), lamNewEffR);
  newdir->GetObject(hList2->FindObject("LamEffVsR")->GetName(), lamNewEffR);
  lamNewEffR->GetYaxis()->SetRangeUser(0, 1.1);

  //newdir->GetObject(hList1->At(20)->GetName(), lamNewTkEffR);
  newdir->GetObject(hList2->FindObject("LamTkEffVsR")->GetName(), lamNewTkEffR);
  lamNewTkEffR->GetYaxis()->SetRangeUser(0, 1.1);

  // Lambda mass plots from new release
  newdir_1->GetObject(hList2_1->FindObject("lamMassFake")->GetName(), lamNewFakeMass);
  lamNewFakeMass->SetXTitle(//"Mass of fake #Lambda, new release");
			    newLabel);
  newdir_1->GetObject(hList2_1->FindObject("lamMassGood")->GetName(), lamNewGoodMass);
  lamNewGoodMass->SetXTitle(//"Mass of good #Lambda, new release");
			    newLabel);
  newdir_1->GetObject(hList2_1->FindObject("lamMassAll")->GetName(), lamNewMassAll);
  lamNewMassAll->SetXTitle(//"Mass of all found #Lambda, new release");
			   newLabel);


  // Lambda fake rate plots from reference release
  //refdir->GetObject(hList1->At(15)->GetName(), lamFakeEta);
  refdir->GetObject(hList1->FindObject("LamFakeVsEta")->GetName(), lamFakeEta);
  lamFakeEta->GetYaxis()->SetRangeUser(0, 1.1);

  //refdir->GetObject(hList1->At(21)->GetName(), lamTkFakeEta);
  refdir->GetObject(hList1->FindObject("LamTkFakeVsEta")->GetName(), lamTkFakeEta);
  lamTkFakeEta->GetYaxis()->SetRangeUser(0, 1.1);

  //refdir->GetObject(hList1->At(16)->GetName(), lamFakePt);
  refdir->GetObject(hList2->FindObject("LamFakeVsPt")->GetName(), lamFakePt);
  lamFakePt->GetYaxis()->SetRangeUser(0, 1.1);
  lamFakePt->GetYaxis()->SetRangeUser(0, 1.1);
  lamFakePt->GetYaxis()->SetTitle("#Lambda^{0} Fake Rate vs p_{T}");
  lamFakePt->GetYaxis()->SetTitleSize(0.05);
  lamFakePt->GetYaxis()->SetTitleOffset(1.2);
  lamFakePt->SetTitle("");

  //refdir->GetObject(hList1->At(22)->GetName(), lamTkFakePt);
  refdir->GetObject(hList1->FindObject("LamTkFakeVsPt")->GetName(), lamTkFakePt);
  lamTkFakePt->GetYaxis()->SetRangeUser(0, 1.1);
  lamTkFakePt->GetYaxis()->SetTitle("#Lambda^{0} Tracking Fake Rate vs p_{T}");
  lamTkFakePt->GetYaxis()->SetTitleSize(0.05);
  lamTkFakePt->GetYaxis()->SetTitleOffset(1.2);
  lamTkFakePt->SetTitle("");

  //refdir->GetObject(hList1->At(17)->GetName(), lamFakeR);
  refdir->GetObject(hList1->FindObject("LamFakeVsR")->GetName(), lamFakeR);
  lamFakeR->GetYaxis()->SetRangeUser(0, 1.1);

  //refdir->GetObject(hList1->At(23)->GetName(), lamTkFakeR);
  refdir->GetObject(hList1->FindObject("LamTkFakeVsR")->GetName(), lamTkFakeR);
  lamTkFakeR->GetYaxis()->SetRangeUser(0, 1.1);



  // Lambda plots from new release we're testing
  //newdir->GetObject(hList1->At(15)->GetName(), lamNewFakeEta);
  newdir->GetObject(hList2->FindObject("LamFakeVsEta")->GetName(), lamNewFakeEta);
  lamNewFakeEta->GetYaxis()->SetRangeUser(0, 1.1);

  //newdir->GetObject(hList1->At(21)->GetName(), lamNewTkFakeEta);
  newdir->GetObject(hList2->FindObject("LamTkFakeVsEta")->GetName(), lamNewTkFakeEta);
  lamNewTkFakeEta->GetYaxis()->SetRangeUser(0, 1.1);

  //newdir->GetObject(hList1->At(16)->GetName(), lamNewFakePt);
  newdir->GetObject(hList2->FindObject("LamFakeVsPt")->GetName(), lamNewFakePt);
  lamNewFakePt->GetYaxis()->SetRangeUser(0, 1.1);
  lamNewFakePt->SetTitle("");

  //newdir->GetObject(hList1->At(22)->GetName(), lamNewTkFakePt);
  newdir->GetObject(hList2->FindObject("LamTkFakeVsPt")->GetName(), lamNewTkFakePt);
  lamNewTkFakePt->GetYaxis()->SetRangeUser(0, 1.1);
  lamNewTkFakePt->SetTitle("");

  //newdir->GetObject(hList1->At(17)->GetName(), lamNewFakeR);
  newdir->GetObject(hList2->FindObject("LamFakeVsR")->GetName(), lamNewFakeR);
  lamNewFakeR->GetYaxis()->SetRangeUser(0, 1.1);

  //newdir->GetObject(hList1->At(23)->GetName(), lamNewTkFakeR);
  newdir->GetObject(hList2->FindObject("LamTkFakeVsR")->GetName(), lamNewTkFakeR);
  lamNewTkFakeR->GetYaxis()->SetRangeUser(0, 1.1);


  canvas = new TCanvas("Lambdas", "#Lambda^{0} Efficiency", 
		       1000, 1400);

  lamEffEta->SetMarkerStyle(20);
  lamEffEta->SetMarkerColor(2);
  lamEffEta->SetMarkerSize(0.5);
  lamEffEta->SetLineColor(2);
  lamNewEffEta->SetMarkerStyle(20);
  lamNewEffEta->SetMarkerColor(4);
  lamNewEffEta->SetMarkerSize(0.5);
  lamNewEffEta->SetLineColor(4);

  lamTkEffEta->SetMarkerStyle(20);
  lamTkEffEta->SetMarkerColor(2);
  lamTkEffEta->SetMarkerSize(0.5);
  lamTkEffEta->SetLineColor(2);
  lamNewTkEffEta->SetMarkerStyle(20);
  lamNewTkEffEta->SetMarkerColor(4);
  lamNewTkEffEta->SetMarkerSize(0.5);
  lamNewTkEffEta->SetLineColor(4);

  lamEffPt->SetMarkerStyle(20);
  lamEffPt->SetMarkerColor(2);
  lamEffPt->SetMarkerSize(0.5);
  lamEffPt->SetLineColor(2);
  lamNewEffPt->SetMarkerStyle(20);
  lamNewEffPt->SetMarkerColor(4);
  lamNewEffPt->SetMarkerSize(0.5);
  lamNewEffPt->SetLineColor(4);

  lamTkEffPt->SetMarkerStyle(20);
  lamTkEffPt->SetMarkerColor(2);
  lamTkEffPt->SetMarkerSize(0.5);
  lamTkEffPt->SetLineColor(2);
  lamNewTkEffPt->SetMarkerStyle(20);
  lamNewTkEffPt->SetMarkerColor(4);
  lamNewTkEffPt->SetMarkerSize(0.5);
  lamNewTkEffPt->SetLineColor(4);

  lamEffR->SetMarkerStyle(20);
  lamEffR->SetMarkerColor(2);
  lamEffR->SetMarkerSize(0.5);
  lamEffR->SetLineColor(2);
  lamNewEffR->SetMarkerStyle(20);
  lamNewEffR->SetMarkerColor(4);
  lamNewEffR->SetMarkerSize(0.5);
  lamNewEffR->SetLineColor(4);

  lamTkEffR->SetMarkerStyle(20);
  lamTkEffR->SetMarkerColor(2);
  lamTkEffR->SetMarkerSize(0.5);
  lamTkEffR->SetLineColor(2);
  lamNewTkEffR->SetMarkerStyle(20);
  lamNewTkEffR->SetMarkerColor(4);
  lamNewTkEffR->SetMarkerSize(0.5);
  lamNewTkEffR->SetLineColor(4);

  canvas->Divide(2,3);

  canvas->cd(1);
  setStats(lamEffEta, lamNewEffEta, -1, 0, false);
  lamEffEta->Draw();
  lamNewEffEta->Draw("sames");

  canvas->cd(2);
  setStats(lamTkEffEta, lamNewTkEffEta, -1, 0, false);
  lamTkEffEta->Draw();
  lamNewTkEffEta->Draw("sames");

  canvas->cd(3);
  setStats(lamEffPt, lamNewEffPt, -1, 0, false);
  lamEffPt->Draw();
  lamNewEffPt->Draw("sames");

  canvas->cd(4);
  setStats(lamTkEffPt, lamNewTkEffPt, 0.6, 0.65, false);
  lamTkEffPt->Draw();
  lamNewTkEffPt->Draw("sames");

  canvas->cd(5);
  setStats(lamEffR, lamNewEffR, -1, 0, false);
  lamEffR->Draw();
  lamNewEffR->Draw("sames");

  canvas->cd(6);
  setStats(lamTkEffR, lamNewTkEffR, 0.6, 0.65, false);
  lamTkEffR->Draw();
  lamNewTkEffR->Draw("sames");

  canvas->cd();
  leg3 = new TLegend(0.20, 0.64, 0.80, 0.69);
  leg3->SetTextSize(0.012);
  leg3->SetLineColor(1);
  leg3->SetLineWidth(1);
  leg3->SetLineStyle(1);
  leg3->SetFillColor(0);
  leg3->SetBorderSize(3);
  leg3->AddEntry(lamEffEta, refLabel, "LPF");
  leg3->AddEntry(lamNewEffEta, newLabel, "LPF");
  leg3->Draw();

  canvas->Print("LamEff.png");
  canvas->Print("LamEff.pdf");

  delete leg3;
  //delete canvas;

  lammcanvas = new TCanvas("LamMass", "#Lambda^{0} mass plots",
			   1000, 1400);

  lamFakeMass->SetLineColor(2);
  lamFakeMass->SetMarkerStyle(20);
  lamFakeMass->SetMarkerColor(2);
  lamNewFakeMass->SetLineColor(4);
  lamNewFakeMass->SetMarkerStyle(20);
  lamNewFakeMass->SetMarkerColor(4);

  lamGoodMass->SetLineColor(2);
  lamGoodMass->SetMarkerStyle(20);
  lamGoodMass->SetMarkerColor(2);
  lamNewGoodMass->SetLineColor(4);
  lamNewGoodMass->SetMarkerStyle(20);
  lamNewGoodMass->SetMarkerColor(4);

  lamMassAll->SetLineColor(2);
  lamMassAll->SetMarkerStyle(20);
  lamMassAll->SetMarkerColor(2);
  lamNewMassAll->SetLineColor(4);
  lamNewMassAll->SetMarkerStyle(20);
  lamNewMassAll->SetMarkerColor(4);

  lammcanvas->Divide(2,3);
  lammcanvas->cd(1);
  lamMassAll->Draw();

  lammcanvas->cd(2);
  lamNewMassAll->Draw();

  lammcanvas->cd(3);
  lamGoodMass->Draw();

  lammcanvas->cd(4);
  lamNewGoodMass->Draw();

  lammcanvas->cd(5);
  lamFakeMass->Draw();

  lammcanvas->cd(6);
  lamNewFakeMass->Draw();

  lammcanvas->Print("LamMass.png");
  lammcanvas->Print("LamMass.pdf");

  canvas = new TCanvas("Lambdas", "#Lambda^{0} Fake Rate", 
		       1000, 1400);

  lamFakeEta->SetMarkerStyle(20);
  lamFakeEta->SetMarkerColor(2);
  lamFakeEta->SetMarkerSize(0.5);
  lamFakeEta->SetLineColor(2);
  lamNewFakeEta->SetMarkerStyle(20);
  lamNewFakeEta->SetMarkerColor(4);
  lamNewFakeEta->SetMarkerSize(0.5);
  lamNewFakeEta->SetLineColor(4);

  lamTkFakeEta->SetMarkerStyle(20);
  lamTkFakeEta->SetMarkerColor(2);
  lamTkFakeEta->SetMarkerSize(0.5);
  lamTkFakeEta->SetLineColor(2);
  lamNewTkFakeEta->SetMarkerStyle(20);
  lamNewTkFakeEta->SetMarkerColor(4);
  lamNewTkFakeEta->SetMarkerSize(0.5);
  lamNewTkFakeEta->SetLineColor(4);

  lamFakePt->SetMarkerStyle(20);
  lamFakePt->SetMarkerColor(2);
  lamFakePt->SetMarkerSize(0.5);
  lamFakePt->SetLineColor(2);
  lamNewFakePt->SetMarkerStyle(20);
  lamNewFakePt->SetMarkerColor(4);
  lamNewFakePt->SetMarkerSize(0.5);
  lamNewFakePt->SetLineColor(4);

  lamTkFakePt->SetMarkerStyle(20);
  lamTkFakePt->SetMarkerColor(2);
  lamTkFakePt->SetMarkerSize(0.5);
  lamTkFakePt->SetLineColor(2);
  lamNewTkFakePt->SetMarkerStyle(20);
  lamNewTkFakePt->SetMarkerColor(4);
  lamNewTkFakePt->SetMarkerSize(0.5);
  lamNewTkFakePt->SetLineColor(4);

  lamFakeR->SetMarkerStyle(20);
  lamFakeR->SetMarkerColor(2);
  lamFakeR->SetMarkerSize(0.5);
  lamFakeR->SetLineColor(2);
  lamNewFakeR->SetMarkerStyle(20);
  lamNewFakeR->SetMarkerColor(4);
  lamNewFakeR->SetMarkerSize(0.5);
  lamNewFakeR->SetLineColor(4);

  lamTkFakeR->SetMarkerStyle(20);
  lamTkFakeR->SetMarkerColor(2);
  lamTkFakeR->SetMarkerSize(0.5);
  lamTkFakeR->SetLineColor(2);
  lamNewTkFakeR->SetMarkerStyle(20);
  lamNewTkFakeR->SetMarkerColor(4);
  lamNewTkFakeR->SetMarkerSize(0.5);
  lamNewTkFakeR->SetLineColor(4);

  canvas->Divide(2,3);

  canvas->cd(1);
  setStats(lamFakeEta, lamNewFakeEta, -1, 0, false);
  lamFakeEta->Draw();
  lamNewFakeEta->Draw("sames");

  canvas->cd(2);
  setStats(lamTkFakeEta, lamNewTkFakeEta, -1, 0, false);
  lamTkFakeEta->Draw();
  lamNewTkFakeEta->Draw("sames");

  canvas->cd(3);
  setStats(lamFakePt, lamNewFakePt, -1, 0, false);
  lamFakePt->Draw();
  lamNewFakePt->Draw("sames");

  canvas->cd(4);
  setStats(lamTkFakePt, lamNewTkFakePt, 0.6, 0.65, false);
  lamTkFakePt->Draw();
  lamNewTkFakePt->Draw("sames");

  canvas->cd(5);
  setStats(lamFakeR, lamNewFakeR, -1, 0, false);
  lamFakeR->Draw();
  lamNewFakeR->Draw("sames");

  canvas->cd(6);
  setStats(lamTkFakeR, lamNewTkFakeR, 0.6, 0.65, false);
  lamTkFakeR->Draw();
  lamNewTkFakeR->Draw("sames");

  canvas->cd();
  leg4 = new TLegend(0.20, 0.64, 0.80, 0.69);
  leg4->SetTextSize(0.012);
  leg4->SetLineColor(1);
  leg4->SetLineWidth(1);
  leg4->SetLineStyle(1);
  leg4->SetFillColor(0);
  leg4->SetBorderSize(3);
  leg4->AddEntry(lamFakeEta, refLabel, "LPF");
  leg4->AddEntry(lamNewFakeEta, newLabel, "LPF");
  leg4->Draw();

  canvas->Print("LamFake.png");
  canvas->Print("LamFake.pdf");

  delete leg4;

}

// Need to fix this to work with 2 plots on each pad
void setStats(TH1* s, TH1* r, 
	      double startingY, 
	      double startingX = .1, 
	      bool fit) {
  if(startingY < 0) {
    s->SetStats(0);
    r->SetStats(0);
  }
  else {
    if(fit) {
      s->Fit("gaus");
      TF1* f1 = (TF1*) s->GetListOfFunctions()->FindObject("gaus");
      f1->SetLineColor(2);
      f1->SetLineWidth(1);
    }
    s->Draw();
    gPad->Update();
    TPaveStats* st1 = (TPaveStats*) s->GetListOfFunctions()->FindObject("stats");
    //TPaveText* tt1 = (TPaveText*) s->GetListOfFunctions()->FindObject("title");
    if (fit) {
      st1->SetOptFit(0010);
      st1->SetOptStat(1001);
    }
    st1->SetX1NDC(startingX);
    st1->SetX2NDC(startingX + 0.30);
    st1->SetY1NDC(startingY + 0.20);
    st1->SetY2NDC(startingY + 0.35);
    st1->SetTextColor(2);
    if(fit) {
      r->Fit("gaus");
      TF1* f2 = (TF1*) r->GetListOfFunctions()->FindObject("gaus");
      f2->SetLineColor(4);
      f2->SetLineWidth(1);
    }
    r->Draw();
    gPad->Update();
    TPaveStats* st2 = (TPaveStats*) r->GetListOfFunctions()->FindObject("stats");
    //    TPaveStats* st2 = (TPaveStats*) r->GetListOfFunctions()->FindObject("stats");
    if(fit) {
      st2->SetOptFit(0010);
      st2->SetOptStats(1001);
    }
    st2->SetX1NDC(startingX);
    st2->SetX2NDC(startingX + 0.30);
    st2->SetY1NDC(startingY);
    st2->SetY2NDC(startingY + 0.15);
    st2->SetTextColor(4);
  }
}

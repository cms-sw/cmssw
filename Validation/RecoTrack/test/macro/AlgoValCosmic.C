void AlgoValCosmic(char* newfile="NEW_FILE")
{
  gROOT ->Reset();
  gROOT ->SetBatch();
  
  //=========  settings ====================
  gROOT->SetStyle("Plain");
  gStyle->SetPadGridX(kTRUE);
  gStyle->SetPadGridY(kTRUE);
  gStyle->SetPadRightMargin(0.07);
  gStyle->SetPadLeftMargin(0.13);
  gStyle->SetPadBottomMargin(0.13);
  
  char* cosLabel("Cosmic TF");
  char* ctfLabel("CTF");
  char* rsLabel("RS");
 
  char* cosLabel2("Cosmic TF NEW_RELEASE NEWSELECTION");
  char* ctfLabel2("CTF NEW_RELEASE NEWSELECTION");
  char* rsLabel2("RS NEW_RELEASE NEWSELECTION ");

  
  TText* te = new TText();
  TCanvas *canvas;

  TH1F *ctfh1,*cosh1,*rsh1;
  TH1F *ctfh2,*cosh2,*rsh2;
  TH1F *ctfh3,*cosh3,*rsh3;
  TH1F *ctfh4,*cosh4,*rsh4;
  TH1F *ctfh5,*cosh5,*rsh5;
  TH1F *ctfh6,*cosh6,*rsh6;

  bool ifploteps = IFPLOTEPS;

  TFile * newFile = new TFile(newfile);
  TDirectory *dir=gDirectory;
  
  if(newFile->GetDirectory("DQMData/Run 1/RecoTrackV")) newFile->cd("DQMData/Run 1/RecoTrackV/Run summary/Track");
  else if(newFile->cd("DQMData/RecoTrackV/Track"))newFile->cd("DQMData/RecoTrackV/Track");
  else if(newFile->GetDirectory("DQMData/Run 1/Tracking")) newFile->cd("DQMData/Run 1/Tracking/Run summary/Track");
  else if(newFile->cd("DQMData/Tracking/Track"))newFile->cd("DQMData/Tracking/Track");
  dir=gDirectory;
  
  TList *sl= dir->GetListOfKeys();
  TString collnamectf=sl->At(0)->GetName(); 
  TString collnamecos=sl->At(1)->GetName(); 
  TString collnamers=sl->At(2)->GetName(); 
  
  //efficiency vs eta
  dir->GetObject(collnamecos+"/effic",cosh1);
  dir->GetObject(collnamectf+"/effic",ctfh1);
  dir->GetObject(collnamers+"/effic",rsh1);

  ctfh1->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
  cosh1->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
  rsh1->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);

  ctfh1->GetYaxis()->SetTitle("efficiency");
  cosh1->GetYaxis()->SetTitle("efficiency");
  rsh1->GetYaxis()->SetTitle("efficiency");
  ctfh1->GetXaxis()->SetTitle("#eta");
  cosh1->GetXaxis()->SetTitle("#eta");
  rsh1->GetXaxis()->SetTitle("#eta");
  

  canvas = new TCanvas("Trackseffvsetacut1","Tracks: efficiency",400,350);
 
  ploteff(canvas, cosh1,ctfh1,rsh1, te,"UU",-0.55,0.65);
  
  canvas->cd();
  l = new TLegend(0.35,0.2,0.7,0.4);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh1,cosLabel,"LPF");
  l->AddEntry(ctfh1,ctfLabel,"LPF");
  l->AddEntry(rsh1,rsLabel,"LPF");
  l->Draw();
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/trackeffvseta.eps");   
  delete l;
  
  //fakerate vs eta
  dir->GetObject(collnamecos+"/fakerate",cosh2);
  dir->GetObject(collnamectf+"/fakerate",ctfh2);
  dir->GetObject(collnamers+"/fakerate",rsh2);

  ctfh2->GetYaxis()->SetRangeUser(0,MAXFAKE);
  cosh2->GetYaxis()->SetRangeUser(0,MAXFAKE);
  rsh2->GetYaxis()->SetRangeUser(0,MAXFAKE);

  ctfh2->GetYaxis()->SetTitle("fakerate");
  cosh2->GetYaxis()->SetTitle("fakerate");
  rsh2->GetYaxis()->SetTitle("fakerate");
  ctfh2->GetXaxis()->SetTitle("#eta");
  cosh2->GetXaxis()->SetTitle("#eta");
  rsh2->GetXaxis()->SetTitle("#eta");

  canvas = new TCanvas("Tracksfakeratevsetacut1","Tracks: fakerate",400,350);
  ploteff(canvas, cosh2,ctfh2,rsh2, te,"UU",-0.55,0.65, false,false,false);
  canvas->cd();  
  l = new TLegend(0.55,0.80,0.95,0.99);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh2,cosLabel,"LPF");
  l->AddEntry(ctfh2,ctfLabel,"LPF");
  l->AddEntry(rsh2,rsLabel,"LPF");
  l->Draw();
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/trackfakeratevseta.eps");   
  delete l;

  //efficiency vs pT
  dir->GetObject(collnamecos+"/efficPt",cosh3);
  dir->GetObject(collnamectf+"/efficPt",ctfh3);
  dir->GetObject(collnamers+"/efficPt",rsh3);

  ctfh3->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
  cosh3->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
  rsh3->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);

  ctfh3->GetYaxis()->SetTitle("efficiency");
  cosh3->GetYaxis()->SetTitle("efficiency");
  rsh3->GetYaxis()->SetTitle("efficiency");
  ctfh3->GetXaxis()->SetTitle("p_{T}");
  cosh3->GetXaxis()->SetTitle("p_{T}");
  rsh3->GetXaxis()->SetTitle("p_{T}");
  
  canvas = new TCanvas("Trackseffvsptcut1","Tracks: efficiency",400,350);
  ploteff(canvas, cosh3,ctfh3,rsh3, te,"UU",-0.55,0.65);
  
  canvas->cd();
  l = new TLegend(0.35,0.2,0.7,0.4);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh1,cosLabel,"LPF");
  l->AddEntry(ctfh1,ctfLabel,"LPF");
  l->AddEntry(rsh1,rsLabel,"LPF");
  l->Draw();
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/trackeffvspt.eps");   

  //fakerate vs pT
  dir->GetObject(collnamecos+"/fakeratePt",cosh4);
  dir->GetObject(collnamectf+"/fakeratePt",ctfh4);
  dir->GetObject(collnamers+"/fakeratePt",rsh4);

  ctfh4->GetYaxis()->SetRangeUser(0,MAXFAKE);
  cosh4->GetYaxis()->SetRangeUser(0,MAXFAKE);
  rsh4->GetYaxis()->SetRangeUser(0,MAXFAKE);

  ctfh4->GetYaxis()->SetTitle("fakerate");
  cosh4->GetYaxis()->SetTitle("fakerate");
  rsh4->GetYaxis()->SetTitle("fakerate");
  ctfh4->GetXaxis()->SetTitle("p_{T}");
  cosh4->GetXaxis()->SetTitle("p_{T}");
  rsh4->GetXaxis()->SetTitle("p_{T}");
  
  canvas = new TCanvas("Tracksfakeratevsptcut1","Tracks: fakerate",400,350);
  ploteff(canvas, cosh4,ctfh4,rsh4, te,"UU",-1,0,false,false,false);
  
  canvas->cd();
  l = new TLegend(0.55,0.80,0.95,0.99);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh4,cosLabel,"LPF");
  l->AddEntry(ctfh4,ctfLabel,"LPF");
  l->AddEntry(rsh4,rsLabel,"LPF");
  l->Draw();
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/trackfakeratevspt.eps");   

  //efficiency vs nHit
  dir->GetObject(collnamecos+"/effic_vs_hit",cosh5);
  dir->GetObject(collnamectf+"/effic_vs_hit",ctfh5);
  dir->GetObject(collnamers+"/effic_vs_hit",rsh5);
 
  ctfh5->GetYaxis()->SetTitle("efficiency");
  cosh5->GetYaxis()->SetTitle("efficiency");
  rsh5->GetYaxis()->SetTitle("efficiency");  

  ctfh5->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
  cosh5->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
  rsh5->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
  ctfh5->GetXaxis()->SetTitle("nHit");
  cosh5->GetXaxis()->SetTitle("nHit");
  rsh5->GetXaxis()->SetTitle("nHit");
  
  canvas = new TCanvas("TrackseffvsnHitcut1","Tracks: efficiency",400,350);
  ploteff(canvas, cosh5,ctfh5,rsh5, te,"UU",-0.55,0.65);

  canvas->cd();
  l = new TLegend(0.35,0.2,0.7,0.4);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh1,cosLabel,"LPF");
  l->AddEntry(ctfh1,ctfLabel,"LPF");
  l->AddEntry(rsh1,rsLabel,"LPF");
  l->Draw();
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/trackeffvshit.eps");   

  //fakerate vs nHit
  dir->GetObject(collnamecos+"/fakerate_vs_hit",cosh6);
  dir->GetObject(collnamectf+"/fakerate_vs_hit",ctfh6);
  dir->GetObject(collnamers+"/fakerate_vs_hit",rsh6);
 
  ctfh6->GetYaxis()->SetTitle("fakerate");
  cosh6->GetYaxis()->SetTitle("fakerate");
  rsh6->GetYaxis()->SetTitle("fakerate");  
  ctfh6->GetYaxis()->SetRangeUser(0,MAXFAKE);
  cosh6->GetYaxis()->SetRangeUser(0,MAXFAKE);
  rsh6->GetYaxis()->SetRangeUser(0,MAXFAKE);
  ctfh6->GetXaxis()->SetTitle("nHit");
  cosh6->GetXaxis()->SetTitle("nHit");
  rsh6->GetXaxis()->SetTitle("nHit");
  
  canvas = new TCanvas("TracksfakeratevsnHitcut1","Tracks: fakerate",400,350);
  ploteff(canvas, cosh6,ctfh6,rsh6, te,"UU",-1,0,false,false,false);
  canvas->cd();

  l = new TLegend(0.55,0.80,0.95,0.99);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh2,cosLabel,"LPF");
  l->AddEntry(ctfh2,ctfLabel,"LPF");
  l->AddEntry(rsh2,rsLabel,"LPF");
  l->Draw();
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/trackfakeratevshit.eps");   

  //plotBuilding
  canvas = new TCanvas("Tracks1","Tracks: efficiency & fakerate",1000,1400);
  
  plotBuilding(canvas,
	       cosh1,ctfh1,rsh1,
	       cosh2,ctfh2,rsh2,
	       cosh3,ctfh3,rsh3,
	       cosh4,ctfh4,rsh4,
	       cosh5,ctfh5,rsh5,
	       cosh6,ctfh6,rsh6,
	       te,"UU",-1,0.1,false,false,false);
  
  canvas->cd();
  l = new TLegend(0.20,0.635,0.80,0.685);
  l->SetTextSize(0.016);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh1,cosLabel2,"LPF");
  l->AddEntry(ctfh1,ctfLabel2,"LPF");
  l->AddEntry(rsh1,rsLabel2,"LPF");
  l->Draw(); 
  canvas->Print("cosmic_algoval_plots/building.pdf"); 
  delete l;
  
  //efficiency vs phi
  dir->GetObject(collnamecos+"/effic_vs_phi",cosh1);
  dir->GetObject(collnamectf+"/effic_vs_phi",ctfh1);
  dir->GetObject(collnamers+"/effic_vs_phi",rsh1);
  
  ctfh1->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
  cosh1->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
  rsh1->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);

  ctfh1->GetYaxis()->SetTitle("efficiency");
  cosh1->GetYaxis()->SetTitle("efficiency");
  rsh1->GetYaxis()->SetTitle("efficiency");
  ctfh1->GetXaxis()->SetTitle("#phi");
  cosh1->GetXaxis()->SetTitle("#phi");
  rsh1->GetXaxis()->SetTitle("#phi");
  

  canvas = new TCanvas("Trackseffvsphicut1","Tracks: efficiency",400,350);
 
  ploteff(canvas, cosh1,ctfh1,rsh1, te,"UU",-0.55,0.65);
  
  canvas->cd();
  l = new TLegend(0.35,0.2,0.7,0.4);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh1,cosLabel,"LPF");
  l->AddEntry(ctfh1,ctfLabel,"LPF");
  l->AddEntry(rsh1,rsLabel,"LPF");
  l->Draw();
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/trackeffvsphi.eps");   
  delete l;
  
  //fakerate vs phi
  dir->GetObject(collnamecos+"/fakerate_vs_phi",cosh2);
  dir->GetObject(collnamectf+"/fakerate_vs_phi",ctfh2);
  dir->GetObject(collnamers+"/fakerate_vs_phi",rsh2);

  ctfh2->GetYaxis()->SetRangeUser(0,MAXFAKE);
  cosh2->GetYaxis()->SetRangeUser(0,MAXFAKE);
  rsh2->GetYaxis()->SetRangeUser(0,MAXFAKE);

  ctfh2->GetYaxis()->SetTitle("fakerate");
  cosh2->GetYaxis()->SetTitle("fakerate");
  rsh2->GetYaxis()->SetTitle("fakerate");
  ctfh2->GetXaxis()->SetTitle("#phi");
  cosh2->GetXaxis()->SetTitle("#phi");
  rsh2->GetXaxis()->SetTitle("#phi");

  canvas = new TCanvas("Tracksfakeratevsphicut1","Tracks: fakerate",400,350);
  ploteff(canvas, cosh2,ctfh2,rsh2, te,"UU",-0.55,0.65, false,false,false);
  canvas->cd();  
  l = new TLegend(0.55,0.80,0.95,0.99);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh2,cosLabel,"LPF");
  l->AddEntry(ctfh2,ctfLabel,"LPF");
  l->AddEntry(rsh2,rsLabel,"LPF");
  l->Draw();
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/trackfakeratevsphi.eps");   
  delete l;

  //efficiency vs dxy
  dir->GetObject(collnamecos+"/effic_vs_dxy",cosh3);
  dir->GetObject(collnamectf+"/effic_vs_dxy",ctfh3);
  dir->GetObject(collnamers+"/effic_vs_dxy",rsh3);
  
  ctfh3->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
  cosh3->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
  rsh3->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);

  ctfh3->GetYaxis()->SetTitle("efficiency");
  cosh3->GetYaxis()->SetTitle("efficiency");
  rsh3->GetYaxis()->SetTitle("efficiency");
  ctfh3->GetXaxis()->SetTitle("dxy");
  cosh3->GetXaxis()->SetTitle("dxy");
  rsh3->GetXaxis()->SetTitle("dxy");

  canvas = new TCanvas("Trackseffvsdxycut1","Tracks: efficiency",400,350);
  ploteff(canvas, cosh3,ctfh3,rsh3, te,"UU",-0.55,0.65);


  canvas->cd();
  l = new TLegend(0.35,0.2,0.7,0.4);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh2,cosLabel,"LPF");
  l->AddEntry(ctfh2,ctfLabel,"LPF");
  l->AddEntry(rsh2,rsLabel,"LPF");
  l->Draw(); 
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/trackeffvsdxy.eps");   
  
 //fakerate vs dxy
  dir->GetObject(collnamecos+"/fakerate_vs_dxy",cosh4);
  dir->GetObject(collnamectf+"/fakerate_vs_dxy",ctfh4);
  dir->GetObject(collnamers+"/fakerate_vs_dxy",rsh4);
  
  ctfh4->GetYaxis()->SetRangeUser(0,MAXFAKE);
  cosh4->GetYaxis()->SetRangeUser(0,MAXFAKE);
  rsh4->GetYaxis()->SetRangeUser(0,MAXFAKE);

  ctfh4->GetYaxis()->SetTitle("fakerate");
  cosh4->GetYaxis()->SetTitle("fakerate");
  rsh4->GetYaxis()->SetTitle("fakerate");
  ctfh4->GetXaxis()->SetTitle("dxy");
  cosh4->GetXaxis()->SetTitle("dxy");
  rsh4->GetXaxis()->SetTitle("dxy");

  canvas = new TCanvas("Tracksfakeratevsdxycut1","Tracks: fakerate",400,350);
  ploteff(canvas, cosh4,ctfh4,rsh4, te,"UU",-1,0,false,false,false);
  canvas->cd();
  l = new TLegend(0.55,0.80,0.95,0.99);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh4,cosLabel,"LPF");
  l->AddEntry(ctfh4,ctfLabel,"LPF");
  l->AddEntry(rsh4,rsLabel,"LPF");
  l->Draw(); 
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/trackfakeratevsdxy.eps");   

 
  //efficiency vs dz
  dir->GetObject(collnamecos+"/effic_vs_dz",cosh5);
  dir->GetObject(collnamectf+"/effic_vs_dz",ctfh5);
  dir->GetObject(collnamers+"/effic_vs_dz",rsh5);

  ctfh5->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
  cosh5->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
  rsh5->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);

  ctfh5->GetYaxis()->SetTitle("efficiency");
  cosh5->GetYaxis()->SetTitle("efficiency");
  rsh5->GetYaxis()->SetTitle("efficiency");
  ctfh5->GetXaxis()->SetTitle("dz");
  cosh5->GetXaxis()->SetTitle("dz");
  rsh5->GetXaxis()->SetTitle("dz");


  canvas = new TCanvas("Trackseffvsdzcut1","Tracks: efficiency",400,350);
  ploteff(canvas, cosh5,ctfh5,rsh5, te,"UU",-0.55,0.65);

  canvas->cd();
  l = new TLegend(0.35,0.2,0.7,0.4);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh5,cosLabel,"LPF");
  l->AddEntry(ctfh5,ctfLabel,"LPF");
  l->AddEntry(rsh5,rsLabel,"LPF");
  l->Draw();
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/trackeffvsdz.eps");  
    
  //fakerate vs dz
  dir->GetObject(collnamecos+"/fakerate_vs_dz",cosh6);
  dir->GetObject(collnamectf+"/fakerate_vs_dz",ctfh6);
  dir->GetObject(collnamers+"/fakerate_vs_dz",rsh6);

  ctfh6->GetYaxis()->SetRangeUser(0,MAXFAKE);
  cosh6->GetYaxis()->SetRangeUser(0,MAXFAKE);
  rsh6->GetYaxis()->SetRangeUser(0,MAXFAKE);

  ctfh6->GetYaxis()->SetTitle("fakerate");
  cosh6->GetYaxis()->SetTitle("fakerate");
  rsh6->GetYaxis()->SetTitle("fakerate");
  ctfh6->GetXaxis()->SetTitle("dz");
  cosh6->GetXaxis()->SetTitle("dz");
  rsh6->GetXaxis()->SetTitle("dz");

  canvas = new TCanvas("Tracksfakeratevsdzcut1","Tracks: fakerate",400,350);
  ploteff(canvas,cosh6,ctfh6,rsh6, te,"UU",-1,0,false,false,false);
  canvas->cd();
  l = new TLegend(0.55,0.80,0.95,0.99);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh6,cosLabel,"LPF");
  l->AddEntry(ctfh6,ctfLabel,"LPF");
  l->AddEntry(rsh6,rsLabel,"LPF");
  l->Draw(); 
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/trackfakeratevsdz.eps");  
  
  //plotBuilding
  canvas = new TCanvas("Tracks2","Tracks: efficiency & fakerate",1000,1400);
  plotBuilding(canvas,
	       cosh1,ctfh1,rsh1,
	       cosh2,ctfh2,rsh2,
	       cosh3,ctfh3,rsh3,
	       cosh4,ctfh4,rsh4,
	       cosh5,ctfh5,rsh5,
	       cosh6,ctfh6,rsh6,
	       te,"UU",-1,0.1,false,false,false);
  
  canvas->cd();   
  l = new TLegend(0.20,0.635,0.80,0.685);
  l->SetTextSize(0.016);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh1,cosLabel2,"LPF");
  l->AddEntry(ctfh1,ctfLabel2,"LPF");
  l->AddEntry(rsh1,rsLabel2,"LPF");
  l->Draw(); 
  canvas->Print("cosmic_algoval_plots/building2.pdf"); 
  delete l;

  //ptPull
  dir->GetObject(collnamecos+"/pullPt",cosh1);
  dir->GetObject(collnamectf+"/pullPt",ctfh1);
  dir->GetObject(collnamers+"/pullPt",rsh1);
  
  canvas = new TCanvas("ptpull","Tracks: ptpull",400,350);
  plotPull(canvas, cosh1,ctfh1,rsh1, te,"UU",0.53,0.65,true);
  canvas->cd();
  l = new TLegend(0.15,0.7,0.42,0.88);
  l->SetTextSize(0.035);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh1,cosLabel,"LPF");
  l->AddEntry(ctfh1,ctfLabel,"LPF");
  l->AddEntry(rsh1,rsLabel,"LPF");
  l->Draw(); 
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/ptpull.eps");   
  
  //QoverpPull
  dir->GetObject(collnamecos+"/pullQoverp",cosh2);
  dir->GetObject(collnamectf+"/pullQoverp",ctfh2);
  dir->GetObject(collnamers+"/pullQoverp",rsh2);
  canvas = new TCanvas("Qoverppull","Tracks: Qoverp pull",400,350);
  plotPull(canvas, cosh2,ctfh2,rsh2, te,"UU",0.53,0.65,true);
  canvas->cd();
  l = new TLegend(0.15,0.7,0.42,0.88);
  l->SetTextSize(0.035);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh2,cosLabel,"LPF");
  l->AddEntry(ctfh2,ctfLabel,"LPF");
  l->AddEntry(rsh2,rsLabel,"LPF");
  l->Draw();
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/Qoverppull.eps");   
  
  //PhiPull
  dir->GetObject(collnamecos+"/pullPhi",cosh3);
  dir->GetObject(collnamectf+"/pullPhi",ctfh3);
  dir->GetObject(collnamers+"/pullPhi",rsh3);
  canvas = new TCanvas("Phipull","Tracks: Phi pull",400,350);
  plotPull(canvas, cosh3,ctfh3,rsh3, te,"UU",0.53,0.65,true);
  canvas->cd();
  l = new TLegend(0.15,0.7,0.42,0.88);
  l->SetTextSize(0.035);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh3,cosLabel,"LPF");
  l->AddEntry(ctfh3,ctfLabel,"LPF");
  l->AddEntry(rsh3,rsLabel,"LPF");
  l->Draw(); 
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/Phipull.eps");   

  //ThetaPull
  dir->GetObject(collnamecos+"/pullTheta",cosh4);
  dir->GetObject(collnamectf+"/pullTheta",ctfh4);
  dir->GetObject(collnamers+"/pullTheta",rsh4);
  canvas = new TCanvas("Thetapull","Tracks: Theta pull",400,350);
  plotPull(canvas, cosh4,ctfh4,rsh4, te,"UU",0.53,0.65,true);
  canvas->cd();
  l = new TLegend(0.15,0.7,0.42,0.88);
  l->SetTextSize(0.035);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh4,cosLabel,"LPF");
  l->AddEntry(ctfh4,ctfLabel,"LPF");
  l->AddEntry(rsh4,rsLabel,"LPF");
  l->Draw(); 
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/Thetapull.eps");   

  //DxyPull
  dir->GetObject(collnamecos+"/pullDxy",cosh5);
  dir->GetObject(collnamectf+"/pullDxy",ctfh5);
  dir->GetObject(collnamers+"/pullDxy",rsh5);
  canvas = new TCanvas("Dxypull","Tracks: Dxy pull",400,350);
  plotPull(canvas, cosh5,ctfh5,rsh5, te,"UU",0.53,0.65,true);
  canvas->cd();
  l = new TLegend(0.15,0.7,0.42,0.88);
  l->SetTextSize(0.035);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh5,cosLabel,"LPF");
  l->AddEntry(ctfh5,ctfLabel,"LPF");
  l->AddEntry(rsh5,rsLabel,"LPF");
  l->Draw();
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/Dxypull.eps");   

  
  //DzPull
  dir->GetObject(collnamecos+"/pullDz",cosh6);
  dir->GetObject(collnamectf+"/pullDz",ctfh6);
  dir->GetObject(collnamers+"/pullDz",rsh6);
  canvas = new TCanvas("Dzpull","Tracks: Dz pull",400,350);
  plotPull(canvas, cosh6,ctfh6,rsh6, te,"UU",0.53,0.65,true);
  canvas->cd();
  l = new TLegend(0.15,0.7,0.42,0.88);
  l->SetTextSize(0.035);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh6,cosLabel,"LPF");
  l->AddEntry(ctfh6,ctfLabel,"LPF");
  l->AddEntry(rsh6,rsLabel,"LPF");
  l->Draw(); 
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/Dzpull.eps");   

 //plotPulls
  canvas = new TCanvas("Tracks3","Tracks: efficiency & fakerate",1000,1400);
  
  plotPulls(canvas,
	    cosh1,ctfh1,rsh1,
	    cosh2,ctfh2,rsh2,
	    cosh3,ctfh3,rsh3,
	    cosh4,ctfh4,rsh4,
	    cosh5,ctfh5,rsh5,
	    cosh6,ctfh6,rsh6,
	    te,"UU",0.53,0.65,true);
  
  canvas->cd();
  l = new TLegend(0.20,0.66,0.80,0.71);
  l->SetTextSize(0.016);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh1,cosLabel2,"LPF");
  l->AddEntry(ctfh1,ctfLabel2,"LPF");
  l->AddEntry(rsh1,rsLabel2,"LPF");
  l->Draw(); 
  canvas->Print("cosmic_algoval_plots/pulls.pdf"); 
  delete l;


  //phi resolution vs eta
  
  dir->GetObject(collnamecos+"/phires_vs_eta_Sigma",cosh1);
  dir->GetObject(collnamectf+"/phires_vs_eta_Sigma",ctfh1);
  dir->GetObject(collnamers+"/phires_vs_eta_Sigma",rsh1);

  cosh1->GetYaxis()->SetTitle("#sigma(#delta #phi) [rad]");
  ctfh1->GetYaxis()->SetTitle("#sigma(#delta #phi) [rad]");
  rsh1->GetYaxis()->SetTitle("#sigma(#delta #phi) [rad]");
  
  cosh1->GetXaxis()->SetTitle("#eta");
  ctfh1->GetXaxis()->SetTitle("#eta");
  rsh1->GetXaxis()->SetTitle("#eta");

  cosh1->GetXaxis()->SetRangeUser(-1,1);
  ctfh1->GetXaxis()->SetRangeUser(-1,1);
  rsh1->GetXaxis()->SetRangeUser(-1,1);

  canvas = new TCanvas("phiresolutioneta","Tracks: resolutions vs eta",400,350);
 
  plotres(canvas, cosh1,ctfh1,rsh1, te,"UU",-1,0.1,false,false,true);
  
  canvas->cd();
  l = new TLegend(0.55,0.80,0.95,0.99);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh1,cosLabel,"LPF");
  l->AddEntry(ctfh1,ctfLabel,"LPF");
  l->AddEntry(rsh1,rsLabel,"LPF");
  l->Draw(); 
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/phiresvseta.eps");   

  
  //cotTheta resolution vs eta
  
  dir->GetObject(collnamecos+"/cotThetares_vs_eta_Sigma",cosh2);
  dir->GetObject(collnamectf+"/cotThetares_vs_eta_Sigma",ctfh2);
  dir->GetObject(collnamers+"/cotThetares_vs_eta_Sigma",rsh2);

  cosh2->GetYaxis()->SetTitle("#sigma(#delta cot#theta) [rad]");
  ctfh2->GetYaxis()->SetTitle("#sigma(#delta cot#theta) [rad]");
  rsh2->GetYaxis()->SetTitle("#sigma(#delta cot#theta) [rad]");
  
  cosh2->GetXaxis()->SetTitle("#eta");
  ctfh2->GetXaxis()->SetTitle("#eta");
  rsh2->GetXaxis()->SetTitle("#eta");
  
  cosh2->GetXaxis()->SetRangeUser(-1,1);
  ctfh2->GetXaxis()->SetRangeUser(-1,1);
  rsh2->GetXaxis()->SetRangeUser(-1,1);
  
  canvas = new TCanvas("cotThetaresolutioneta","Tracks: resolutions vs eta",400,350);
  plotres(canvas, cosh2,ctfh2,rsh2, te,"UU",-1,0.1,false,false,true);

  
  canvas->cd();
  l = new TLegend(0.55,0.80,0.95,0.99);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh2,cosLabel,"LPF");
  l->AddEntry(ctfh2,ctfLabel,"LPF");
  l->AddEntry(rsh2,rsLabel,"LPF");
  l->Draw(); 
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/cotThetaresvseta.eps");   


  //dxy resolution vs eta
  dir->GetObject(collnamecos+"/dxyres_vs_eta_Sigma",cosh3);
  dir->GetObject(collnamectf+"/dxyres_vs_eta_Sigma",ctfh3);
  dir->GetObject(collnamers+"/dxyres_vs_eta_Sigma",rsh3);
    
  cosh3->GetYaxis()->SetTitle("#sigma(#delta dxy) [cm]");
  ctfh3->GetYaxis()->SetTitle("#sigma(#delta dxy) [cm]");
  rsh3->GetYaxis()->SetTitle("#sigma(#delta dxy) [cm]");
  
  cosh3->GetXaxis()->SetTitle("#eta");
  ctfh3->GetXaxis()->SetTitle("#eta");
  rsh3->GetXaxis()->SetTitle("#eta");
  
  cosh3->GetXaxis()->SetRangeUser(-1,1);
  ctfh3->GetXaxis()->SetRangeUser(-1,1);
  rsh3->GetXaxis()->SetRangeUser(-1,1);

  canvas = new TCanvas("dxyresolutioneta","Tracks: resolutions vs eta",400,350);
  plotres(canvas, cosh3,ctfh3,rsh3, te,"UU",-1,0.1,false,false,true);

  
  canvas->cd();
  l = new TLegend(0.55,0.80,0.95,0.99);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh3,cosLabel,"LPF");
  l->AddEntry(ctfh3,ctfLabel,"LPF");
  l->AddEntry(rsh3,rsLabel,"LPF");
  l->Draw(); 
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/dxyresvseta.eps");   


  //dz resolution vs eta

  dir->GetObject(collnamecos+"/dzres_vs_eta_Sigma",cosh4);
  dir->GetObject(collnamectf+"/dzres_vs_eta_Sigma",ctfh4);
  dir->GetObject(collnamers+"/dzres_vs_eta_Sigma",rsh4);

  cosh4->GetYaxis()->SetTitle("#sigma(#delta dz) [cm]");
  ctfh4->GetYaxis()->SetTitle("#sigma(#delta dz) [cm]");
  rsh4->GetYaxis()->SetTitle("#sigma(#delta dz) [cm]");
  
  cosh4->GetXaxis()->SetTitle("#eta");
  ctfh4->GetXaxis()->SetTitle("#eta");
  rsh4->GetXaxis()->SetTitle("#eta");
  
  cosh4->GetXaxis()->SetRangeUser(-1,1);
  ctfh4->GetXaxis()->SetRangeUser(-1,1);
  rsh4->GetXaxis()->SetRangeUser(-1,1);


  canvas = new TCanvas("dzresolutioneta","Tracks: resolutions vs eta",400,350);
  plotres(canvas, cosh4,ctfh4,rsh4, te,"UU",-1,0.1,false,false,true);
  
  canvas->cd();
  l = new TLegend(0.55,0.80,0.95,0.99);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh4,cosLabel,"LPF");
  l->AddEntry(ctfh4,ctfLabel,"LPF");
  l->AddEntry(rsh4,rsLabel,"LPF");
  l->Draw(); 
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/dzresvseta.eps");



  //pt resolution vs eta
  dir->GetObject(collnamecos+"/ptres_vs_eta_Sigma",cosh5);
  dir->GetObject(collnamectf+"/ptres_vs_eta_Sigma",ctfh5);
  dir->GetObject(collnamers+"/ptres_vs_eta_Sigma",rsh5);
  
  cosh5->GetYaxis()->SetTitle("#sigma(#delta p_{t}/p_{t}) ");
  ctfh5->GetYaxis()->SetTitle("#sigma(#delta p_{t}/p_{t}) ");
  rsh5->GetYaxis()->SetTitle("#sigma(#delta p_{t}/p_{t}) ");
  
  cosh5->GetXaxis()->SetTitle("#eta");
  ctfh5->GetXaxis()->SetTitle("#eta");
  rsh5->GetXaxis()->SetTitle("#eta");

  cosh5->GetXaxis()->SetRangeUser(-1,1);
  ctfh5->GetXaxis()->SetRangeUser(-1,1);
  rsh5->GetXaxis()->SetRangeUser(-1,1);

  canvas = new TCanvas("ptresolutioneta","Tracks: resolutions vs eta",400,350);
  plotres(canvas, cosh5,ctfh5,rsh5, te,"UU",-1,0.1,false,false,true);
  
  canvas->cd();
  l = new TLegend(0.55,0.80,0.95,0.99);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh5,cosLabel,"LPF");
  l->AddEntry(ctfh5,ctfLabel,"LPF");
  l->AddEntry(rsh5,rsLabel,"LPF");
  l->Draw(); 
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/ptresvseta.eps");

  //plotResolutions
  canvas = new TCanvas("Tracks6","Tracks: Dxy, Dz, Theta resolution",1000,1400);
  
  plotResolutions(canvas,
		  cosh1,ctfh1,rsh1,
		  cosh2,ctfh2,rsh2,
		  cosh3,ctfh3,rsh3,
		  cosh4,ctfh4,rsh4,
		  cosh5,ctfh5,rsh5,
		  cosh6,ctfh6,rsh6,		  
		  te,"UU",-1,0.1,false,false,true);
  canvas->cd();  
  l = new TLegend(0.20,0.635,0.80,0.685);
  l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);  
   l->AddEntry(cosh1,cosLabel2,"LPF");
   l->AddEntry(ctfh1,ctfLabel2,"LPF");
   l->AddEntry(rsh1,rsLabel2,"LPF");
   l->Draw(); 
   canvas->Print("cosmic_algoval_plots/resolutionsEta.pdf"); 
   delete l;

   //phi resolution vs pt
   dir->GetObject(collnamecos+"/phires_vs_pt_Sigma",cosh1);
   dir->GetObject(collnamectf+"/phires_vs_pt_Sigma",ctfh1);
   dir->GetObject(collnamers+"/phires_vs_pt_Sigma",rsh1);
   
   
   cosh1->GetYaxis()->SetTitle("#sigma(#delta #phi) [rad]");
   ctfh1->GetYaxis()->SetTitle("#sigma(#delta #phi) [rad]");
   rsh1->GetYaxis()->SetTitle("#sigma(#delta #phi) [rad]");
   
   cosh1->GetXaxis()->SetTitle("P_{t} (GeV)");
   ctfh1->GetXaxis()->SetTitle("P_{t} (GeV)");
   rsh1->GetXaxis()->SetTitle("P_{t} (GeV)");
   
   
  canvas = new TCanvas("phiresolutionpt","Tracks: resolutions vs pt",400,350);
 
  plotres(canvas, cosh1,ctfh1,rsh1, te,"UU",-1,0.1,false,false,true);
  
  canvas->cd();
  l = new TLegend(0.55,0.80,0.95,0.99);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh1,cosLabel,"LPF");
  l->AddEntry(ctfh1,ctfLabel,"LPF");
  l->AddEntry(rsh1,rsLabel,"LPF");
  l->Draw(); 
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/phiresvspt.eps");   
  
  
  //cotTheta resolution vs pt
  dir->GetObject(collnamecos+"/cotThetares_vs_pt_Sigma",cosh2);
  dir->GetObject(collnamectf+"/cotThetares_vs_pt_Sigma",ctfh2);
  dir->GetObject(collnamers+"/cotThetares_vs_pt_Sigma",rsh2);
  

  cosh2->GetYaxis()->SetTitle("#sigma(#delta cot#theta) [rad]");
  ctfh2->GetYaxis()->SetTitle("#sigma(#delta cot#theta) [rad]");
  rsh2->GetYaxis()->SetTitle("#sigma(#delta cot#theta) [rad]");

  cosh2->GetXaxis()->SetTitle("P_{t} (GeV)");
  ctfh2->GetXaxis()->SetTitle("P_{t} (GeV)");
  rsh2->GetXaxis()->SetTitle("P_{t} (GeV)");

  
  canvas = new TCanvas("cotThetaresolutionpt","Tracks: resolutions vs pt",400,350);
  plotres(canvas, cosh2,ctfh2,rsh2, te,"UU",-1,0.1,false,false,true);

  
  canvas->cd();
  l = new TLegend(0.55,0.80,0.95,0.99);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh2,cosLabel,"LPF");
  l->AddEntry(ctfh2,ctfLabel,"LPF");
  l->AddEntry(rsh2,rsLabel,"LPF");
  l->Draw(); 
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/cotThetaresvspt.eps");   
  

  //dxy resolution vs pt
  dir->GetObject(collnamecos+"/dxyres_vs_pt_Sigma",cosh3);
  dir->GetObject(collnamectf+"/dxyres_vs_pt_Sigma",ctfh3);
  dir->GetObject(collnamers+"/dxyres_vs_pt_Sigma",rsh3);
  
  cosh3->GetYaxis()->SetTitle("#sigma(#delta dxy) [cm]");
  ctfh3->GetYaxis()->SetTitle("#sigma(#delta dxy) [cm]");
  rsh3->GetYaxis()->SetTitle("#sigma(#delta dxy) [cm]");
  
  cosh3->GetXaxis()->SetTitle("P_{t} (GeV)");
  ctfh3->GetXaxis()->SetTitle("P_{t} (GeV)");
  rsh3->GetXaxis()->SetTitle("P_{t} (GeV)");


  canvas = new TCanvas("dxyresolutionpt","Tracks: resolutions vs pt",400,350);
  plotres(canvas, cosh3,ctfh3,rsh3, te,"UU",-1,0.1,false,false,true);

  
  canvas->cd();
  l = new TLegend(0.55,0.80,0.95,0.99);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh3,cosLabel,"LPF");
  l->AddEntry(ctfh3,ctfLabel,"LPF");
  l->AddEntry(rsh3,rsLabel,"LPF");
  l->Draw(); 
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/dxyresvspt.eps");   
  

  //dz resolution vs pt
  dir->GetObject(collnamecos+"/dzres_vs_pt_Sigma",cosh4);
  dir->GetObject(collnamectf+"/dzres_vs_pt_Sigma",ctfh4);
  dir->GetObject(collnamers+"/dzres_vs_pt_Sigma",rsh4);
  
  cosh4->GetYaxis()->SetTitle("#sigma(#delta dz) [cm]");
  ctfh4->GetYaxis()->SetTitle("#sigma(#delta dz) [cm]");
  rsh4->GetYaxis()->SetTitle("#sigma(#delta dz) [cm]");
  
  cosh4->GetXaxis()->SetTitle("P_{t} (GeV)");
  ctfh4->GetXaxis()->SetTitle("P_{t} (GeV)");
  rsh4->GetXaxis()->SetTitle("P_{t} (GeV)");

  canvas = new TCanvas("dzresolutionpt","Tracks: resolutions vs pt",400,350);
  plotres(canvas, cosh4,ctfh4,rsh4, te,"UU",-1,0.1,false,false,true);
  
  canvas->cd();
  l = new TLegend(0.55,0.80,0.95,0.99);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh4,cosLabel,"LPF");
  l->AddEntry(ctfh4,ctfLabel,"LPF");
  l->AddEntry(rsh4,rsLabel,"LPF");
  l->Draw(); 
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/dzresvspt.eps");



  //pt resolution vs pt
  dir->GetObject(collnamecos+"/ptres_vs_pt_Sigma",cosh5);
  dir->GetObject(collnamectf+"/ptres_vs_pt_Sigma",ctfh5);
  dir->GetObject(collnamers+"/ptres_vs_pt_Sigma",rsh5);
  
  cosh5->GetYaxis()->SetTitle("#sigma(#delta p_{t}/p_{t}) ");
  ctfh5->GetYaxis()->SetTitle("#sigma(#delta p_{t}/p_{t}) ");
  rsh5->GetYaxis()->SetTitle("#sigma(#delta p_{t}/p_{t}) ");
  
  cosh5->GetXaxis()->SetTitle("P_{t} (GeV)");
  ctfh5->GetXaxis()->SetTitle("P_{t} (GeV)");
  rsh5->GetXaxis()->SetTitle("P_{t} (GeV)");


  canvas = new TCanvas("ptresolutionpt","Tracks: resolutions vs pt",400,350);
  plotres(canvas, cosh5,ctfh5,rsh5, te,"UU",-1,0.1,false,false,true);
  
  canvas->cd();
  l = new TLegend(0.55,0.80,0.95,0.99);
  l->SetTextSize(0.05);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(cosh5,cosLabel,"LPF");
  l->AddEntry(ctfh5,ctfLabel,"LPF");
  l->AddEntry(rsh5,rsLabel,"LPF");
  l->Draw(); 
  if(IFPLOTEPS)
    canvas->Print("cosmic_algoval_plots/ptresvspt.eps");

  //plotResolutions
  canvas = new TCanvas("Tracks7","Tracks: Dxy, Dz, Theta resolution",1000,1400);
  
  plotResolutions(canvas,
		  cosh1,ctfh1,rsh1,
		  cosh2,ctfh2,rsh2,
		  cosh3,ctfh3,rsh3,
		  cosh4,ctfh4,rsh4,
		  cosh5,ctfh5,rsh5,
		  cosh6,ctfh6,rsh6,		  
		  te,"UU",-1,0.1,false,false,true);
  canvas->cd();  
  l = new TLegend(0.20,0.635,0.80,0.685);
  l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);  
   l->AddEntry(cosh1,cosLabel2,"LPF");
   l->AddEntry(ctfh1,ctfLabel2,"LPF");
   l->AddEntry(rsh1,rsLabel2,"LPF");
   l->Draw(); 
   canvas->Print("cosmic_algoval_plots/resolutionsPt.pdf"); 
   delete l;



}

void ploteff(TCanvas *canvas, TH1F *cos1,TH1F *ctf1, TH1F *rs1, TText* te,
	     char * option, double startingY, double startingX = .1,bool fit = false, bool logx=false, bool logy=false){
  
  cos1->SetMarkerStyle(20);
  ctf1->SetMarkerStyle(21);
  rs1->SetMarkerStyle(22);
  cos1->SetMarkerColor(2);
  ctf1->SetMarkerColor(4);
  rs1->SetMarkerColor(3);
  cos1->SetMarkerSize(1.2);
  ctf1->SetMarkerSize(1.2);
  rs1->SetMarkerSize(1.2);
  cos1->SetLineColor(1);
  ctf1->SetLineColor(1);
  rs1->SetLineColor(1);
 


  ctf1->GetYaxis()->SetTitleSize(0.06);
  cos1->GetYaxis()->SetTitleSize(0.06);
  rs1->GetYaxis()->SetTitleSize(0.06);
  ctf1->GetYaxis()->SetTitleOffset(1.05);
  cos1->GetYaxis()->SetTitleOffset(1.05);
  rs1->GetYaxis()->SetTitleOffset(1.05);
  ctf1->GetXaxis()->SetTitleSize(0.06);
  cos1->GetXaxis()->SetTitleSize(0.06);
  rs1->GetXaxis()->SetTitleSize(0.06);
  ctf1->GetXaxis()->SetTitleOffset(0.85);
  cos1->GetXaxis()->SetTitleOffset(0.85);
  rs1->GetXaxis()->SetTitleOffset(0.85);

  canvas->cd(); 
  if(logx)gPad->SetLogx(); 
  if(logy)gPad->SetLogy(); 
  setStats(cos1,ctf1, rs1, startingY, startingX, fit);
  cos1->Draw();
  ctf1->Draw("sames");
  rs1->Draw("sames");

}

void plotBuilding(TCanvas *canvas, 
		  TH1F *cos1,TH1F *ctf1, TH1F *rs1, 
		  TH1F *cos2,TH1F *ctf2, TH1F *rs2, 
		  TH1F *cos3,TH1F *ctf3, TH1F *rs3, 
		  TH1F *cos4,TH1F *ctf4, TH1F *rs4, 
		  TH1F *cos5,TH1F *ctf5, TH1F *rs5, 
		  TH1F *cos6,TH1F *ctf6, TH1F *rs6, 
		  TText* te, char * option, 
		  double startingY, double startingX = .1,
		  bool fit = false, bool logx=false, bool logy=false){

  canvas->Divide(2,3);
  cos1->SetMarkerStyle(20);
  ctf1->SetMarkerStyle(21);
  rs1->SetMarkerStyle(22);
  cos1->SetMarkerColor(2);
  ctf1->SetMarkerColor(4);
  rs1->SetMarkerColor(3);
  cos1->SetMarkerSize(0.7);
  ctf1->SetMarkerSize(0.7);
  rs1->SetMarkerSize(0.7);
  cos1->SetLineColor(1);
  ctf1->SetLineColor(1);
  rs1->SetLineColor(1);
  cos1->SetTitle("");
  ctf1->SetTitle("");
  rs1->SetTitle("");

  ctf1->GetYaxis()->SetTitleSize(0.06);
  cos1->GetYaxis()->SetTitleSize(0.06);
  rs1->GetYaxis()->SetTitleSize(0.06);
  ctf1->GetYaxis()->SetTitleOffset(1.05);
  cos1->GetYaxis()->SetTitleOffset(1.05);
  rs1->GetYaxis()->SetTitleOffset(1.05);
  ctf1->GetXaxis()->SetTitleSize(0.06);
  cos1->GetXaxis()->SetTitleSize(0.06);
  rs1->GetXaxis()->SetTitleSize(0.06);
  ctf1->GetXaxis()->SetTitleOffset(0.85);
  cos1->GetXaxis()->SetTitleOffset(0.85);
  rs1->GetXaxis()->SetTitleOffset(0.85);

  cos2->SetMarkerStyle(20);
  ctf2->SetMarkerStyle(21);
  rs2->SetMarkerStyle(22);
  cos2->SetMarkerColor(2);
  ctf2->SetMarkerColor(4);
  rs2->SetMarkerColor(3);
  cos2->SetMarkerSize(0.7);
  ctf2->SetMarkerSize(0.7);
  rs2->SetMarkerSize(0.7);
  cos2->SetLineColor(1);
  ctf2->SetLineColor(1);
  rs2->SetLineColor(1);
  cos2->SetTitle("");
  ctf2->SetTitle("");
  rs2->SetTitle("");
  
  ctf2->GetYaxis()->SetTitleSize(0.06);
  cos2->GetYaxis()->SetTitleSize(0.06);
  rs2->GetYaxis()->SetTitleSize(0.06);
  ctf2->GetYaxis()->SetTitleOffset(1.05);
  cos2->GetYaxis()->SetTitleOffset(1.05);
  rs2->GetYaxis()->SetTitleOffset(1.05);
  ctf2->GetXaxis()->SetTitleSize(0.06);
  cos2->GetXaxis()->SetTitleSize(0.06);
  rs2->GetXaxis()->SetTitleSize(0.06);
  ctf2->GetXaxis()->SetTitleOffset(0.85);
  cos2->GetXaxis()->SetTitleOffset(0.85);
  rs2->GetXaxis()->SetTitleOffset(0.85);

  cos3->SetMarkerStyle(20);
  ctf3->SetMarkerStyle(21);
  rs3->SetMarkerStyle(22);
  cos3->SetMarkerColor(2);
  ctf3->SetMarkerColor(4);
  rs3->SetMarkerColor(3);
  cos3->SetMarkerSize(0.7);
  ctf3->SetMarkerSize(0.7);
  rs3->SetMarkerSize(0.7);
  cos3->SetLineColor(1);
  ctf3->SetLineColor(1);
  rs3->SetLineColor(1);
  cos3->SetTitle("");
  ctf3->SetTitle("");
  rs3->SetTitle("");

  ctf3->GetYaxis()->SetTitleSize(0.06);
  cos3->GetYaxis()->SetTitleSize(0.06);
  rs3->GetYaxis()->SetTitleSize(0.06);
  ctf3->GetYaxis()->SetTitleOffset(1.05);
  cos3->GetYaxis()->SetTitleOffset(1.05);
  rs3->GetYaxis()->SetTitleOffset(1.05);
  ctf3->GetXaxis()->SetTitleSize(0.06);
  cos3->GetXaxis()->SetTitleSize(0.06);
  rs3->GetXaxis()->SetTitleSize(0.06);
  ctf3->GetXaxis()->SetTitleOffset(0.85);
  cos3->GetXaxis()->SetTitleOffset(0.85);
  rs3->GetXaxis()->SetTitleOffset(0.85);

  cos4->SetMarkerStyle(20);
  ctf4->SetMarkerStyle(21);
  rs4->SetMarkerStyle(22);
  cos4->SetMarkerColor(2);
  ctf4->SetMarkerColor(4);
  rs4->SetMarkerColor(3);
  cos4->SetMarkerSize(0.7);
  ctf4->SetMarkerSize(0.7);
  rs4->SetMarkerSize(0.7);
  cos4->SetLineColor(1);
  ctf4->SetLineColor(1);
  rs4->SetLineColor(1);
  cos4->SetTitle("");
  ctf4->SetTitle("");
  rs4->SetTitle("");

  ctf4->GetYaxis()->SetTitleSize(0.06);
  cos4->GetYaxis()->SetTitleSize(0.06);
  rs4->GetYaxis()->SetTitleSize(0.06);
  ctf4->GetYaxis()->SetTitleOffset(1.05);
  cos4->GetYaxis()->SetTitleOffset(1.05);
  rs4->GetYaxis()->SetTitleOffset(1.05);
  ctf4->GetXaxis()->SetTitleSize(0.06);
  cos4->GetXaxis()->SetTitleSize(0.06);
  rs4->GetXaxis()->SetTitleSize(0.06);
  ctf4->GetXaxis()->SetTitleOffset(0.85);
  cos4->GetXaxis()->SetTitleOffset(0.85);
  rs4->GetXaxis()->SetTitleOffset(0.85);
  

  cos5->SetMarkerStyle(20);
  ctf5->SetMarkerStyle(21);
  rs5->SetMarkerStyle(22);
  cos5->SetMarkerColor(2);
  ctf5->SetMarkerColor(4);
  rs5->SetMarkerColor(3);
  cos5->SetMarkerSize(0.7);
  ctf5->SetMarkerSize(0.7);
  rs5->SetMarkerSize(0.7);
  cos5->SetLineColor(1);
  ctf5->SetLineColor(1);
  rs5->SetLineColor(1);
  cos5->SetTitle("");
  ctf5->SetTitle("");
  rs5->SetTitle("");


  ctf5->GetYaxis()->SetTitleSize(0.06);
  cos5->GetYaxis()->SetTitleSize(0.06);
  rs5->GetYaxis()->SetTitleSize(0.06);
  ctf5->GetYaxis()->SetTitleOffset(1.05);
  cos5->GetYaxis()->SetTitleOffset(1.05);
  rs5->GetYaxis()->SetTitleOffset(1.05);
  ctf5->GetXaxis()->SetTitleSize(0.06);
  cos5->GetXaxis()->SetTitleSize(0.06);
  rs5->GetXaxis()->SetTitleSize(0.06);
  ctf5->GetXaxis()->SetTitleOffset(0.85);
  cos5->GetXaxis()->SetTitleOffset(0.85);
  rs5->GetXaxis()->SetTitleOffset(0.85);

  cos6->SetMarkerStyle(20);
  ctf6->SetMarkerStyle(21);
  rs6->SetMarkerStyle(22);
  cos6->SetMarkerColor(2);
  ctf6->SetMarkerColor(4);
  rs6->SetMarkerColor(3);
  cos6->SetMarkerSize(0.7);
  ctf6->SetMarkerSize(0.7);
  rs6->SetMarkerSize(0.7);
  cos6->SetLineColor(1);
  ctf6->SetLineColor(1);
  rs6->SetLineColor(1);
  cos6->SetTitle("");
  ctf6->SetTitle("");
  rs6->SetTitle("");


  ctf6->GetYaxis()->SetTitleSize(0.06);
  cos6->GetYaxis()->SetTitleSize(0.06);
  rs6->GetYaxis()->SetTitleSize(0.06);
  ctf6->GetYaxis()->SetTitleOffset(1.05);
  cos6->GetYaxis()->SetTitleOffset(1.05);
  rs6->GetYaxis()->SetTitleOffset(1.05);
  ctf6->GetXaxis()->SetTitleSize(0.06);
  cos6->GetXaxis()->SetTitleSize(0.06);
  rs6->GetXaxis()->SetTitleSize(0.06);
  ctf6->GetXaxis()->SetTitleOffset(0.85);
  cos6->GetXaxis()->SetTitleOffset(0.85);
  rs6->GetXaxis()->SetTitleOffset(0.85);

  canvas->cd(1); 
  if(logx)gPad->SetLogx(); 
  setStats(cos1,ctf1, rs1, startingY, startingX, fit);
  cos1->Draw();
  ctf1->Draw("sames");
  rs1->Draw("sames");

  canvas->cd(2); 
  if(logx)gPad->SetLogx(); 
  if(logy)gPad->SetLogy(); 
  setStats(cos2,ctf2, rs2, startingY, startingX, fit);
  cos2->Draw();
  ctf2->Draw("sames");
  rs2->Draw("sames");

  canvas->cd(3); 
  if(logx)gPad->SetLogx(); 
  setStats(cos3,ctf3, rs3, startingY, startingX, fit);
  cos3->Draw();
  ctf3->Draw("sames");
  rs3->Draw("sames");


  canvas->cd(4); 
  if(logx)gPad->SetLogx(); 
  if(logy)gPad->SetLogy(); 
  setStats(cos4,ctf4, rs4, startingY, startingX, fit);
  cos4->Draw();
  ctf4->Draw("sames");
  rs4->Draw("sames");


  canvas->cd(5); 
  if(logx)gPad->SetLogx(); 
  setStats(cos5,ctf5, rs5, startingY, startingX, fit);
  cos5->Draw();
  ctf5->Draw("sames");
  rs5->Draw("sames");

 canvas->cd(6); 
  if(logx)gPad->SetLogx(); 
  if(logy)gPad->SetLogy(); 
  setStats(cos6,ctf6, rs6, startingY, startingX, fit);
  cos6->Draw();
  ctf6->Draw("sames");
  rs6->Draw("sames");




}


void setStats(TH1* cos, TH1* ctf, TH1* rs, double startingY, double startingX = .1,bool fit){
  if (startingY<0){
    cos->SetStats(0);
    ctf->SetStats(0);
    rs->SetStats(0);
  } else {
    cos->SetStats(1);
    ctf->SetStats(1);
    rs->SetStats(1);
    
    if (fit){
      cos->Fit("gaus", "Q", "0", -2, 2);
      TF1* f1 = (TF1*) cos->GetListOfFunctions()->FindObject("gaus");
      f1->SetLineColor(2);
      f1->SetLineWidth(1);
    }
    cos->Draw();
    gPad->Update(); 
    TPaveStats* st1 = (TPaveStats*) cos->GetListOfFunctions()->FindObject("stats");
    if (fit) {st1->SetOptFit(0010);    st1->SetOptStat(1001);}
    st1->SetX1NDC(startingX);
    st1->SetX2NDC(startingX+0.30);
    st1->SetY1NDC(startingY+0.30);
    st1->SetY2NDC(startingY+0.45);
    st1->SetTextColor(2);

    if (fit) {
      ctf->Fit("gaus","Q","0", -2,2);
      TF1* f2 = (TF1*) ctf->GetListOfFunctions()->FindObject("gaus");
      f2->SetLineColor(4);
      f2->SetLineWidth(1);    
    }
    ctf->Draw();
    gPad->Update(); 
    TPaveStats* st2 = (TPaveStats*) ctf->GetListOfFunctions()->FindObject("stats");
    if (fit) {st2->SetOptFit(0010);    st2->SetOptStat(1001);}
    st2->SetX1NDC(startingX);
    st2->SetX2NDC(startingX+0.30);
    st2->SetY1NDC(startingY+0.15);
    st2->SetY2NDC(startingY+0.30);
    st2->SetTextColor(4);

   if (fit) {
     rs->Fit("gaus","Q","0",-2,2);
      TF1* f2 = (TF1*) rs->GetListOfFunctions()->FindObject("gaus");
      f2->SetLineColor(3);
      f2->SetLineWidth(1);    
    }
    rs->Draw();
    gPad->Update(); 
    TPaveStats* st3 = (TPaveStats*) rs->GetListOfFunctions()->FindObject("stats");
    if (fit) {st3->SetOptFit(0010);    st3->SetOptStat(1001);}
    st3->SetX1NDC(startingX);
    st3->SetX2NDC(startingX+0.30);
    st3->SetY1NDC(startingY);
    st3->SetY2NDC(startingY+0.15);
    st3->SetTextColor(3);

  }
}

void fixRangeY(TH1* cos,TH1* ctf, TH1* rs){

  double ymin1 = (cos->GetBinContent(cos->GetMinimumBin()) < ctf->GetBinContent(ctf->GetMinimumBin())) ? 
    cos->GetBinContent(cos->GetMinimumBin()) : ctf->GetBinContent(ctf->GetMinimumBin());

  double ymin2 = (ctf->GetBinContent(ctf->GetMinimumBin()) < rs->GetBinContent(rs->GetMinimumBin())) ? 
    ctf->GetBinContent(ctf->GetMinimumBin()) : rs->GetBinContent(rs->GetMinimumBin());

  double ymin = (ymin1<ymin2)?ymin1:ymin2;


  double ymax1 = (cos->GetBinContent(cos->GetMaximumBin()) > ctf->GetBinContent(ctf->GetMaximumBin())) ?
    cos->GetBinContent(cos->GetMaximumBin()) : ctf->GetBinContent(ctf->GetMaximumBin());

  double ymax2 = (ctf->GetBinContent(ctf->GetMaximumBin()) > rs->GetBinContent(rs->GetMaximumBin())) ?
    ctf->GetBinContent(ctf->GetMaximumBin()) : rs->GetBinContent(rs->GetMaximumBin());
  
  double ymax= (ymax1>ymax2)?ymax1:ymax2;

  cos->GetYaxis()->SetRangeUser(ymin*0.9,ymax*1.1);
  ctf->GetYaxis()->SetRangeUser(ymin*0.9,ymax*1.1);
  rs->GetYaxis()->SetRangeUser(ymin*0.9,ymax*1.1);

}

void plotPull(TCanvas *canvas, 
	      TH1F *cos1,TH1F *ctf1, TH1F *rs1,
	      TText* te,  char * option, 
	      double startingY, double startingX = .1,bool fit = false){
  
  cos1->SetMarkerStyle(20);
  ctf1->SetMarkerStyle(21);
  rs1->SetMarkerStyle(22);
  cos1->SetMarkerColor(2);
  ctf1->SetMarkerColor(4);
  rs1->SetMarkerColor(3);
  cos1->SetMarkerSize(0.7);
  ctf1->SetMarkerSize(0.7);
  rs1->SetMarkerSize(0.7);
  cos1->SetLineColor(2);
  ctf1->SetLineColor(4);
  rs1->SetLineColor(3);
  cos1->GetXaxis()->SetRangeUser(-10,10);
  ctf1->GetXaxis()->SetRangeUser(-10,10);
  rs1->GetXaxis()->SetRangeUser(-10,10);
  NormalizeHistograms(cos1,ctf1,rs1);
  fixRangeY(cos1, ctf1, rs1);
  canvas->cd();
  setStats(cos1,ctf1, rs1, startingY, startingX, fit);
  cos1->Draw();
  ctf1->Draw("sames");
  rs1->Draw("sames");
}


void plotPulls(TCanvas *canvas, 
	       TH1F *cos1,TH1F *ctf1, TH1F *rs1, 
	       TH1F *cos2,TH1F *ctf2, TH1F *rs2, 
	       TH1F *cos3,TH1F *ctf3, TH1F *rs3, 
	       TH1F *cos4,TH1F *ctf4, TH1F *rs4, 
	       TH1F *cos5,TH1F *ctf5, TH1F *rs5, 
	       TH1F *cos6,TH1F *ctf6, TH1F *rs6, 	     
	       TText* te,  char * option, 
	       double startingY, double startingX = .1,bool fit = true){

  canvas->Divide(2,3);
  
  cos1->SetMarkerStyle(20);
  ctf1->SetMarkerStyle(21);
  rs1->SetMarkerStyle(22);
  cos1->SetMarkerColor(2);
  ctf1->SetMarkerColor(4);
  rs1->SetMarkerColor(3);
  cos1->SetMarkerSize(0.7);
  ctf1->SetMarkerSize(0.7);
  rs1->SetMarkerSize(0.7);
  cos1->SetLineColor(2);
  ctf1->SetLineColor(4);
  rs1->SetLineColor(3);
  cos1->GetXaxis()->SetRangeUser(-10,10);
  ctf1->GetXaxis()->SetRangeUser(-10,10);
  rs1->GetXaxis()->SetRangeUser(-10,10);
  NormalizeHistograms(cos1,ctf1,rs1);
  fixRangeY(cos1, ctf1, rs1);

  cos2->SetMarkerStyle(20);
  ctf2->SetMarkerStyle(21);
  rs2->SetMarkerStyle(22);
  cos2->SetMarkerColor(2);
  ctf2->SetMarkerColor(4);
  rs2->SetMarkerColor(3);
  cos2->SetMarkerSize(0.7);
  ctf2->SetMarkerSize(0.7);
  rs2->SetMarkerSize(0.7);
  cos2->SetLineColor(2);
  ctf2->SetLineColor(4);
  rs2->SetLineColor(3);
  cos2->GetXaxis()->SetRangeUser(-10,10);
  ctf2->GetXaxis()->SetRangeUser(-10,10);
  rs2->GetXaxis()->SetRangeUser(-10,10);
  //NormalizeHistograms(cos2,ctf2,rs2);
  fixRangeY(cos2, ctf2, rs2);


  cos3->SetMarkerStyle(20);
  ctf3->SetMarkerStyle(21);
  rs3->SetMarkerStyle(22);
  cos3->SetMarkerColor(2);
  ctf3->SetMarkerColor(4);
  rs3->SetMarkerColor(3);
  cos3->SetMarkerSize(0.7);
  ctf3->SetMarkerSize(0.7);
  rs3->SetMarkerSize(0.7);
  cos3->SetLineColor(2);
  ctf3->SetLineColor(4);
  rs3->SetLineColor(3);
  cos3->GetXaxis()->SetRangeUser(-10,10);
  ctf3->GetXaxis()->SetRangeUser(-10,10);
  rs3->GetXaxis()->SetRangeUser(-10,10);
  //NormalizeHistograms(cos3,ctf3,rs3);
  fixRangeY(cos3, ctf3, rs3);



  cos4->SetMarkerStyle(20);
 ctf4->SetMarkerStyle(21);
 rs4->SetMarkerStyle(22);
 cos4->SetMarkerColor(2);
 ctf4->SetMarkerColor(4);
 rs4->SetMarkerColor(3);
 cos4->SetMarkerSize(0.7);
 ctf4->SetMarkerSize(0.7);
 rs4->SetMarkerSize(0.7);
 cos4->SetLineColor(2);
 ctf4->SetLineColor(4);
 rs4->SetLineColor(3);
 cos4->GetXaxis()->SetRangeUser(-10,10);
 ctf4->GetXaxis()->SetRangeUser(-10,10);
  rs4->GetXaxis()->SetRangeUser(-10,10);
  //NormalizeHistograms(cos4,ctf4,rs4);
  fixRangeY(cos4, ctf4, rs4);
  
  

 cos5->SetMarkerStyle(20);
 ctf5->SetMarkerStyle(21);
 rs5->SetMarkerStyle(22);
 cos5->SetMarkerColor(2);
 ctf5->SetMarkerColor(4);
 rs5->SetMarkerColor(3);
 cos5->SetMarkerSize(0.7);
 ctf5->SetMarkerSize(0.7);
 rs5->SetMarkerSize(0.7);
 cos5->SetLineColor(2);
 ctf5->SetLineColor(4);
 rs5->SetLineColor(3);
 cos5->GetXaxis()->SetRangeUser(-10,10);
 ctf5->GetXaxis()->SetRangeUser(-10,10);
  rs5->GetXaxis()->SetRangeUser(-10,10);
  //NormalizeHistograms(cos5,ctf5,rs5);
  fixRangeY(cos5, ctf5, rs5);
  


 cos6->SetMarkerStyle(20);
 ctf6->SetMarkerStyle(21);
 rs6->SetMarkerStyle(22);
 cos6->SetMarkerColor(2);
 ctf6->SetMarkerColor(4);
 rs6->SetMarkerColor(3);
 cos6->SetMarkerSize(0.7);
 ctf6->SetMarkerSize(0.7);
 rs6->SetMarkerSize(0.7);
 cos6->SetLineColor(2);
 ctf6->SetLineColor(4);
 rs6->SetLineColor(3);
 cos6->GetXaxis()->SetRangeUser(-10,10);
 ctf6->GetXaxis()->SetRangeUser(-10,10);
  rs6->GetXaxis()->SetRangeUser(-10,10);
  //NormalizeHistograms(cos6,ctf6,rs6);
  fixRangeY(cos6, ctf6, rs6);

  canvas->cd(1);
  setStats(cos1,ctf1, rs1, startingY, startingX, fit);
  cos1->Draw();
  ctf1->Draw("sames");
  rs1->Draw("sames");
  
  canvas->cd(2);
  setStats(cos2,ctf2, rs2, startingY, startingX, fit);
  cos2->Draw();
  ctf2->Draw("sames");
  rs2->Draw("sames");
  
  
  canvas->cd(3);
  setStats(cos3,ctf3, rs3, startingY, startingX, fit);
  cos3->Draw();
  ctf3->Draw("sames");
  rs3->Draw("sames");
  
  canvas->cd(4);
  setStats(cos4,ctf4, rs4, startingY, startingX, fit);
  cos4->Draw();
  ctf4->Draw("sames");
  rs4->Draw("sames");
  
  canvas->cd(5);
  setStats(cos5,ctf5, rs5, startingY, startingX, fit);
  cos5->Draw();
  ctf5->Draw("sames");
  rs5->Draw("sames");


  canvas->cd(6);
  setStats(cos6,ctf6, rs6, startingY, startingX, fit);
  cos6->Draw();
  ctf6->Draw("sames");
  rs6->Draw("sames");


}  


void NormalizeHistograms(TH1F* h1, TH1F* h2, TH1F *h3)
{
  if (h1==0 || h2==0 || h3 ==0) return;
  float scale1 = -9999.9;
  float scale2 = -9999.9;
  float scale3 = -9999.9;

  if ( h1->Integral() != 0 && h2->Integral() != 0 && h3->Integral()!=0){
      scale1 = 1.0/(float)h1->Integral();
      scale2 = 1.0/(float)h2->Integral();
      scale3 = 1.0/(float)h3->Integral();
    
      //h1->Sumw2();
      //h2->Sumw2();
      //h3->Sumw2();
      h1->Scale(scale1);
      h2->Scale(scale2);
      h3->Scale(scale3);
    }
}



void plotres(TCanvas *canvas, TH1F *cos1,TH1F *ctf1, TH1F *rs1, 
	     TText* te, char * option, double startingY, double startingX = .1,
	     bool fit = false, bool logx = false, bool logy=false){

  cos1->SetMarkerStyle(20);
  ctf1->SetMarkerStyle(21);
  rs1->SetMarkerStyle(22);
  cos1->SetMarkerColor(2);
  ctf1->SetMarkerColor(4);
  rs1->SetMarkerColor(3);
  cos1->SetMarkerSize(0.7);
  ctf1->SetMarkerSize(0.7);
  rs1->SetMarkerSize(0.7);
  cos1->SetLineColor(1);
  ctf1->SetLineColor(1);
  rs1->SetLineColor(1);
  cos1->SetLineWidth(1);
  ctf1->SetLineWidth(1);
  rs1->SetLineWidth(1);

  ctf1->SetTitle("");
  cos1->SetTitle("");
  rs1->SetTitle("");

  ctf1->GetYaxis()->SetTitleSize(0.06);
  cos1->GetYaxis()->SetTitleSize(0.06);
  rs1->GetYaxis()->SetTitleSize(0.06);
  ctf1->GetYaxis()->SetTitleOffset(1.05);
  cos1->GetYaxis()->SetTitleOffset(1.05);
  rs1->GetYaxis()->SetTitleOffset(1.05);
  ctf1->GetXaxis()->SetTitleSize(0.06);
  cos1->GetXaxis()->SetTitleSize(0.06);
  rs1->GetXaxis()->SetTitleSize(0.06);
  ctf1->GetXaxis()->SetTitleOffset(0.85);
  cos1->GetXaxis()->SetTitleOffset(0.85);
  rs1->GetXaxis()->SetTitleOffset(0.85);

  canvas->cd();
  if(logx) gPad->SetLogx();
  if(logy) gPad->SetLogy();
  setStats(cos1,ctf1, rs1, startingY, startingX, fit);
  cos1->Draw("p");
  ctf1->Draw("psames");
  rs1->Draw("psames");

}


void plotResolutions(TCanvas *canvas, 
		     TH1F *cos1,TH1F *ctf1, TH1F *rs1, 
		     TH1F *cos2,TH1F *ctf2, TH1F *rs2, 
		     TH1F *cos3,TH1F *ctf3, TH1F *rs3, 
		     TH1F *cos4,TH1F *ctf4, TH1F *rs4, 
		     TH1F *cos5,TH1F *ctf5, TH1F *rs5, 
		     TH1F *cos6,TH1F *ctf6, TH1F *rs6, 
		     TText* te, char * option, 
		     double startingY, double startingX = .1,
		     bool fit = false, bool logx=false, bool logy=false){

  canvas->Divide(2,3);

  cos1->SetMarkerStyle(20);
  ctf1->SetMarkerStyle(21);
  rs1->SetMarkerStyle(22);
  cos1->SetMarkerColor(2);
  ctf1->SetMarkerColor(4);
  rs1->SetMarkerColor(3);
  cos1->SetMarkerSize(0.7);
  ctf1->SetMarkerSize(0.7);
  rs1->SetMarkerSize(0.7);
  cos1->SetLineColor(1);
  ctf1->SetLineColor(1);
  rs1->SetLineColor(1);
  cos1->SetLineWidth(1);
  ctf1->SetLineWidth(1);
  rs1->SetLineWidth(1);
  
  ctf1->GetYaxis()->SetTitleSize(0.06);
  cos1->GetYaxis()->SetTitleSize(0.06);
  rs1->GetYaxis()->SetTitleSize(0.06);
  ctf1->GetYaxis()->SetTitleOffset(1.05);
  cos1->GetYaxis()->SetTitleOffset(1.05);
  rs1->GetYaxis()->SetTitleOffset(1.05);
  ctf1->GetXaxis()->SetTitleSize(0.06);
  cos1->GetXaxis()->SetTitleSize(0.06);
  rs1->GetXaxis()->SetTitleSize(0.06);
  ctf1->GetXaxis()->SetTitleOffset(0.85);
  cos1->GetXaxis()->SetTitleOffset(0.85);
  rs1->GetXaxis()->SetTitleOffset(0.85);


  cos2->SetMarkerStyle(20);
  ctf2->SetMarkerStyle(21);
  rs2->SetMarkerStyle(22);
  cos2->SetMarkerColor(2);
  ctf2->SetMarkerColor(4);
  rs2->SetMarkerColor(3);
  cos2->SetMarkerSize(0.7);
  ctf2->SetMarkerSize(0.7);
  rs2->SetMarkerSize(0.7);
  cos2->SetLineColor(1);
  ctf2->SetLineColor(1);
  rs2->SetLineColor(1);
  cos2->SetLineWidth(1);
  ctf2->SetLineWidth(1);
  rs2->SetLineWidth(1);
 
  ctf2->GetYaxis()->SetTitleSize(0.06);
  cos2->GetYaxis()->SetTitleSize(0.06);
  rs2->GetYaxis()->SetTitleSize(0.06);
  ctf2->GetYaxis()->SetTitleOffset(1.05);
  cos2->GetYaxis()->SetTitleOffset(1.05);
  rs2->GetYaxis()->SetTitleOffset(1.05);
  ctf2->GetXaxis()->SetTitleSize(0.06);
  cos2->GetXaxis()->SetTitleSize(0.06);
  rs2->GetXaxis()->SetTitleSize(0.06);
  ctf2->GetXaxis()->SetTitleOffset(0.85);
  cos2->GetXaxis()->SetTitleOffset(0.85);
  rs2->GetXaxis()->SetTitleOffset(0.85);

  cos3->SetMarkerStyle(20);
  ctf3->SetMarkerStyle(21);
  rs3->SetMarkerStyle(22);
  cos3->SetMarkerColor(2);
  ctf3->SetMarkerColor(4);
  rs3->SetMarkerColor(3);
  cos3->SetMarkerSize(0.7);
  ctf3->SetMarkerSize(0.7);
  rs3->SetMarkerSize(0.7);
  cos3->SetLineColor(1);
  ctf3->SetLineColor(1);
  rs3->SetLineColor(1);
  cos3->SetLineWidth(1);
  ctf3->SetLineWidth(1);
  rs3->SetLineWidth(1);
  
  ctf3->GetYaxis()->SetTitleSize(0.06);
  cos3->GetYaxis()->SetTitleSize(0.06);
  rs3->GetYaxis()->SetTitleSize(0.06);
  ctf3->GetYaxis()->SetTitleOffset(1.05);
  cos3->GetYaxis()->SetTitleOffset(1.05);
  rs3->GetYaxis()->SetTitleOffset(1.05);
  ctf3->GetXaxis()->SetTitleSize(0.06);
  cos3->GetXaxis()->SetTitleSize(0.06);
  rs3->GetXaxis()->SetTitleSize(0.06);
  ctf3->GetXaxis()->SetTitleOffset(0.85);
  cos3->GetXaxis()->SetTitleOffset(0.85);
  rs3->GetXaxis()->SetTitleOffset(0.85);

  cos4->SetMarkerStyle(20);
  ctf4->SetMarkerStyle(21);
  rs4->SetMarkerStyle(22);
  cos4->SetMarkerColor(2);
  ctf4->SetMarkerColor(4);
  rs4->SetMarkerColor(3);
  cos4->SetMarkerSize(0.7);
  ctf4->SetMarkerSize(0.7);
  rs4->SetMarkerSize(0.7);
  cos4->SetLineColor(1);
  ctf4->SetLineColor(1);
  rs4->SetLineColor(1);
  cos4->SetLineWidth(1);
  ctf4->SetLineWidth(1);
  rs4->SetLineWidth(1);
 

  
  ctf4->GetYaxis()->SetTitleSize(0.06);
  cos4->GetYaxis()->SetTitleSize(0.06);
  rs4->GetYaxis()->SetTitleSize(0.06);
  ctf4->GetYaxis()->SetTitleOffset(1.05);
  cos4->GetYaxis()->SetTitleOffset(1.05);
  rs4->GetYaxis()->SetTitleOffset(1.05);
  ctf4->GetXaxis()->SetTitleSize(0.06);
  cos4->GetXaxis()->SetTitleSize(0.06);
  rs4->GetXaxis()->SetTitleSize(0.06);
  ctf4->GetXaxis()->SetTitleOffset(0.85);
  cos4->GetXaxis()->SetTitleOffset(0.85);
  rs4->GetXaxis()->SetTitleOffset(0.85);



 cos5->SetMarkerStyle(20);
 ctf5->SetMarkerStyle(21);
 rs5->SetMarkerStyle(22);
  cos5->SetMarkerColor(2);
  ctf5->SetMarkerColor(4);
  rs5->SetMarkerColor(3);
  cos5->SetMarkerSize(0.7);
  ctf5->SetMarkerSize(0.7);
  rs5->SetMarkerSize(0.7);
  cos5->SetLineColor(1);
  ctf5->SetLineColor(1);
  rs5->SetLineColor(1);
  cos5->SetLineWidth(1);
  ctf5->SetLineWidth(1);
  rs5->SetLineWidth(1);
  

  ctf5->GetYaxis()->SetTitleSize(0.06);
  cos5->GetYaxis()->SetTitleSize(0.06);
  rs5->GetYaxis()->SetTitleSize(0.06);
  ctf5->GetYaxis()->SetTitleOffset(1.05);
  cos5->GetYaxis()->SetTitleOffset(1.05);
  rs5->GetYaxis()->SetTitleOffset(1.05);
  ctf5->GetXaxis()->SetTitleSize(0.06);
  cos5->GetXaxis()->SetTitleSize(0.06);
  rs5->GetXaxis()->SetTitleSize(0.06);
  ctf5->GetXaxis()->SetTitleOffset(0.85);
  cos5->GetXaxis()->SetTitleOffset(0.85);
  rs5->GetXaxis()->SetTitleOffset(0.85);

  cos6->SetMarkerStyle(20);
  ctf6->SetMarkerStyle(21);
  rs6->SetMarkerStyle(22);
  cos6->SetMarkerColor(2);
  ctf6->SetMarkerColor(4);
  rs6->SetMarkerColor(3);
  cos6->SetMarkerSize(0.7);
  ctf6->SetMarkerSize(0.7);
  rs6->SetMarkerSize(0.7);
  rs6->SetMarkerSize(0.7);
  cos4->SetLineColor(1);
  ctf4->SetLineColor(1);
  rs4->SetLineColor(1);
  cos4->SetLineWidth(1);
  ctf4->SetLineWidth(1);
  rs4->SetLineWidth(1);
  
  
  ctf6->GetYaxis()->SetTitleSize(0.06);
  cos6->GetYaxis()->SetTitleSize(0.06);
  rs6->GetYaxis()->SetTitleSize(0.06);
  ctf6->GetYaxis()->SetTitleOffset(1.05);
  cos6->GetYaxis()->SetTitleOffset(1.05);
  rs6->GetYaxis()->SetTitleOffset(1.05);
  ctf6->GetXaxis()->SetTitleSize(0.06);
  cos6->GetXaxis()->SetTitleSize(0.06);
  rs6->GetXaxis()->SetTitleSize(0.06);
  ctf6->GetXaxis()->SetTitleOffset(0.85);
  cos6->GetXaxis()->SetTitleOffset(0.85);
  rs6->GetXaxis()->SetTitleOffset(0.85);


  canvas->cd(1);
  if(logx) gPad->SetLogx();
  if(logy) gPad->SetLogy();
  setStats(cos1,ctf1, rs1, startingY, startingX, fit);
  cos1->Draw("p");
  ctf1->Draw("psames");
  rs1->Draw("psames");


  canvas->cd(2);
  if(logx) gPad->SetLogx();
  if(logy) gPad->SetLogy();
  setStats(cos2,ctf2, rs2, startingY, startingX, fit);
  cos2->Draw("p");
  ctf2->Draw("psames");
  rs2->Draw("psames");


  canvas->cd(3);
  if(logx) gPad->SetLogx();
  if(logy) gPad->SetLogy();
  setStats(cos3,ctf3, rs3, startingY, startingX, fit);
  cos3->Draw("p");
  ctf3->Draw("psames");
  rs3->Draw("psames");

  canvas->cd(4);
  if(logx) gPad->SetLogx();
  if(logy) gPad->SetLogy();
  setStats(cos4,ctf4, rs4, startingY, startingX, fit);
  cos4->Draw("p");
  ctf4->Draw("psames");
  rs4->Draw("psames");


 canvas->cd(5);
  if(logx) gPad->SetLogx();
  if(logy) gPad->SetLogy();
  setStats(cos5,ctf5, rs5, startingY, startingX, fit);
  cos5->Draw("p");
  ctf5->Draw("psames");
  rs5->Draw("psames");

  /*
canvas->cd(6);
  if(logx) gPad->SetLogx();
  if(logy) gPad->SetLogy();
  setStats(cos6,ctf6, rs6, startingY, startingX, fit);
  cos6->Draw("p");
  ctf6->Draw("psames");
  rs6->Draw("psames");
  */
}

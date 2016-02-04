void RecoMuonValHistoPublisher(char* newFile="NEW_FILE",char* refFile="REF_FILE")
{
  //gROOT->ProcessLine(".x HistoCompare_Tracks.C");
 gROOT ->Reset();
 gROOT ->SetBatch();

 //=========  settings ====================

 char* dataType = "DATATYPE";

 gROOT->SetStyle("Plain");
 gStyle->SetPadGridX(kTRUE);
 gStyle->SetPadGridY(kTRUE);
 gStyle->SetPadRightMargin(0.07);
 gStyle->SetPadLeftMargin(0.13);
 //gStyle->SetTitleXSize(0.07); 
 //gStyle->SetTitleXOffset(0.6); 
 //tyle->SetTitleYSize(0.3);
 //gStyle->SetLabelSize(0.6) 
 //gStyle->SetTextSize(0.5);
 char* refLabel("REF_LABEL, REF_RELEASE REFSELECTION");
 char* newLabel("NEW_LABEL, NEW_RELEASE NEWSELECTION");

 Float_t maxPT=1500.;

 char* fastSim = "IS_FSIM";

 //=============================================


 delete gROOT->GetListOfFiles()->FindObject(refFile);
 delete gROOT->GetListOfFiles()->FindObject(newFile); 

//INPUT FILE IS DEFINED HERE, passed from python wrapper
 TText* te = new TText();
 TFile * sfile = new TFile(newFile);
 TDirectory * sdir=gDirectory;
 TFile * rfile = new TFile(refFile);
 TDirectory * rdir=gDirectory;

//check datatype
 if (dataType == "RECO") {
   if(sfile->cd("DQMData/Run 1/Muons/Run summary")) {;}
   else {
     cout << " Muon Histos for " << dataType << " not found" << endl;
     return;
   }
 }
 else {
   cout << " Data type " << dataType << " not allowed: only RECO is considered" << endl;
   return;
 }
 sdir=gDirectory;
 TIter nextkey( sdir->GetListOfKeys() );
 TList *sl = new TList();
 TKey *key, *oldkey=0;
 cout << "- New DQM muon reco collections: " << endl;
 while ( key = (TKey*)nextkey() ) {
   TObject *obj = key->ReadObj();
   if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
     TString theName = obj->GetName();
     if (theName.Contains("RecoMuonV")) {
       cout << " -> " << theName << endl;
       sl->Add(obj);
     }
   }
 }
 if (sl->GetSize()>0) {
   TString collname2 =sl->At(0)->GetName(); 
 }
 else {
   cout << " No DQM muon reco histos found in NEW file " << endl;
   return;
 }
 
 if (dataType == "RECO") {
   if(rfile->cd("DQMData/Run 1/Muons/Run summary")) {;}
   else {
     cout << " Muon Histos for " << dataType << " not found" << endl;
     return;
   }
 }
 rdir=gDirectory;
 TIter nextkeyr( rdir->GetListOfKeys() );
 TList *rl = new TList();
 TKey *keyr, *oldkeyr=0;
 cout << "- Ref DQM muon reco collections: " << endl;
 while ( keyr = (TKey*)nextkeyr() ) {
   TObject *obj = keyr->ReadObj();
   if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
     TString theName = obj->GetName();
     if (theName.Contains("RecoMuonV")) {
       cout << " -> " << theName << endl;
       rl->Add(obj);
     }
   }
 }
 if (rl->GetSize()>0) {
   TString collname1=rl->At(0)->GetName();
  }
 else {
   cout << " No DQM muon reco histos found in REF file " << endl;
   return;
 }

 TCanvas *canvas;
 
 TH1F *sh1,*rh1;
 TH1F *sh2,*rh2;
 TH1F *sh3,*rh3;
 TH1F *sh4,*rh4;
 TH1F *sh5,*rh5;
 TH1F *sh6,*rh6;
 
 TIter iter_r( rl );
 TIter iter_s( sl );
 TKey * myKey1, *myKey2;
 while ( (myKey1 = (TKey*)iter_r()) ) {
   TString myName = myKey1->GetName();
   collname1 = myName;
   myKey2 = (TKey*)iter_s();
   if (!myKey2) continue;
   collname2 = myKey2->GetName();
      if ( (collname1 != collname2) && (collname1+"FS" != collname2) && (collname1 != collname2+"FS") ) {
     cout << " Different collection names, please check: " << collname1 << " : " << collname2 << endl;
     continue;
   }

   TString newDir("NEW_RELEASE/NEWSELECTION/NEW_LABEL/");
   newDir+=myName;
   gSystem->mkdir(newDir,kTRUE);
 
   //rh1 = 0;
   //sh1 = 0;
   //===== For each monitored type as defined in muonValidation_cff.py 
 
   //===== reco muon distributions: GLB
   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Glb"+fastSim+"/ErrPt",rh1);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Glb"+fastSim+"/ErrPt",sh1);
 //  if(! rh1 && sh1) continue;
   rh1->GetYaxis()->SetTitle("GlobalMuon(GLB) #Delta p_{T}/p_{T}");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Glb"+fastSim+"/ErrP",rh2);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Glb"+fastSim+"/ErrP",sh2);
   rh2->GetYaxis()->SetTitle("GlobalMuon(GLB) #Delta p/p");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Glb"+fastSim+"/ErrPt_vs_Eta_Sigma",rh3);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Glb"+fastSim+"/ErrPt_vs_Eta_Sigma",sh3);
   rh3->GetYaxis()->SetTitle("GlobalMuon(GLB) #Delta p_{T}/p_{T} vs #sigma(#eta)");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Glb"+fastSim+"/ErrPt_vs_Pt_Sigma",rh4);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Glb"+fastSim+"/ErrPt_vs_Pt_Sigma",sh4);
   rh4->GetYaxis()->SetTitle("GlobalMuon(GLB) #Delta p_{T}/p_{T} vs #sigma(p_{T})");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("RecHistosGlb","Distributions for GlobalMuons (GLB)",1000,1050);

   // Normalize to the same number of "new" events:
   NormalizeHistograms(rh1,sh1);
   NormalizeHistograms(rh2,sh2);

   plot4histos(canvas,
	       sh1,rh1,sh2,rh2,
	       sh3,rh3,sh4,rh4,
	       te,"UU",-1);

   canvas->cd();

   l = new TLegend(0.20,0.48,0.90,0.53);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/muonRecoGlb.pdf");   
   delete l;
   delete canvas;

   //==== efficiencies and fractions GLB
   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Glb"+fastSim+"/EffP",rh1);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Glb"+fastSim+"/EffP",sh1);
   //if(! rh1 && sh1) continue;
   rh1->GetYaxis()->SetTitle("GlobalMuon(GLB) #epsilon vs. p");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Glb"+fastSim+"/EffEta",rh2);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Glb"+fastSim+"/EffEta",sh2);
   rh2->GetYaxis()->SetTitle("GlobalMuon(GLB) #epsilon vs. #eta");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Glb"+fastSim+"/FractP",rh3);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Glb"+fastSim+"/FractP",sh3);
   rh3->GetYaxis()->SetTitle("GlobalMuon(GLB) fraction vs. p");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Glb"+fastSim+"/FractEta",rh4);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Glb"+fastSim+"/FractEta",sh4);
   rh4->GetYaxis()->SetTitle("GlobalMuon(GLB) fraction vs. #eta");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("RecEffHistosGlb","Distributions for GlobalMuons (GLB), efficiencies and fractions",1000,1050);

   plot4histos(canvas,
               sh1,rh1,sh2,rh2,
               sh3,rh3,sh4,rh4,
               te,"UU",-1);

   canvas->cd();

   l = new TLegend(0.20,0.48,0.90,0.53);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/muonRecoGlbEff.pdf");
   delete l;
   delete canvas;

   //===== reco muon distributions: GLBPF
   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_GlbPF"+fastSim+"/ErrPt",rh1);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_GlbPF"+fastSim+"/ErrPt",sh1);
  // if(! rh1 && sh1) continue;
   rh1->GetYaxis()->SetTitle("PFGlobalMuon(GLBPF) #Delta p_{T}/p_{T}");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_GlbPF"+fastSim+"/ErrP",rh2);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_GlbPF"+fastSim+"/ErrP",sh2);
   rh2->GetYaxis()->SetTitle("PFGlobalMuon(GLBPF) #Delta p/p");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_GlbPF"+fastSim+"/ErrPt_vs_Eta_Sigma",rh3);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_GlbPF"+fastSim+"/ErrPt_vs_Eta_Sigma",sh3);
   rh3->GetYaxis()->SetTitle("PFGlobalMuon(GLBPF) #Delta p_{T}/p_{T} vs #sigma(#eta)");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_GlbPF"+fastSim+"/ErrPt_vs_Pt_Sigma",rh4);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_GlbPF"+fastSim+"/ErrPt_vs_Pt_Sigma",sh4);
   rh4->GetYaxis()->SetTitle("PFGlobalMuon(GLBPF) #Delta p_{T}/p_{T} vs #sigma(p_{T})");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("RecHistosGlbPF","Distributions for PFGlobalMuons (GLBPF)",1000,1050);

   // Normalize to the same number of "new" events:
   NormalizeHistograms(rh1,sh1);
   NormalizeHistograms(rh2,sh2);

   plot4histos(canvas,
               sh1,rh1,sh2,rh2,
               sh3,rh3,sh4,rh4,
               te,"UU",-1);

   canvas->cd();

   l = new TLegend(0.20,0.48,0.90,0.53);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/muonRecoGlbPF.pdf");
   delete l;
   delete canvas;

   //==== efficiencies and fractions GLBPF
   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_GlbPF"+fastSim+"/EffP",rh1);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_GlbPF"+fastSim+"/EffP",sh1);
   //if(! rh1 && sh1) continue;
   rh1->GetYaxis()->SetTitle("PFGlobalMuon(GLBPF) #epsilon vs. p");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_GlbPF"+fastSim+"/EffEta",rh2);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_GlbPF"+fastSim+"/EffEta",sh2);
   rh2->GetYaxis()->SetTitle("PFGlobalMuon(GLBPF) #epsilon vs. #eta");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_GlbPF"+fastSim+"/FractP",rh3);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_GlbPF"+fastSim+"/FractP",sh3);
   rh3->GetYaxis()->SetTitle("PFGlobalMuon(GLBPF) fraction vs. p");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_GlbPF"+fastSim+"/FractEta",rh4);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_GlbPF"+fastSim+"/FractEta",sh4);
   rh4->GetYaxis()->SetTitle("PFGlobalMuon(GLBPF) fraction vs. #eta");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("RecEffHistosGlbPF","Distributions for PFGlobalMuons (GLBPF), efficiencies and fractions",1000,1050);

   plot4histos(canvas,
               sh1,rh1,sh2,rh2,
               sh3,rh3,sh4,rh4,
               te,"UU",-1);

   canvas->cd();

   l = new TLegend(0.20,0.48,0.90,0.53);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/muonRecoGlbPFEff.pdf");
   delete l;
   delete canvas;


   //===== reco muon distributions: STA
   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Sta"+fastSim+"/ErrPt",rh1);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Sta"+fastSim+"/ErrPt",sh1);
  // if(! rh1 && sh1) continue;
   rh1->GetYaxis()->SetTitle("StandAloneMuon(STA) #Delta p_{T}/p_{T}");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Sta"+fastSim+"/ErrP",rh2);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Sta"+fastSim+"/ErrP",sh2);
   rh2->GetYaxis()->SetTitle("StandAloneMuon(STA) #Delta p/p");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Sta"+fastSim+"/ErrPt_vs_Eta_Sigma",rh3);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Sta"+fastSim+"/ErrPt_vs_Eta_Sigma",sh3);
   rh3->GetYaxis()->SetTitle("StandAloneMuon(STA) #Delta p_{T}/p_{T} vs #sigma(#eta)");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Sta"+fastSim+"/ErrPt_vs_Pt_Sigma",rh4);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Sta"+fastSim+"/ErrPt_vs_Pt_Sigma",sh4);
   rh4->GetYaxis()->SetTitle("StandAloneMuon(STA) #Delta p_{T}/p_{T} vs #sigma(p_{T})");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("RecHistosSta","Distributions for StandAloneMuons (STA)",1000,1050);

   // Normalize to the same number of "new" events:
   NormalizeHistograms(rh1,sh1);
   NormalizeHistograms(rh2,sh2);

   plot4histos(canvas,
               sh1,rh1,sh2,rh2,
               sh3,rh3,sh4,rh4,
               te,"UU",-1);

   canvas->cd();

   l = new TLegend(0.20,0.48,0.90,0.53);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/muonRecoSta.pdf");
   delete l;
   delete canvas;

   //==== efficiencies and fractions STA
   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Sta"+fastSim+"/EffP",rh1);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Sta"+fastSim+"/EffP",sh1);
  // if(! rh1 && sh1) continue;
   rh1->GetYaxis()->SetTitle("StandAloneMuon(STA) #epsilon vs. p");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Sta"+fastSim+"/EffEta",rh2);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Sta"+fastSim+"/EffEta",sh2);
   rh2->GetYaxis()->SetTitle("StandAloneMuon(STA) #epsilon vs. #eta");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Sta"+fastSim+"/FractP",rh3);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Sta"+fastSim+"/FractP",sh3);
   rh3->GetYaxis()->SetTitle("StandAloneMuon(STA) fraction vs. p");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Sta"+fastSim+"/FractEta",rh4);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Sta"+fastSim+"/FractEta",sh4);
   rh4->GetYaxis()->SetTitle("StandAloneMuon(STA) fraction vs. #eta");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("RecEffHistosSta","Distributions for StandAloneMuons (STA), efficiencies and fractions",1000,1050);

   plot4histos(canvas,
               sh1,rh1,sh2,rh2,
               sh3,rh3,sh4,rh4,
               te,"UU",-1);

   canvas->cd();

   l = new TLegend(0.20,0.48,0.90,0.53);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/muonRecoStaEff.pdf");
   delete l;
   delete canvas;

   //===== reco muon distributions: TRK
   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Trk"+fastSim+"/ErrPt",rh1);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Trk"+fastSim+"/ErrPt",sh1);
  // if(! rh1 && sh1) continue;
   rh1->GetYaxis()->SetTitle("TrackerMuon(TRK) #Delta p_{T}/p_{T}");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Trk"+fastSim+"/ErrP",rh2);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Trk"+fastSim+"/ErrP",sh2);
   rh2->GetYaxis()->SetTitle("TrackerMuon(TRK) #Delta p/p");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Trk"+fastSim+"/ErrPt_vs_Eta_Sigma",rh3);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Trk"+fastSim+"/ErrPt_vs_Eta_Sigma",sh3);
   rh3->GetYaxis()->SetTitle("TrackerMuon(TRK) #Delta p_{T}/p_{T} vs #sigma(#eta)");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Trk"+fastSim+"/ErrPt_vs_Pt_Sigma",rh4);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Trk"+fastSim+"/ErrPt_vs_Pt_Sigma",sh4);
   rh4->GetYaxis()->SetTitle("TrackerMuon(TRK) #Delta p_{T}/p_{T} vs #sigma(p_{T})");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("RecHistosTrk","Distributions for TrackerMuons (TRK)",1000,1050);

   // Normalize to the same number of "new" events:
   NormalizeHistograms(rh1,sh1);
   NormalizeHistograms(rh2,sh2);

   plot4histos(canvas,
               sh1,rh1,sh2,rh2,
               sh3,rh3,sh4,rh4,
               te,"UU",-1);

   canvas->cd();

   l = new TLegend(0.20,0.48,0.90,0.53);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/muonRecoTrk.pdf");
   delete l;
   delete canvas;


   //==== efficiencies and fractions TRK
   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Trk"+fastSim+"/EffP",rh1);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Trk"+fastSim+"/EffP",sh1);
  // if(! rh1 && sh1) continue;
   rh1->GetYaxis()->SetTitle("TrackerMuon(TRK) #epsilon vs. p");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Trk"+fastSim+"/EffEta",rh2);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Trk"+fastSim+"/EffEta",sh2);
   rh2->GetYaxis()->SetTitle("TrackerMuon(TRK) #epsilon vs. #eta");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Trk"+fastSim+"/FractP",rh3);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Trk"+fastSim+"/FractP",sh3);
   rh3->GetYaxis()->SetTitle("TrackerMuon(TRK) fraction vs. p");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Trk"+fastSim+"/FractEta",rh4);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Trk"+fastSim+"/FractEta",sh4);
   rh4->GetYaxis()->SetTitle("TrackerMuon(TRK) fraction vs. #eta");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("RecEffHistosTrk","Distributions for TrackerMuons (TRK), efficiencies and fractions",1000,1050);

   plot4histos(canvas,
               sh1,rh1,sh2,rh2,
               sh3,rh3,sh4,rh4,
               te,"UU",-1);

   canvas->cd();

   l = new TLegend(0.20,0.48,0.90,0.53);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/muonRecoTrkEff.pdf");
   delete l;
   delete canvas;

   //
   //===== reco muon distributions: Tight Muons
   //
   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Tgt"+fastSim+"/ErrPt",rh1);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Tgt"+fastSim+"/ErrPt",sh1);
  // if(! rh1 && sh1) continue;
   rh1->GetYaxis()->SetTitle("Tight Muon #Delta p_{T}/p_{T}");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Tgt"+fastSim+"/ErrP",rh2);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Tgt"+fastSim+"/ErrP",sh2);
   rh2->GetYaxis()->SetTitle("Tight Muon #Delta p/p");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Tgt"+fastSim+"/ErrPt_vs_Eta_Sigma",rh3);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Tgt"+fastSim+"/ErrPt_vs_Eta_Sigma",sh3);
   rh3->GetYaxis()->SetTitle("Tight Muon #Delta p_{T}/p_{T} vs #sigma(#eta)");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Tgt"+fastSim+"/ErrPt_vs_Pt_Sigma",rh4);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Tgt"+fastSim+"/ErrPt_vs_Pt_Sigma",sh4);
   rh4->GetYaxis()->SetTitle("Tigh Muon) #Delta p_{T}/p_{T} vs #sigma(p_{T})");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("RecHistosTgt","Distributions for Tight Muons",1000,1050);

   // Normalize to the same number of "new" events:
   NormalizeHistograms(rh1,sh1);
   NormalizeHistograms(rh2,sh2);

   plot4histos(canvas,
               sh1,rh1,sh2,rh2,
               sh3,rh3,sh4,rh4,
               te,"UU",-1);

   canvas->cd();

   l = new TLegend(0.20,0.48,0.90,0.53);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/muonRecoTgt.pdf");
   delete l;
   delete canvas;


   //==== efficiencies and fractions Tight Muons
   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Tgt"+fastSim+"/EffP",rh1);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Tgt"+fastSim+"/EffP",sh1);
  // if(! rh1 && sh1) continue;
   rh1->GetYaxis()->SetTitle("Tight Muon #epsilon vs. p");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Tgt"+fastSim+"/EffEta",rh2);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Tgt"+fastSim+"/EffEta",sh2);
   rh2->GetYaxis()->SetTitle("Tight Muon #epsilon vs. #eta");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Tgt"+fastSim+"/FractP",rh3);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Tgt"+fastSim+"/FractP",sh3);
   rh3->GetYaxis()->SetTitle("Tight Muon fraction vs. p");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/RecoMuon_MuonAssoc_Tgt"+fastSim+"/FractEta",rh4);
   sdir->GetObject(collname2+"/RecoMuon_MuonAssoc_Tgt"+fastSim+"/FractEta",sh4);
   rh4->GetYaxis()->SetTitle("Tight Muon fraction vs. #eta");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("RecEffHistosTgt","Distributions for Tight Muons, efficiencies and fractions",1000,1050);

   plot4histos(canvas,
               sh1,rh1,sh2,rh2,
               sh3,rh3,sh4,rh4,
               te,"UU",-1);

   canvas->cd();

   l = new TLegend(0.20,0.48,0.90,0.53);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/muonRecoTgtEff.pdf");
   delete l;
   delete canvas;


   //
   // Merge pdf histograms together into larger files, and name them based on the collection names
   //
 gSystem->Exec("gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=merged.pdf "
	       +newDir+"/muonRecoGlb.pdf "
	       +newDir+"/muonRecoGlbEff.pdf "
               +newDir+"/muonRecoGlbPF.pdf "
               +newDir+"/muonRecoGlbPFEff.pdf "
               +newDir+"/muonRecoSta.pdf "
               +newDir+"/muonRecoStaEff.pdf "
               +newDir+"/muonRecoTrk.pdf "
               +newDir+"/muonRecoTrkEff.pdf "
               +newDir+"/muonRecoTgt.pdf "
               +newDir+"/muonRecoTgtEff.pdf ");
 gSystem->Exec("mv merged.pdf "+newDir+"/../"+myName+".pdf");
 gSystem->Exec("rm -r "+newDir);
 
 }  // end of "while loop"
 
}


void NormalizeHistograms(TH1F* h1, TH1F* h2)
{
  if (h1==0 || h2==0) return;
  float scale2 = -9999.9;

  if ( h1->Integral() != 0 && h2->Integral() != 0 ){
      scale2 = (float)h1->Integral()/(float)h2->Integral();
    
      //h1->Sumw2();
      //h2->Sumw2();
      h2->Scale(scale2);
    }
}

/*in case histogram ranges are very different*/
void NormalizeHistogramsPeakRegion(TH1F* h1, TH1F* h2)
{
  if (h1==0 || h2==0) return;
  float scale2 = -9999.9;
  //identify the peak region
  int bin_start, bin_end;
  int bin_tot = 0;
  bin_tot = (h1->GetXaxis()->GetNbins() > h2->GetXaxis()->GetNbins()) ? h1->GetXaxis()->GetNbins() : h2->GetXaxis()->GetNbins();
  bin_start = 0.1*bin_tot;
  bin_end = 0.2*bin_tot; 

  if ( h1->Integral(bin_start,bin_end) != 0 && h2->Integral(bin_start,bin_end) != 0 ){
      scale2 = (float)h1->Integral(bin_start,bin_end)/(float)h2->Integral(bin_start,bin_end);

      //h1->Sumw2();
      //h2->Sumw2();
      h2->Scale(scale2);
    }
}

void plot4histos(TCanvas *canvas, 
		TH1F *s1,TH1F *r1, TH1F *s2,TH1F *r2, 
		TH1F *s3,TH1F *r3, TH1F *s4,TH1F *r4,
		TText* te,
	       char * option, double startingY, double startingX = .1,bool fit = false){
  canvas->Divide(2,2);

  s1->SetMarkerStyle(20);
  r1->SetMarkerStyle(21);
  s1->SetMarkerColor(2);
  r1->SetMarkerColor(4);
  s1->SetMarkerSize(0.7);
  r1->SetMarkerSize(0.7);
  s1->SetLineColor(2);
  r1->SetLineColor(4);
  s1->SetLineWidth(2);
  r1->SetLineWidth(2);

  s2->SetMarkerStyle(20);
  r2->SetMarkerStyle(21);
  s2->SetMarkerColor(2);
  r2->SetMarkerColor(4);
  s2->SetMarkerSize(0.1);
  r2->SetMarkerSize(0.1);
  s2->SetLineColor(2);
  r2->SetLineColor(4);
  s2->SetLineWidth(2);
  r2->SetLineWidth(2);

  s3->SetMarkerStyle(20);
  r3->SetMarkerStyle(21);
  s3->SetMarkerColor(2);
  r3->SetMarkerColor(4);
  s3->SetMarkerSize(0.7);
  r3->SetMarkerSize(0.7);
  s3->SetLineColor(2);
  r3->SetLineColor(4);
  r3->SetLineWidth(2);
  s3->SetLineWidth(2);

  s4->SetMarkerStyle(20);
  r4->SetMarkerStyle(21);
  s4->SetMarkerColor(2);
  r4->SetMarkerColor(4);
  s4->SetMarkerSize(0.7);
  r4->SetMarkerSize(0.7);
  s4->SetLineColor(2);
  r4->SetLineColor(4);
  r4->SetLineWidth(2);
  s4->SetLineWidth(2);


  //setStats(r1,s1, startingY, startingX, fit);
  canvas->cd(1);
  setStats(s1,r1, -1, 0, false);
  r1->Draw();
  s1->Draw("sames");

  canvas->cd(2);
  setStats(s2,r2, -1, 0, false);
  r2->Draw();
  s2->Draw("sames");

  canvas->cd(3);
  setStats(s3,r3, -1, 0, false);
  r3->Draw();
  s3->Draw("sames");

  canvas->cd(4);
  setStats(s4,r4, -1, 0, false);
  s4->Draw();
  r4->Draw("sames");

}

void plot6histos(TCanvas *canvas, 
		TH1F *s1,TH1F *r1, TH1F *s2,TH1F *r2, 
		TH1F *s3,TH1F *r3, TH1F *s4,TH1F *r4,
		TH1F *s5,TH1F *r5, TH1F *s6,TH1F *r6,
		TText* te,
	       char * option, double startingY, double startingX = .1,bool fit = false){
  canvas->Divide(2,3);

  s1->SetMarkerStyle(20);
  r1->SetMarkerStyle(21);
  s1->SetMarkerColor(2);
  r1->SetMarkerColor(4);
  s1->SetMarkerSize(0.7);
  r1->SetMarkerSize(0.7);
  s1->SetLineColor(2);
  r1->SetLineColor(4);
  s1->SetLineWidth(2);
  r1->SetLineWidth(2);

  s2->SetMarkerStyle(20);
  r2->SetMarkerStyle(21);
  s2->SetMarkerColor(2);
  r2->SetMarkerColor(4);
  s2->SetMarkerSize(0.1);
  r2->SetMarkerSize(0.1);
  s2->SetLineColor(2);
  r2->SetLineColor(4);
  s2->SetLineWidth(2);
  r2->SetLineWidth(2);

  s3->SetMarkerStyle(20);
  r3->SetMarkerStyle(21);
  s3->SetMarkerColor(2);
  r3->SetMarkerColor(4);
  s3->SetMarkerSize(0.7);
  r3->SetMarkerSize(0.7);
  s3->SetLineColor(2);
  r3->SetLineColor(4);
  r3->SetLineWidth(2);
  s3->SetLineWidth(2);

  s4->SetMarkerStyle(20);
  r4->SetMarkerStyle(21);
  s4->SetMarkerColor(2);
  r4->SetMarkerColor(4);
  s4->SetMarkerSize(0.7);
  r4->SetMarkerSize(0.7);
  s4->SetLineColor(2);
  r4->SetLineColor(4);
  r4->SetLineWidth(2);
  s4->SetLineWidth(2);

  s5->SetMarkerStyle(20);
  r5->SetMarkerStyle(21);
  s5->SetMarkerColor(2);
  r5->SetMarkerColor(4);
  s5->SetMarkerSize(0.7);
  r5->SetMarkerSize(0.7);
  s5->SetLineColor(2);
  r5->SetLineColor(4);
  r5->SetLineWidth(2);
  s5->SetLineWidth(2);

  s6->SetMarkerStyle(20);
  r6->SetMarkerStyle(21);
  s6->SetMarkerColor(2);
  r6->SetMarkerColor(4);
  s6->SetMarkerSize(0.7);
  r6->SetMarkerSize(0.7);
  s6->SetLineColor(2);
  r6->SetLineColor(4);
  r6->SetLineWidth(2);
  s6->SetLineWidth(2);


  //setStats(r1,s1, startingY, startingX, fit);
  canvas->cd(1);
  setStats(s1,r1, -1, 0, false);
  r1->Draw();
  s1->Draw("sames");

  canvas->cd(2);
  gPad->SetLogy(); 
  setStats(s2,r2, -1, 0, false);
  s2->Draw();
  r2->Draw("sames");

  canvas->cd(3);
  setStats(s3,r3, -1, 0, false);
  r3->Draw();
  s3->Draw("sames");

  canvas->cd(4);
  gPad->SetLogy(); 
  setStats(s4,r4, -1, 0, false);
  s4->Draw();
  r4->Draw("sames");

  canvas->cd(5);
  setStats(s5,r5, -1, 0, false);
  r5->Draw();
  s5->Draw("sames");

  canvas->cd(6);
  gPad->SetLogy(); 
  setStats(s6,r6, -1, 0, false);
  r6->Draw();
  s6->Draw("sames");

}



void setStats(TH1* s,TH1* r, double startingY, double startingX = .1,bool fit){
  if (startingY<0){
    s->SetStats(0);
    r->SetStats(0);
  } else {
    //gStyle->SetOptStat(1001);

    if (fit){
      s->Fit("gaus");
      TF1* f1 = (TF1*) s->GetListOfFunctions()->FindObject("gaus");
      if (f1) {
	f1->SetLineColor(2);
	f1->SetLineWidth(1);
      }
    }
    s->Draw();
    gPad->Update(); 
    TPaveStats* st1 = (TPaveStats*) s->GetListOfFunctions()->FindObject("stats");
    if (st1) {
      if (fit) {st1->SetOptFit(0010);    st1->SetOptStat(1001);}
      st1->SetX1NDC(startingX);
      st1->SetX2NDC(startingX+0.30);
      st1->SetY1NDC(startingY+0.20);
      st1->SetY2NDC(startingY+0.35);
      st1->SetTextColor(2);
    }
    else s->SetStats(0);
    if (fit) {
      r->Fit("gaus");
      TF1* f2 = (TF1*) r->GetListOfFunctions()->FindObject("gaus");
      if (f2) {
	f2->SetLineColor(4);
	f2->SetLineWidth(1);
      }
    }
    r->Draw();
    gPad->Update(); 
    TPaveStats* st2 = (TPaveStats*) r->GetListOfFunctions()->FindObject("stats");
    if (st2) {
      if (fit) {st2->SetOptFit(0010);    st2->SetOptStat(1001);}
      st2->SetX1NDC(startingX);
      st2->SetX2NDC(startingX+0.30);
      st2->SetY1NDC(startingY);
      st2->SetY2NDC(startingY+0.15);
      st2->SetTextColor(4);
    }
    else r->SetStats(0);
  }
}

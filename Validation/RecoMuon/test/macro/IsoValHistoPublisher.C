void IsoValHistoPublisher(char* newFile="NEW_FILE",char* refFile="REF_FILE")
{
  //gROOT->ProcessLine(".x HistoCompare_Tracks.C");
 gROOT ->Reset();
 gROOT ->SetBatch();

 //=========  settings ====================

 char* dataType = "RECO";

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


 //=============================================


 delete gROOT->GetListOfFiles()->FindObject(refFile);
 delete gROOT->GetListOfFiles()->FindObject(newFile); 

 TText* te = new TText();
 TFile * sfile = new TFile(newFile);
 TDirectory * sdir=gDirectory;
 TFile * rfile = new TFile(refFile);
 TDirectory * rdir=gDirectory;

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
     if (theName.Contains("MuonIsolationV_inc")) {
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
     if (theName.Contains("MuonIsolationV_inc")) {
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


 // Get the number of events for the normalization
 // Not needed for muonisolation validation: already handled in its postprocessor

 /* TH1F *sevt, *revt;
    sdir->GetObject("RecoMuonV/RecoMuon_TrackAssoc/Muons/NMuon",sevt);
    rdir->GetObject("RecoMuonV/RecoMuon_TrackAssoc/Muons/NMuon",revt);
    Double_t snorm = 1.;
    if (sevt && revt) {
    if (revt->GetEntries()>0) snorm = sevt->GetEntries()/revt->GetEntries();
    }
    else {  cout << " *** Missing normalization histos"; }
 */

 TCanvas *canvas;
 
 TH1F *sh1,*rh1;
 TH1F *sh2,*rh2;
 TH1F *sh3,*rh3;
 TH1F *sh4,*rh4;

 TProfile *sp1,*rp1;
 TProfile *sp2,*rp2;
 TProfile *sp3,*rp3;
 TProfile *sp4,*rp4;
 
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
 
   rh1 = 0;
   sh1 = 0;

   rp1 = 0;
   sp1 = 0;
 
   //===== Tracker and ECAL Deposits
   rdir->GetObject(collname1+"/sumPt",rh1);
   sdir->GetObject(collname2+"/sumPt",sh1);
   if(! rh1 && sh1) continue;
   //   rh1->GetYaxis()->SetTitle("GlobalMuon(GLB) #eta");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);
   rdir->GetObject(collname1+"/emEt",rh2);
   sdir->GetObject(collname2+"/emEt",sh2);
   //   rh2->GetYaxis()->SetTitle("GlobalMuon(GLB) #eta");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/sumPt_cd",rh3);
   sdir->GetObject(collname2+"/sumPt_cd",sh3);
   //   rh3->GetYaxis()->SetTitle("GlobalMuon(GLB) pT");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rdir->GetObject(collname1+"/emEt_cd",rh4);
   sdir->GetObject(collname2+"/emEt_cd",sh4);
   //   rh4->GetYaxis()->SetTitle("GlobalMuon(GLB) #chi^{2}/ndf");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("IsoHistos1","Tracker, ECAL Depostis",1000,1050);

   // Normalize to the same number of "new" events:
   /*   NormalizeHistograms(rh1,snorm);
	NormalizeHistograms(rh2,snorm);
	NormalizeHistograms(rh3,snorm);
	NormalizeHistograms(rh4,snorm);
   */

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
   canvas->Print(newDir+"/muonIso1.pdf");   
   delete l;
   delete canvas;

   //===== HCAL and HO Isolation Distributions
   rdir->GetObject(collname1+"/hadEt",rh1);
   sdir->GetObject(collname2+"/hadEt",sh1);
   if(! rh1 && sh1) continue;
   //   rh1->GetYaxis()->SetTitle("GlobalMuon(STA) #eta");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);
   rdir->GetObject(collname1+"/hoEt",rh2);
   sdir->GetObject(collname2+"/hoEt",sh2);
   //   rh2->GetYaxis()->SetTitle("GlobalMuon(STA) #eta");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/hadEt_cd",rh3);
   sdir->GetObject(collname2+"/hadEt_cd",sh3);
   //   rh3->GetYaxis()->SetTitle("GlobalMuon(STA) pT");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rdir->GetObject(collname1+"/hoEt_cd",rh4);
   sdir->GetObject(collname2+"/hoEt_cd",sh4);
   // rh4->GetYaxis()->SetTitle("GlobalMuon(STA) #chi^{2}/ndf");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("IsoHistos2","HCAL, HO Deposits",1000,1050);

   // Normalize to the same number of "new" events:
   /*   NormalizeHistograms(rh1,snorm);
	NormalizeHistograms(rh2,snorm);
	NormalizeHistograms(rh3,snorm);
	NormalizeHistograms(rh4,snorm);
   */

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
   canvas->Print(newDir+"/muonIso2.pdf");   
   delete l;
   delete canvas;


   //===== N_Tracks, N_Jets around #mu
   rdir->GetObject(collname1+"/nTracks",rh1);
   sdir->GetObject(collname2+"/nTracks",sh1);
   if(! rh1 && sh1) continue;
   //   rh1->GetYaxis()->SetTitle("GlobalMuon(TK) #eta");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);
   rdir->GetObject(collname1+"/nJets",rh2);
   sdir->GetObject(collname2+"/nJets",sh2);
   //   rh2->GetYaxis()->SetTitle("GlobalMuon(TK) #eta");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/nTracks_cd",rh3);
   sdir->GetObject(collname2+"/nTracks_cd",sh3);
   //   rh3->GetYaxis()->SetTitle("GlobalMuon(TK) pT");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rdir->GetObject(collname1+"/nJets_cd",rh4);
   sdir->GetObject(collname2+"/nJets_cd",sh4);
   //   rh4->GetYaxis()->SetTitle("GlobalMuon(TK) #chi^{2}/ndf");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("IsoHistos3","Number of tracks, jets around #mu",1000,1050);

   // Normalize to the same number of "new" events:
   /*   NormalizeHistograms(rh1,snorm);
	NormalizeHistograms(rh2,snorm);
	NormalizeHistograms(rh3,snorm);
	NormalizeHistograms(rh4,snorm);
   */

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
   canvas->Print(newDir+"/muonIso3.pdf");   
   delete l;
   delete canvas;

   //===== avg Pt, weighted Et around #mu
   rdir->GetObject(collname1+"/avgPt",rh1);
   sdir->GetObject(collname2+"/avgPt",sh1);
   if(! rh1 && sh1) continue;
   //   rh1->GetYaxis()->SetTitle("StaMuon #eta");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);
   rdir->GetObject(collname1+"/weightedEt",rh2);
   sdir->GetObject(collname2+"/weightedEt",sh2);
   ///   rh2->GetYaxis()->SetTitle("StaMuon #eta");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/avgPt_cd",rh3);
   sdir->GetObject(collname2+"/avgPt_cd",sh3);
   //   rh3->GetYaxis()->SetTitle("StaMuon pT");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rdir->GetObject(collname1+"/weightedEt_cd",rh4);
   sdir->GetObject(collname2+"/weightedEt_cd",sh4);
   //   rh4->GetYaxis()->SetTitle("StaMuon #chi^{2}/ndf");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("IsoHistos4","Average p_{T}, weighted E_{T} aroun #mu",1000,1050);

   // Normalize to the same number of "new" events:
   /*   NormalizeHistograms(rh1,snorm);
	NormalizeHistograms(rh2,snorm);
	NormalizeHistograms(rh3,snorm);
	NormalizeHistograms(rh4,snorm);
   */

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
   canvas->Print(newDir+"/muonIso4.pdf");   
   delete l;
   delete canvas;


   //===== Trakcer and CAL deposits vs muon pT
   rdir->GetObject(collname1+"/muonPt_sumPt",rp1);
   sdir->GetObject(collname2+"/muonPt_sumPt",sp1);
   if(! rp1 && sp1) continue;
   //   rp1->GetYaxis()->SetTitle("TkMuon #eta");
   rp1->GetYaxis()->SetTitleSize(0.05);
   rp1->GetYaxis()->SetTitleOffset(1.2);
   rdir->GetObject(collname1+"/muonPt_emEt",rp2);
   sdir->GetObject(collname2+"/muonPt_emEt",sp2);
   //   rp2->GetYaxis()->SetTitle("TkMuon #eta");
   rp2->GetYaxis()->SetTitleSize(0.05);
   rp2->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/muonPt_hadEt",rp3);
   sdir->GetObject(collname2+"/muonPt_hadEt",sp3);
   //   rp3->GetYaxis()->SetTitle("TkMuon pT");
   rp3->GetYaxis()->SetTitleSize(0.05);
   rp3->GetYaxis()->SetTitleOffset(1.2);
   rdir->GetObject(collname1+"/muonPt_hoEt",rp4);
   sdir->GetObject(collname2+"/muonPt_hoEt",sp4);
   //   rp4->GetYaxis()->SetTitle("TkMuon #chi^{2}/ndf");
   rp4->GetYaxis()->SetTitleSize(0.05);
   rp4->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("IsoHistos5","Trk, CAL Isolations vs. #mu p_{T}",1000,1050);

   // Normalize to the same number of "new" events:
   /*   NormalizeHistograms(rh1,snorm);
	NormalizeHistograms(rh2,snorm);
	NormalizeHistograms(rh3,snorm);
	NormalizeHistograms(rh4,snorm);
   */

   plot4histos(canvas,
	       sp1,rp1,sp2,rp2,
	       sp3,rp3,sp4,rp4,
	       te,"UU",-1);

   canvas->cd();

   l = new TLegend(0.20,0.48,0.90,0.53);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rp1,refLabel,"LPF");
   l->AddEntry(sp1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/muonIso5.pdf");   
   delete l;
   delete canvas;

   //===== NTracks, NJets, avgPt, weightedEt vs Muon pT
   rdir->GetObject(collname1+"/muonPt_nTracks",rp1);
   sdir->GetObject(collname2+"/muonPt_nTracks",sp1);
   if(! rp1 && sp1) continue;
   //   rp1->GetYaxis()->SetTitle("TkMuon #eta");
   rp1->GetYaxis()->SetTitleSize(0.05);
   rp1->GetYaxis()->SetTitleOffset(1.2);
   rdir->GetObject(collname1+"/muonPt_nJets",rp2);
   sdir->GetObject(collname2+"/muonPt_nJets",sp2);
   //   rp2->GetYaxis()->SetTitle("TkMuon #eta");
   rp2->GetYaxis()->SetTitleSize(0.05);
   rp2->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/muonPt_avgPt",rp3);
   sdir->GetObject(collname2+"/muonPt_avgPt",sp3);
   //   rp3->GetYaxis()->SetTitle("TkMuon pT");
   rp3->GetYaxis()->SetTitleSize(0.05);
   rp3->GetYaxis()->SetTitleOffset(1.2);
   rdir->GetObject(collname1+"/muonPt_weightedEt",rp4);
   sdir->GetObject(collname2+"/muonPt_weightedEt",sp4);
   //   rp4->GetYaxis()->SetTitle("TkMuon #chi^{2}/ndf");
   rp4->GetYaxis()->SetTitleSize(0.05);
   rp4->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("IsoHistos6","Other stuff vs #mu p_{T}",1000,1050);

   // Normalize to the same number of "new" events:
   /*   NormalizeHistograms(rh1,snorm);
        NormalizeHistograms(rh2,snorm);
        NormalizeHistograms(rh3,snorm);
        NormalizeHistograms(rh4,snorm);
   */

   plot4histos(canvas,
               sp1,rp1,sp2,rp2,
               sp3,rp3,sp4,rp4,
               te,"UU",-1);

   canvas->cd();

   l = new TLegend(0.20,0.48,0.90,0.53);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rp1,refLabel,"LPF");
   l->AddEntry(sp1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/muonIso6.pdf");
   delete l;
   delete canvas;


 
 //// Merge pdf histograms together into larger files, and name them based on the collection names
 gSystem->Exec("gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=merged_iso.pdf "
	       +newDir+"/muonIso1.pdf "
	       +newDir+"/muonIso2.pdf "
	       +newDir+"/muonIso3.pdf "
	       +newDir+"/muonIso4.pdf "
	       +newDir+"/muonIso5.pdf "
	       +newDir+"/muonIso6.pdf ");
 gSystem->Exec("mv merged_iso.pdf "+newDir+"/../"+myName+".pdf");
 gSystem->Exec("rm -r "+newDir);
 
 }  // end of "while loop"
 
}


void NormalizeHistograms(TH1F* h1, TH1F* h2)
{
  if (h1==0 || h2==0) return;
  float scale1 = -9999.9;
  float scale2 = -9999.9;

  if ( h1->Integral() != 0 && h2->Integral() != 0 ){
      scale1 = 1.0/(float)h1->Integral();
      scale2 = 1.0/(float)h2->Integral();
    
      //h1->Sumw2();
      //h2->Sumw2();
      h1->Scale(scale1);
      h2->Scale(scale2);
    }
}

void NormalizeHistograms(TH1F* h1, Double_t nrm)
{
  if (h1==0) return;
  h1->Scale(nrm);
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
  s2->Draw();
  r2->Draw("sames");

  canvas->cd(3);
  setStats(s3,r3, -1, 0, false);
  r3->Draw();
  s3->Draw("sames");

  canvas->cd(4);
  setStats(s4,r4, -1, 0, false);
  s4->Draw();
  r4->Draw("sames");

}

void plot4histos(TCanvas *canvas,
		 TProfile *s1,TProfile *r1, TProfile *s2,TProfile *r2,
		 TProfile *s3,TProfile *r3, TProfile *s4,TProfile *r4,
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
  s2->Draw();
  r2->Draw("sames");

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

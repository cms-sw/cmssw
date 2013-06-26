#include <string>


void TrackValHistoPublisher(char* newFile="NEW_FILE",char* refFile="REF_FILE")
{
  //gROOT->ProcessLine(".x HistoCompare_Tracks.C");
 gROOT ->Reset();
 gROOT ->SetBatch();

 //=========  settings ====================
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


 //=============================================


 delete gROOT->GetListOfFiles()->FindObject(refFile);
 delete gROOT->GetListOfFiles()->FindObject(newFile); 

 TText* te = new TText();
 TFile * sfile = new TFile(newFile);
 TDirectory * sdir=gDirectory;
 TFile * rfile = new TFile(refFile);
 TDirectory * rdir=gDirectory;

 if(sfile->GetDirectory("DQMData/Run 1/RecoTrackV")) sfile->cd("DQMData/Run 1/RecoTrackV/Run summary/Track");
 else if(sfile->GetDirectory("DQMData/RecoTrackV/Track"))sfile->cd("DQMData/RecoTrackV/Track");
 else if(sfile->GetDirectory("DQMData/Run 1/Tracking")) sfile->cd("DQMData/Run 1/Tracking/Run summary/Track");
 else if(sfile->GetDirectory("DQMData/Tracking/Track"))sfile->cd("DQMData/Tracking/Track");
 sdir=gDirectory;
 TList *sl= sdir->GetListOfKeys();
 TString collname2 =sl->At(0)->GetName(); 

 if(rfile->GetDirectory("DQMData/Run 1/RecoTrackV")) rfile->cd("DQMData/Run 1/RecoTrackV/Run summary/Track");
 else if(rfile->GetDirectory("DQMData/RecoTrackV/Track"))rfile->cd("DQMData/RecoTrackV/Track");
 else if(rfile->GetDirectory("DQMData/Run 1/Tracking")) rfile->cd("DQMData/Run 1/Tracking/Run summary/Track");
 else if(rfile->GetDirectory("DQMData/Tracking/Track"))rfile->cd("DQMData/Tracking/Track");
 rdir=gDirectory;
 TList *rl= rdir->GetListOfKeys();
 TString collname1=rl->At(0)->GetName(); 
 //HistoCompare_Tracks * myPV = new HistoCompare_Tracks();

 TCanvas *canvas;

 TH1 *sh1,*rh1;
 TH1 *sh2,*rh2;
 TH1 *sh3,*rh3;
 TH1 *sh4,*rh4;
 TH1 *sh5,*rh5;
 TH1 *sh6,*rh6;

 TH2F *sh1_2,*rh1_2;
 TH2F *sh2_2,*rh2_2;
 TH2F *sh3_2,*rh3_2;
 TH2F *sh4_2,*rh4_2;
 TH2F *sh5_2,*rh5_2;
 TH2F *sh6_2,*rh6_2;


 TH1 *sc1,*rc1;
 TH1 *sc2,*rc2;
 TH1 *sc3,*rc3;

 bool hit=1;
 bool chi2=1;
 bool ctf=1;
 bool rs=0;


 //////////////////////////////////////
 /////////// CTF //////////////////////
 //////////////////////////////////////
 if (ctf){
   //===== building
   rdir->GetObject(collname1+"/effic",rh1);
   sdir->GetObject(collname2+"/effic",sh1);
   rh1->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   sh1->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   rh1->GetXaxis()->SetTitle("#eta");
   rh1->GetYaxis()->SetTitle("efficiency vs #eta");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);
//   rh1->GetYaxis()->SetRangeUser(0.5,1.025);
//   sh1->GetYaxis()->SetRangeUser(0.5,1.025);
   rh2 = getEfficiency((TH1*)rdir->Get(collname1+"/num_assoc(recoToSim)_eta"), (TH1*)rdir->Get(collname1+"/num_duplicate_eta"), (TH1*)rdir->Get(collname1+"/num_reco_eta"), "fakerate_vs_eta_r", "fake+duplicates vs #eta", false);    
   sh2 = getEfficiency((TH1*)sdir->Get(collname2+"/num_assoc(recoToSim)_eta"), (TH1*)sdir->Get(collname2+"/num_duplicate_eta"), (TH1*)sdir->Get(collname2+"/num_reco_eta"), "fakerate_vs_eta_s", "fake+duplicates vs #eta", false);    
   rh2->GetYaxis()->SetRangeUser(0.,MAXFAKE);
   sh2->GetYaxis()->SetRangeUser(0.,MAXFAKE);
   rh2->GetXaxis()->SetTitle("#eta");
   rh2->GetYaxis()->SetTitle("fake+duplicates rate vs #eta");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);
//   rh2->GetYaxis()->SetRangeUser(0.,.70);
//   sh2->GetYaxis()->SetRangeUser(0.,.70);


   rdir->GetObject(collname1+"/efficPt",rh3);
   sdir->GetObject(collname2+"/efficPt",sh3);
   rh3->GetXaxis()->SetRangeUser(0,300);
   sh3->GetXaxis()->SetRangeUser(0,300);
   rh3->GetXaxis()->SetTitle("p_{t}");
   rh3->GetYaxis()->SetTitle("efficiency vs p_{t}");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rh3->SetTitle("");
   rh4 = getEfficiency((TH1*)rdir->Get(collname1+"/num_assoc(recoToSim)_pT"), (TH1*)rdir->Get(collname1+"/num_duplicate_pT"), (TH1*)rdir->Get(collname1+"/num_reco_pT"), "fakerate_vs_pT_r", "fake+duplicates vs p_{t}", false);    
   sh4 = getEfficiency((TH1*)sdir->Get(collname2+"/num_assoc(recoToSim)_pT"), (TH1*)sdir->Get(collname2+"/num_duplicate_pT"), (TH1*)sdir->Get(collname2+"/num_reco_pT"), "fakerate_vs_pT_s", "fake+duplicates vs p_{t}", false);    
   rh4->SetTitle("");
   rh4->GetXaxis()->SetTitle("p_{t}");
   rh4->GetYaxis()->SetTitle("fake+duplicates rate vs p_{t}");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);
   rh4->GetYaxis()->SetRangeUser(0.,.80);
   sh4->GetYaxis()->SetRangeUser(0.,.80);
   rh4->GetXaxis()->SetRangeUser(0.2,300);
   sh4->GetXaxis()->SetRangeUser(0.2,300);


   rdir->GetObject(collname1+"/effic_vs_hit",rh5);
   sdir->GetObject(collname2+"/effic_vs_hit",sh5);
   rh5->GetXaxis()->SetTitle("hits");
   rh5->GetYaxis()->SetTitle("efficiency vs hits");
   rh5->GetYaxis()->SetTitleSize(0.05);
   rh5->GetYaxis()->SetTitleOffset(1.2);
   //rh3->GetXaxis()->SetRangeUser(0,30);
   //sh3->GetXaxis()->SetRangeUser(0,30);
   rh6 = getEfficiency((TH1*)rdir->Get(collname1+"/num_assoc(recoToSim)_hit"), (TH1*)rdir->Get(collname1+"/num_duplicate_hit"), (TH1*)rdir->Get(collname1+"/num_reco_hit"), "fakerate_vs_hit_r", "fake+duplicates vs hit", false);    
   sh6 = getEfficiency((TH1*)sdir->Get(collname2+"/num_assoc(recoToSim)_hit"), (TH1*)sdir->Get(collname2+"/num_duplicate_hit"), (TH1*)sdir->Get(collname2+"/num_reco_hit"), "fakerate_vs_hit_s", "fake+duplicates vs hit", false);    
   rh6->GetYaxis()->SetRangeUser(0.,1.0);
   rh6->GetYaxis()->SetRangeUser(0.,1.0);
   rh6->GetXaxis()->SetTitle("hits");
   rh6->GetYaxis()->SetTitle("fake+duplicates rate vs hits");
   rh6->GetYaxis()->SetTitleSize(0.05);
   rh6->GetYaxis()->SetTitleOffset(1.2);

   //rdir->GetObject(collname1+"/num_reco_pT",rh6);
   //sdir->GetObject(collname2+"/num_reco_pT",sh6);



   canvas = new TCanvas("Tracks","Tracks: efficiency & fakerate",1000,1400);


   //NormalizeHistograms(rh2,sh2);
   //NormalizeHistograms(rh6,sh6);
   //rh1->GetYaxis()->SetRangeUser(8,24);
   //sh1->GetYaxis()->SetRangeUser(8,24);

   //rh6->GetXaxis()->SetRangeUser(0,10);
   //sh6->GetXaxis()->SetRangeUser(0,10);

   TH1 * r[6]={rh1,rh2,rh3,rh4,rh5,rh6};
   TH1 * s[6]={sh1,sh2,sh3,sh4,sh5,sh6};

   plotBuilding(canvas,s, r,6,
		te,"UU",-1, 1, false, 0xC);

   canvas->cd();
   //TPaveText* text = new TPaveText(0.25,0.72,0.75,0.77,"prova");
   //text->SetFillColor(0);
   //text->SetTextColor(1);
   //text->Draw();
   l = new TLegend(0.10,0.64,0.90,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print("effandfake1.pdf");   
   delete l;



   //===== fakes and duplicates
   //rdir->GetObject(collname1+"/fakerate",rh1);
   //sdir->GetObject(collname2+"/fakerate",sh1);
   rh1 = getEfficiency((TH1*)rdir->Get(collname1+"/num_assoc(recoToSim)_eta"), (TH1*)rdir->Get(collname1+"/num_reco_eta"), "fakeduprate_vs_eta_r", "fake vs p_{t}", false);    
   sh1 = getEfficiency((TH1*)sdir->Get(collname2+"/num_assoc(recoToSim)_eta"), (TH1*)sdir->Get(collname2+"/num_reco_eta"), "fakeduprate_vs_eta_s", "fake vs p_{t}", false);    
   rh1->GetYaxis()->SetRangeUser(0.,MAXFAKE);
   sh1->GetYaxis()->SetRangeUser(0.,MAXFAKE);
   rh1->GetXaxis()->SetTitle("#eta");
   rh1->GetYaxis()->SetTitle("fakerate vs #eta");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);
//   rh1->GetYaxis()->SetRangeUser(0.5,1.025);
//   sh1->GetYaxis()->SetRangeUser(0.5,1.025);
   rdir->GetObject(collname1+"/duplicatesRate",rh2);
   sdir->GetObject(collname2+"/duplicatesRate",sh2);
   rh2->GetYaxis()->SetRangeUser(0.,MAXFAKE);
   sh2->GetYaxis()->SetRangeUser(0.,MAXFAKE);
   rh2->GetXaxis()->SetTitle("#eta");
   rh2->GetYaxis()->SetTitle("duplicates rate vs #eta");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);
//   rh2->GetYaxis()->SetRangeUser(0.,.70);
//   sh2->GetYaxis()->SetRangeUser(0.,.70);


   //rdir->GetObject(collname1+"/fakeratePt",rh3);
   //sdir->GetObject(collname2+"/fakeratePt",sh3);
   rh3 = getEfficiency((TH1*)rdir->Get(collname1+"/num_assoc(recoToSim)_pT"), (TH1*)rdir->Get(collname1+"/num_reco_pT"), "fakeduprate_vs_pT_r", "fake vs p_{t}", false);    
   sh3 = getEfficiency((TH1*)sdir->Get(collname2+"/num_assoc(recoToSim)_pT"), (TH1*)sdir->Get(collname2+"/num_reco_pT"), "fakeduprate_vs_pT_s", "fake vs p_{t}", false);    
   rh3->GetXaxis()->SetRangeUser(0,300);
   sh3->GetXaxis()->SetRangeUser(0,300);
   rh3->GetXaxis()->SetTitle("p_{t}");
   rh3->GetYaxis()->SetTitle("fakerate vs p_{t}");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rh3->SetTitle("");
   rdir->GetObject(collname1+"/duplicatesRate_Pt",rh4);
   sdir->GetObject(collname2+"/duplicatesRate_Pt",sh4);
   rh4->SetTitle("");
   rh4->GetXaxis()->SetTitle("p_{t}");
   rh4->GetYaxis()->SetTitle("duplicates rate vs p_{t}");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);
   rh4->GetYaxis()->SetRangeUser(0.,.80);
   sh4->GetYaxis()->SetRangeUser(0.,.80);
   rh4->GetXaxis()->SetRangeUser(0.2,300);
   sh4->GetXaxis()->SetRangeUser(0.2,300);


   //rdir->GetObject(collname1+"/fakerate_vs_hit",rh5);
   //sdir->GetObject(collname2+"/fakerate_vs_hit",sh5);
   rh5 = getEfficiency((TH1*)rdir->Get(collname1+"/num_assoc(recoToSim)_hit"), (TH1*)rdir->Get(collname1+"/num_reco_hit"), "fakeduprate_vs_hit_r", "fake vs hit", false);    
   sh5 = getEfficiency((TH1*)sdir->Get(collname2+"/num_assoc(recoToSim)_hit"), (TH1*)sdir->Get(collname2+"/num_reco_hit"), "fakeduprate_vs_hit_s", "fake vs hit", false);    
   rh5->GetXaxis()->SetTitle("hits");
   rh5->GetYaxis()->SetTitle("fakerate vs hits");
   rh5->GetYaxis()->SetTitleSize(0.05);
   rh5->GetYaxis()->SetTitleOffset(1.2);
   //rh3->GetXaxis()->SetRangeUser(0,30);
   //sh3->GetXaxis()->SetRangeUser(0,30);
   rdir->GetObject(collname1+"/duplicatesRate_hit",rh6);
   sdir->GetObject(collname2+"/duplicatesRate_hit",sh6);
   rh6->GetYaxis()->SetRangeUser(0.,1.0);
   rh6->GetYaxis()->SetRangeUser(0.,1.0);
   rh6->GetXaxis()->SetTitle("hits");
   rh6->GetYaxis()->SetTitle("duplicates rate vs hits");
   rh6->GetYaxis()->SetTitleSize(0.05);
   rh6->GetYaxis()->SetTitleOffset(1.2);
   
   //rdir->GetObject(collname1+"/num_reco_pT",rh6);
   //sdir->GetObject(collname2+"/num_reco_pT",sh6);
   


   canvas = new TCanvas("TracksDups","Tracks: fakerate & duplicates rate",1000,1400);


   //NormalizeHistograms(rh2,sh2);
   //NormalizeHistograms(rh6,sh6);
   //rh1->GetYaxis()->SetRangeUser(8,24);
   //sh1->GetYaxis()->SetRangeUser(8,24);
   
   //rh6->GetXaxis()->SetRangeUser(0,10);
   //sh6->GetXaxis()->SetRangeUser(0,10);
   
   TH1 * r[6]={rh1,rh2,rh3,rh4,rh5,rh6};
   TH1 * s[6]={sh1,sh2,sh3,sh4,sh5,sh6};
   
   plotBuilding(canvas,s, r,6,
        te,"UU",-1, 1, false, 0xC);

   canvas->cd();
   //TPaveText* text = new TPaveText(0.25,0.72,0.75,0.77,"prova");
   //text->SetFillColor(0);
   //text->SetTextColor(1);
   //text->Draw();
   l = new TLegend(0.10,0.64,0.90,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print("dupandfake1.pdf");
   delete l;


   // Charge Mid-Id (1)
   rdir->GetObject(collname1+"/chargeMisIdRate",rh1);
   sdir->GetObject(collname2+"/chargeMisIdRate",sh1);
   rh1->GetYaxis()->SetRangeUser(0, 0.05);
   sh1->GetYaxis()->SetRangeUser(0, 0.05);
   rh1->GetXaxis()->SetTitle("#eta");
   rh1->GetYaxis()->SetTitle("charge mis-id rate vs #eta");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/chargeMisIdRate_Pt",rh2);
   sdir->GetObject(collname2+"/chargeMisIdRate_Pt",sh2);
   rh2->GetXaxis()->SetRangeUser(0,300);
   sh2->GetXaxis()->SetRangeUser(0,300);
   rh2->GetXaxis()->SetTitle("p_{t}");
   rh2->GetYaxis()->SetTitle("charge mis-id rate vs p_{t}");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);
   rh2->GetYaxis()->SetRangeUser(0.,0.1);
   sh2->GetYaxis()->SetRangeUser(0.,0.1);
   rh2->SetTitle("");

   rdir->GetObject(collname1+"/chargeMisIdRate_hit",rh3);
   sdir->GetObject(collname2+"/chargeMisIdRate_hit",sh3);
   rh3->GetXaxis()->SetTitle("hits");
   rh3->GetYaxis()->SetTitle("charge mis-id rate vs hits");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/chargeMisIdRate_phi",rh4);
   sdir->GetObject(collname2+"/chargeMisIdRate_phi",sh4);
   rh4->GetXaxis()->SetTitle("#phi");
   rh4->GetYaxis()->SetTitle("charge mis-id rate vs #phi");
   rh4->GetYaxis()->SetRangeUser(0.,0.01);
   sh4->GetYaxis()->SetRangeUser(0.,0.01);
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);
   
   rdir->GetObject(collname1+"/chargeMisIdRate_dxy",rh5);
   sdir->GetObject(collname2+"/chargeMisIdRate_dxy",sh5);
   rh5->SetTitle("");
   rh5->GetXaxis()->SetTitle("dxy");
   rh5->GetYaxis()->SetTitle("charge mis-id rate vs dxy");
   rh5->GetYaxis()->SetTitleSize(0.05);
   rh5->GetYaxis()->SetTitleOffset(1.2);
   rh5->GetYaxis()->SetRangeUser(0.,0.1);
   sh5->GetYaxis()->SetRangeUser(0.,0.1);


   rdir->GetObject(collname1+"/chargeMisIdRate_dz",rh6);
   sdir->GetObject(collname2+"/chargeMisIdRate_dz",sh6);
   rh6->GetYaxis()->SetRangeUser(0.,0.1);
   rh6->GetYaxis()->SetRangeUser(0.,0.1);
   rh6->GetXaxis()->SetTitle("dz");
   rh6->GetYaxis()->SetTitle("charge mis-id rate vs dz");
   rh6->GetYaxis()->SetTitleSize(0.05);
   rh6->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("Charge Mis-IDed Tracks","Tracks: charge mis-id rate",1000,1400);

   TH1 * r[6]={rh1,rh2,rh3,rh4,rh5,rh6};
   TH1 * s[6]={sh1,sh2,sh3,sh4,sh5,sh6};

   plotBuilding(canvas,s, r,6,
        te,"UU",-1, 1, false, 2);

   canvas->cd();
   l = new TLegend(0.10,0.64,0.90,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print("chargemisid.pdf");
   delete l;

   // ====== hits and pt

   TH2F* nhits_vs_eta_r; 
   TH2F* nhits_vs_eta_s; 
   rdir->GetObject(collname1+"/nhits_vs_eta",nhits_vs_eta_r);  
   nhits_vs_eta_r->SetName("nhits_vs_eta_r");                                                                                                                               
   sdir->GetObject(collname2+"/nhits_vs_eta",nhits_vs_eta_s); 
   nhits_vs_eta_s->SetName("nhits_vs_eta_s");                                                                                                                                                                               
   (TProfile*)rh1 = nhits_vs_eta_r->ProfileX();
   (TProfile*)sh1 = nhits_vs_eta_s->ProfileX();

   rdir->GetObject(collname1+"/hits",rh2);                                                                                                                                                                                    
   sdir->GetObject(collname2+"/hits",sh2);         
   
   sdir->GetObject(collname2+"/hits",sh2);         
   rdir->GetObject(collname1+"/num_simul_pT",rh3);
   sdir->GetObject(collname2+"/num_simul_pT",sh3);
   rdir->GetObject(collname1+"/num_reco_pT",rh4);
   sdir->GetObject(collname2+"/num_reco_pT",sh4);
   
   canvas = new TCanvas("Tracks1","Tracks: hits and Pt",1000,1050);
   
   rh1->GetYaxis()->SetRangeUser(8,24);
   sh1->GetYaxis()->SetRangeUser(8,24);
   rh1->GetXaxis()->SetTitle("#eta");
   rh1->GetYaxis()->SetTitle("<hits> vs #eta");
   rh2->GetXaxis()->SetRangeUser(0,40);
   sh2->GetXaxis()->SetRangeUser(0,40);
   rh2->GetXaxis()->SetTitle("hits");


   rh3->GetXaxis()->SetRangeUser(0,10);
   sh3->GetXaxis()->SetRangeUser(0,10);
   rh3->GetXaxis()->SetTitle("p_{t}");
   rh4->GetXaxis()->SetRangeUser(0,10);
   sh4->GetXaxis()->SetRangeUser(0,10);
   rh4->GetXaxis()->SetTitle("p_{t}");
   NormalizeHistograms(rh3,sh3);
   NormalizeHistograms(rh4,sh4);
   
   plot4histos(canvas,
	      sh1,rh1,sh2,rh2,
	      sh3,rh3,sh4,rh4,
	      te,"UU",-1);
   
   canvas->cd();
   
   l = new TLegend(0.20,0.49,0.90,0.54);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print("hitsAndPt.pdf");
   delete l;


   //===== tuning
   TH2F* chi2_vs_eta_r; 
   TH2F* chi2_vs_eta_s; 
 
   rdir->GetObject(collname1+"/chi2_vs_eta", chi2_vs_eta_r);
   chi2_vs_eta_r->SetName("chi2_vs_eta_r");

   sdir->GetObject(collname2+"/chi2_vs_eta", chi2_vs_eta_s);
   chi2_vs_eta_s->SetName("chi2_vs_eta_s");

   (TProfile*)rh3 = chi2_vs_eta_r->ProfileX();
   (TProfile*)sh3 = chi2_vs_eta_s->ProfileX();

   rdir->GetObject(collname1+"/chi2",rh1);
   sdir->GetObject(collname2+"/chi2",sh1);
   rdir->GetObject(collname1+"/chi2_prob",rh2);
   sdir->GetObject(collname2+"/chi2_prob",sh2);

   rdir->GetObject(collname1+"/ptres_vs_eta_Mean",rh4);
   sdir->GetObject(collname2+"/ptres_vs_eta_Mean",sh4);


   canvas = new TCanvas("Tracks2","Tracks: chi2 & chi2 probability",1000,1050);

   NormalizeHistograms(rh1,sh1);
   NormalizeHistograms(rh2,sh2);
   fixRangeY(rh1,sh1);
   fixRangeY(rh2,sh2);
   rh1->GetXaxis()->SetTitle("#chi^{2}");
   rh2->GetXaxis()->SetTitle("Prob(#chi^{2})");
   
   rh3->GetYaxis()->SetRangeUser(0,2.5);
   sh3->GetYaxis()->SetRangeUser(0,2.5);


   rh3->SetTitle("");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rh3->GetYaxis()->SetTitle("<  #chi^{2} / ndf >");
   rh3->GetXaxis()->SetTitleSize(0.07);
   rh3->GetXaxis()->SetTitleOffset(0.6);
   rh3->GetXaxis()->SetTitle("#eta");

   rh4->Scale(100.);
   sh4->Scale(100.);
   rh4->GetYaxis()->SetRangeUser(-1.5,1.5);
   sh4->GetYaxis()->SetRangeUser(-1.5,1.5);
   rh4->SetTitle("");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);
   rh4->GetYaxis()->SetTitle("< #delta p_{t} / p_{t} > [%]");
   rh4->GetXaxis()->SetTitleSize(0.07);
   rh4->GetXaxis()->SetTitleOffset(0.6);
   rh4->GetXaxis()->SetTitle("#eta");

     
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
   canvas->Print("tuning.pdf");
   delete l;
   

   //===== pulls
   rdir->GetObject(collname1+"/pullPt",rh1);
   sdir->GetObject(collname2+"/pullPt",sh1);

   rdir->GetObject(collname1+"/pullQoverp",rh2);
   sdir->GetObject(collname2+"/pullQoverp",sh2);

   rdir->GetObject(collname1+"/pullPhi",rh3);

   sdir->GetObject(collname2+"/pullPhi",sh3);

   rdir->GetObject(collname1+"/pullTheta",rh4);
   sdir->GetObject(collname2+"/pullTheta",sh4);

   rdir->GetObject(collname1+"/pullDxy",rh5);
   sdir->GetObject(collname2+"/pullDxy",sh5);

   rdir->GetObject(collname1+"/pullDz",rh6);
   sdir->GetObject(collname2+"/pullDz",sh6);


   canvas = new TCanvas("Tracks4","Tracks: pull of Pt, Qoverp and Phi",1000,1400);

   NormalizeHistograms(rh1,sh1);
   NormalizeHistograms(rh2,sh2);
   NormalizeHistograms(rh3,sh3);
   NormalizeHistograms(rh4,sh4);
   NormalizeHistograms(rh5,sh5);
   NormalizeHistograms(rh6,sh6);

   rh1->GetXaxis()->SetRangeUser(-10,10);
   sh1->GetXaxis()->SetRangeUser(-10,10);
   rh2->GetXaxis()->SetRangeUser(-10,10);
   sh2->GetXaxis()->SetRangeUser(-10,10);
   rh3->GetXaxis()->SetRangeUser(-10,10);
   sh3->GetXaxis()->SetRangeUser(-10,10);
   rh4->GetXaxis()->SetRangeUser(-10,10);
   sh4->GetXaxis()->SetRangeUser(-10,10);
   rh5->GetXaxis()->SetRangeUser(-10,10);
   sh5->GetXaxis()->SetRangeUser(-10,10);
   rh6->GetXaxis()->SetRangeUser(-10,10);
   sh6->GetXaxis()->SetRangeUser(-10,10);


   plotPulls(canvas,
	     sh1,rh1,sh2,rh2,
	     sh3,rh3,sh4,rh4,
	     sh5,rh5,sh6,rh6,
	     te,"UU",-1);

   canvas->cd();

   l = new TLegend(0.20,0.655,0.80,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print("pulls.pdf");
   delete l;




   const char* ystr1 = "#sigma(#delta#phi) [rad]"; 
   const char* ystr2 = "#sigma(#delta cot(#theta))"; 
   const char* ystr3 = "#sigma(#delta d_{0}) [cm]"; 
   const char* ystr4 = "#sigma(#delta z_{0}) [cm]"; 
   const char* ystr5 = "#sigma(#delta p_{t}/p_{t})";
   const char* xstr  = "#eta"; 

   
   //===== resolutions vs eta
   rdir->GetObject(collname1+"/phires_vs_eta",rh1_2);
   sdir->GetObject(collname2+"/phires_vs_eta",sh1_2);

   rdir->GetObject(collname1+"/cotThetares_vs_eta",rh2_2);
   sdir->GetObject(collname2+"/cotThetares_vs_eta",sh2_2);

   rdir->GetObject(collname1+"/dxyres_vs_eta",rh3_2);
   sdir->GetObject(collname2+"/dxyres_vs_eta",sh3_2);

   rdir->GetObject(collname1+"/dzres_vs_eta",rh4_2);
   sdir->GetObject(collname2+"/dzres_vs_eta",sh4_2);

   rdir->GetObject(collname1+"/ptres_vs_eta",rh5_2);
   sdir->GetObject(collname2+"/ptres_vs_eta",sh5_2);



   canvas = new TCanvas("Tracks7","Tracks: Dxy, Dz, Theta resolution",1000,1400);

   plotResolutions(canvas,
		   sh1_2,rh1_2,sh2_2,rh2_2,
		   sh3_2,rh3_2,sh4_2,rh4_2,
		   sh5_2,rh5_2,
           ystr1, ystr2, ystr3, ystr4, ystr5, xstr,
		   te,"UU",-1);

   /*
   //===== resolutions vs eta
   rdir->GetObject(collname1+"/phires_vs_eta_Sigma",rh1);
   sdir->GetObject(collname2+"/phires_vs_eta_Sigma",sh1);

   rdir->GetObject(collname1+"/cotThetares_vs_eta_Sigma",rh2);
   sdir->GetObject(collname2+"/cotThetares_vs_eta_Sigma",sh2);

   rdir->GetObject(collname1+"/dxyres_vs_eta_Sigma",rh3);
   sdir->GetObject(collname2+"/dxyres_vs_eta_Sigma",sh3);

   rdir->GetObject(collname1+"/dzres_vs_eta_Sigma",rh4);
   sdir->GetObject(collname2+"/dzres_vs_eta_Sigma",sh4);

   rdir->GetObject(collname1+"/ptres_vs_eta_Sigma",rh5);
   sdir->GetObject(collname2+"/ptres_vs_eta_Sigma",sh5);



   canvas = new TCanvas("Tracks7","Tracks: Dxy, Dz, Theta resolution",1000,1400);

   plotResolutionsDirect(canvas,
           sh1,rh1,sh2,rh2,
           sh3,rh3,sh4,rh4,
           sh5,rh5,sh6,rh6,
           te,"UU",-1);
   */   


   // new general range
   rh1->GetYaxis()->SetRangeUser(0.000009,0.01);
   sh1->GetYaxis()->SetRangeUser(0.000009,0.01);

   rh1->SetTitle(""); 
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);
   //   rh1->GetYaxis()->SetTitleColor(2);
   rh1->GetYaxis()->SetTitle("#sigma(#delta #phi) [rad]");
   rh1->GetXaxis()->SetTitleSize(0.07);
   rh1->GetXaxis()->SetTitleOffset(0.6);
   rh1->GetXaxis()->SetTitle("#eta");


   // new general range
   rh2->GetYaxis()->SetRangeUser(0.00009,0.03);
   sh2->GetYaxis()->SetRangeUser(0.00009,0.03);
   // for multi-track samples
   //rh2->GetYaxis()->SetRangeUser(0.0009,0.01);
   //sh2->GetYaxis()->SetRangeUser(0.0009,0.01);
   // for single particle pt 10
   //rh2->GetYaxis()->SetRangeUser(0.00009,0.01);
   //sh2->GetYaxis()->SetRangeUser(0.00009,0.01);
   // for single particle pt 1
   //rh2->GetYaxis()->SetRangeUser(0.0009,0.01);
   //sh2->GetYaxis()->SetRangeUser(0.0009,0.01);
   rh2->SetTitle("");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);
   rh2->GetYaxis()->SetTitle("#sigma(#delta cot(#theta)) ");
   rh2->GetXaxis()->SetTitleSize(0.07);
   rh2->GetXaxis()->SetTitleOffset(0.6);
   rh2->GetXaxis()->SetTitle("#eta");


   // new general range
   rh3->GetYaxis()->SetRangeUser(0.00009,0.05);  
   sh3->GetYaxis()->SetRangeUser(0.00009,0.05);
   // for multi-track samples
   //rh3->GetYaxis()->SetRangeUser(0.0009,0.02);  
   //sh3->GetYaxis()->SetRangeUser(0.0009,0.02);
   //rh3->GetYaxis()->SetRangeUser(0.0009,0.02);
   //sh3->GetYaxis()->SetRangeUser(0.0009,0.02);
   // for single particle pt 100    
   //rh3->GetYaxis()->SetRangeUser(0.00009,0.002);
   //sh3->GetYaxis()->SetRangeUser(0.00009,0.002);
   rh3->SetTitle("");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rh3->GetYaxis()->SetTitle("#sigma(#delta d_{0}) [cm]");
   rh3->GetXaxis()->SetTitleSize(0.07);
   rh3->GetXaxis()->SetTitleOffset(0.6);
   rh3->GetXaxis()->SetTitle("#eta"); 


   // new general range
   rh4->GetYaxis()->SetRangeUser(0.0009,0.1);  
   sh4->GetYaxis()->SetRangeUser(0.0009,0.1);
   // for multi-track samples
   //rh4->GetYaxis()->SetRangeUser(0.0009,0.08);
   //sh4->GetYaxis()->SetRangeUser(0.0009,0.08);

   rh4->SetTitle("");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);
   rh4->GetYaxis()->SetTitle("#sigma(#delta z_{0}) [cm]");
   rh4->GetXaxis()->SetTitleSize(0.07);
   rh4->GetXaxis()->SetTitleOffset(0.6);
   rh4->GetXaxis()->SetTitle("#eta");

   rh5->SetTitle("");
   rh5->GetYaxis()->SetTitleSize(0.05);
   rh5->GetYaxis()->SetTitleOffset(1.2);
   rh5->GetYaxis()->SetTitle("#sigma(#delta p_{t}/p_{t}) ");
   rh5->GetXaxis()->SetTitleSize(0.07);
   rh5->GetXaxis()->SetTitleOffset(0.6);
   rh5->GetXaxis()->SetTitle("#eta");



   canvas->cd();

   l = new TLegend(0.10,0.63,0.90,0.67);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print("resolutionsEta.pdf");
   delete l;

   //===== resolutions vs pt
   xstr  = "p_{t}"; 
   rdir->GetObject(collname1+"/phires_vs_pt",rh1_2);
   sdir->GetObject(collname2+"/phires_vs_pt",sh1_2);

   rdir->GetObject(collname1+"/cotThetares_vs_pt",rh2_2);
   sdir->GetObject(collname2+"/cotThetares_vs_pt",sh2_2);


   rdir->GetObject(collname1+"/dxyres_vs_pt",rh3_2);
   sdir->GetObject(collname2+"/dxyres_vs_pt",sh3_2);

   rdir->GetObject(collname1+"/dzres_vs_pt",rh4_2);
   sdir->GetObject(collname2+"/dzres_vs_pt",sh4_2);

   rdir->GetObject(collname1+"/ptres_vs_pt",rh5_2);
   sdir->GetObject(collname2+"/ptres_vs_pt",sh5_2);


   rh1->SetTitle("");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);
   rh1->GetYaxis()->SetTitle("#sigma(#delta #phi) [rad]");
   rh1->GetXaxis()->SetTitleSize(0.055);
   rh1->GetXaxis()->SetTitleOffset(0.8);
   rh1->GetXaxis()->SetTitle("p_{t}");
   rh1->GetXaxis()->SetRangeUser(0,1000.);
   sh1->GetXaxis()->SetRangeUser(0,1000.);
 

   rh2->SetTitle("");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);
   rh2->GetYaxis()->SetTitle("#sigma(#delta cot(#theta)) ");
   rh2->GetXaxis()->SetTitleSize(0.055);
   rh2->GetXaxis()->SetTitleOffset(0.8);
   rh2->GetXaxis()->SetTitle("p_{t}");

   rh2->GetXaxis()->SetRangeUser(0,1000.);
   sh2->GetXaxis()->SetRangeUser(0,1000.);

   rh3->SetTitle("");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rh3->GetYaxis()->SetTitle("#sigma(#delta d_{0}) [cm]");
   rh3->GetXaxis()->SetTitleSize(0.055);
   rh3->GetXaxis()->SetTitleOffset(0.8);
   rh3->GetXaxis()->SetTitle("p_{t}");


   rh3->GetXaxis()->SetRangeUser(0,1000.);
   sh3->GetXaxis()->SetRangeUser(0,1000.);


   rh4->SetTitle("");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);
   rh4->GetYaxis()->SetTitle("#sigma(#delta z_{0}) [cm]");
   rh4->GetXaxis()->SetTitleSize(0.055);
   rh4->GetXaxis()->SetTitleOffset(0.8);
   rh4->GetXaxis()->SetTitle("p_{t}");

   rh4->GetXaxis()->SetRangeUser(0,1000.);
   sh4->GetXaxis()->SetRangeUser(0,1000.);


   rh5->SetTitle("");
   rh5->GetYaxis()->SetTitleSize(0.05);
   rh5->GetYaxis()->SetTitleOffset(1.2);
   rh5->GetYaxis()->SetTitle("#sigma(#delta p_{t}/p_{t}) ");
   rh5->GetXaxis()->SetTitleSize(0.055);
   rh5->GetXaxis()->SetTitleOffset(0.8);
   rh5->GetXaxis()->SetTitle("p_{t}");


   rh5->GetXaxis()->SetRangeUser(0,1000.);
   sh5->GetXaxis()->SetRangeUser(0,1000.);


   rh6->GetXaxis()->SetRangeUser(0,1000.);
   sh6->GetXaxis()->SetRangeUser(0,1000.);


   canvas = new TCanvas("Tracks7b","Tracks: Dxy, Dz, Theta resolution",1000,1400);

   plotResolutions(canvas,
	     sh1_2,rh1_2,sh2_2,rh2_2,
	     sh3_2,rh3_2,sh4_2,rh4_2,
	     sh5_2,rh5_2,
           ystr1, ystr2, ystr3, ystr4, ystr5, xstr,
		   te,"UU",-1, true);

   canvas->cd();

   l = new TLegend(0.10,0.63,0.90,0.67);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print("resolutionsPt.pdf");
   delete l;

    //===== building 2
   rdir->GetObject(collname1+"/effic_vs_phi",rh1);
   sdir->GetObject(collname2+"/effic_vs_phi",sh1);
   rh1->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   sh1->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   rh1->GetXaxis()->SetTitle("#phi");
   rh1->GetYaxis()->SetTitle("efficiency vs #phi");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);
   rh2 = getEfficiency((TH1*)rdir->Get(collname1+"/num_assoc(recoToSim)_phi"), (TH1*)rdir->Get(collname1+"/num_duplicate_phi"), (TH1*)rdir->Get(collname1+"/num_reco_phi"), "fakerate_vs_phi_r", "fake+duplicates vs #phi", false);    
   sh2 = getEfficiency((TH1*)sdir->Get(collname2+"/num_assoc(recoToSim)_phi"), (TH1*)sdir->Get(collname2+"/num_duplicate_phi"), (TH1*)sdir->Get(collname2+"/num_reco_phi"), "fakerate_vs_phi_s", "fake+duplicates vs #phi", false);    
   rh2->GetXaxis()->SetTitle("#phi");
   rh2->GetYaxis()->SetTitle("fake+duplicates rate vs #phi");
   rh2->GetYaxis()->SetRangeUser(0.,MAXFAKE);
   sh2->GetYaxis()->SetRangeUser(0.,MAXFAKE);
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);



   rdir->GetObject(collname1+"/effic_vs_dxy",rh3);
   sdir->GetObject(collname2+"/effic_vs_dxy",sh3);
   rh3->GetXaxis()->SetTitle("dxy");
   rh3->GetYaxis()->SetTitle("efficiency vs dxy");
   rh3->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   sh3->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rh3->SetTitle("");
   rh4 = getEfficiency((TH1*)rdir->Get(collname1+"/num_assoc(recoToSim)_dxy"), (TH1*)rdir->Get(collname1+"/num_duplicate_dxy"), (TH1*)rdir->Get(collname1+"/num_reco_dxy"), "fakerate_vs_dxy_r", "fake+duplicates vs dxy", false);    
   sh4 = getEfficiency((TH1*)sdir->Get(collname2+"/num_assoc(recoToSim)_dxy"), (TH1*)sdir->Get(collname2+"/num_duplicate_dxy"), (TH1*)sdir->Get(collname2+"/num_reco_dxy"), "fakerate_vs_dxy_s", "fake+duplicates vs dxy", false);    
   rh4->SetTitle("");
   rh4->GetXaxis()->SetTitle("dxy");
   rh4->GetYaxis()->SetTitle("fake+duplicates rate vs dxy");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);
   rh4->GetYaxis()->SetRangeUser(0.,MAXFAKE);
   sh4->GetYaxis()->SetRangeUser(0.,MAXFAKE);


   rdir->GetObject(collname1+"/effic_vs_dz",rh5);
   sdir->GetObject(collname2+"/effic_vs_dz",sh5);
   rh5->GetXaxis()->SetTitle("dz");
   rh5->GetYaxis()->SetTitle("efficiency vs dz");
   rh5->GetYaxis()->SetTitleSize(0.05);
   rh5->GetYaxis()->SetTitleOffset(1.2);
   rh5->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   sh5->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   rh6 = getEfficiency((TH1*)rdir->Get(collname1+"/num_assoc(recoToSim)_dz"), (TH1*)rdir->Get(collname1+"/num_duplicate_dz"), (TH1*)rdir->Get(collname1+"/num_reco_dz"), "fakerate_vs_dz_r", "fake+duplicates vs dz", false);    
   sh6 = getEfficiency((TH1*)sdir->Get(collname2+"/num_assoc(recoToSim)_dz"), (TH1*)sdir->Get(collname2+"/num_duplicate_dz"), (TH1*)sdir->Get(collname2+"/num_reco_dz"), "fakerate_vs_dz_s", "fake+duplicates vs dz", false);    
   rh6->GetYaxis()->SetRangeUser(0.,0.7);
   rh6->GetXaxis()->SetTitle("dz");
   rh6->GetYaxis()->SetTitle("fake+duplicates rate vs dz");
   rh6->GetYaxis()->SetTitleSize(0.05);
   rh6->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("Tracks8","Tracks: efficiency & fakerate",1000,1400);

   TH1 * r[6]={rh1,rh2,rh3,rh4,rh5,rh6};
   TH1 * s[6]={sh1,sh2,sh3,sh4,sh5,sh6};

   plotBuilding(canvas,s, r,6,
		te,"UU",-1);

   canvas->cd();
   l = new TLegend(0.10,0.64,0.90,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print("effandfake2.pdf");   
   delete l;



    //===== building 2
   //rdir->GetObject(collname1+"/fakerate_vs_phi",rh1);
   //sdir->GetObject(collname2+"/fakerate_vs_phi",sh1);
   rh1 = getEfficiency((TH1*)rdir->Get(collname1+"/num_assoc(recoToSim)_phi"), (TH1*)rdir->Get(collname1+"/num_reco_phi"), "fakerate_vs_phi_r", "fake vs #phi", false);    
   sh1 = getEfficiency((TH1*)sdir->Get(collname2+"/num_assoc(recoToSim)_phi"), (TH1*)sdir->Get(collname2+"/num_reco_phi"), "fakerate_vs_phi_s", "fake vs #phi", false);    
   rh1->GetYaxis()->SetRangeUser(0., MAXFAKE);
   sh1->GetYaxis()->SetRangeUser(0., MAXFAKE);
   rh1->GetXaxis()->SetTitle("#phi");
   rh1->GetYaxis()->SetTitle("fakerate vs #phi");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);
   rh2 = (TH1*)rdir->Get(collname1+"/duplicatesRate_phi");
   sh2 = (TH1*)sdir->Get(collname2+"/duplicatesRate_phi");
   //rh2 = getEfficiency((TH1*)rdir->Get(collname1+"/num_duplicate_phi"), (TH1*)rdir->Get(collname1+"/num_reco_phi"), "duprate_vs_phi_r", "duplicates vs phi", true);    
   //sh2 = getEfficiency((TH1*)sdir->Get(collname2+"/num_duplicate_phi"), (TH1*)sdir->Get(collname2+"/num_reco_phi"), "duprate_vs_phi_s", "duplicates vs phi", true);    
   rh2->GetXaxis()->SetTitle("#phi");
   rh2->GetYaxis()->SetTitle("duplicates rate vs #phi");
   rh2->GetYaxis()->SetRangeUser(0.,MAXFAKE);
   sh2->GetYaxis()->SetRangeUser(0.,MAXFAKE);
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);



   //rdir->GetObject(collname1+"/fakerate_vs_dxy",rh3);
   //sdir->GetObject(collname2+"/fakerate_vs_dxy",sh3);
   rh3 = getEfficiency((TH1*)rdir->Get(collname1+"/num_assoc(recoToSim)_dxy"),(TH1*)rdir->Get(collname1+"/num_reco_dxy"), "fakerate_vs_dxy_r", "fake vs dxy", false);    
   sh3 = getEfficiency((TH1*)sdir->Get(collname2+"/num_assoc(recoToSim)_dxy"),(TH1*)sdir->Get(collname2+"/num_reco_dxy"), "fakerate_vs_dxy_s", "fake vs dxy", false);    
   rh3->GetXaxis()->SetTitle("dxy");
   rh3->GetYaxis()->SetTitle("fakerate vs dxy");
   rh3->GetYaxis()->SetRangeUser(0.,MAXFAKE);
   sh3->GetYaxis()->SetRangeUser(0.,MAXFAKE);
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rh3->SetTitle("");
   rh4 = ((TH1*)rdir->Get(collname1+"/duplicatesRate_dxy"));
   sh4 = ((TH1*)sdir->Get(collname2+"/duplicatesRate_dxy"));
   rh4->SetTitle("");
   rh4->GetXaxis()->SetTitle("dxy");
   rh4->GetYaxis()->SetTitle("duplicates rate vs dxy");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);
   rh4->GetYaxis()->SetRangeUser(0.,MAXFAKE);
   sh4->GetYaxis()->SetRangeUser(0.,MAXFAKE);


   //rdir->GetObject(collname1+"/fakerate_vs_dz",rh5);
   //sdir->GetObject(collname2+"/fakerate_vs_dz",sh5);
   rh5 = getEfficiency((TH1*)rdir->Get(collname1+"/num_assoc(recoToSim)_dz"), (TH1*)rdir->Get(collname1+"/num_reco_dz"), "fakerate_vs_dz_r", "fake vs dz", false);    
   sh5 = getEfficiency((TH1*)sdir->Get(collname2+"/num_assoc(recoToSim)_dz"), (TH1*)sdir->Get(collname2+"/num_reco_dz"), "fakerate_vs_dz_s", "fake vs dz", false);    
   rh5->GetXaxis()->SetTitle("dz");
   rh5->GetYaxis()->SetTitle("fakerate vs dz");
   rh5->GetYaxis()->SetTitleSize(0.05);
   rh5->GetYaxis()->SetTitleOffset(1.2);
   rh5->GetYaxis()->SetRangeUser(0.,MAXFAKE);
   sh5->GetYaxis()->SetRangeUser(0.,MAXFAKE);
   rdir->GetObject(collname1+"/duplicatesRate_dz",rh6);
   sdir->GetObject(collname2+"/duplicatesRate_dz",sh6);
   rh6->GetYaxis()->SetRangeUser(0.,0.7);
   rh6->GetYaxis()->SetRangeUser(0.,0.7);
   rh6->GetXaxis()->SetTitle("dz");
   rh6->GetYaxis()->SetTitle("duplicates rate vs dz");
   rh6->GetYaxis()->SetTitleSize(0.05);
   rh6->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("TracksDup8","Tracks: efficiency & fakerate",1000,1400);

   TH1 * r[6]={rh1,rh2,rh3,rh4,rh5,rh6};
   TH1 * s[6]={sh1,sh2,sh3,sh4,sh5,sh6};

   plotBuilding(canvas,s, r,6,
        te,"UU",-1);

   canvas->cd();
   l = new TLegend(0.10,0.64,0.90,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print("dupandfake2.pdf");
   delete l;


    /*
    //===== Duplicates and Charge Mis-ID 2
   rdir->GetObject(collname1+"/duplicatesRate_phi",rh1);
   sdir->GetObject(collname2+"/duplicatesRate_phi",sh1);
   rh1->GetYaxis()->SetRangeUser(0, 0.1);
   sh1->GetYaxis()->SetRangeUser(0, 0.1);
   rh1->GetXaxis()->SetTitle("#phi");
   rh1->GetYaxis()->SetTitle("duplicates rate vs #phi");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);
   rdir->GetObject(collname1+"/chargeMisIdRate_phi",rh2);
   sdir->GetObject(collname2+"/chargeMisIdRate_phi",sh2);
   rh2->GetXaxis()->SetTitle("#phi");
   rh2->GetYaxis()->SetTitle("charge mis-id rate vs #phi");
   rh2->GetYaxis()->SetRangeUser(0.,0.1);
   sh2->GetYaxis()->SetRangeUser(0.,0.1);
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);



   rdir->GetObject(collname1+"/duplicatesRate_dxy",rh3);
   sdir->GetObject(collname2+"/duplicatesRate_dxy",sh3);
   rh3->GetXaxis()->SetTitle("dxy");
   rh3->GetYaxis()->SetTitle("duplicates rate vs dxy");
   rh3->GetYaxis()->SetRangeUser(0, 0.2);
   sh3->GetYaxis()->SetRangeUser(0, 0.2);
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rh3->SetTitle("");
   rdir->GetObject(collname1+"/chargeMisIdRate_dxy",rh4);
   sdir->GetObject(collname2+"/chargeMisIdRate_dxy",sh4);
   rh4->SetTitle("");
   rh4->GetXaxis()->SetTitle("dxy");
   rh4->GetYaxis()->SetTitle("charge mis-id rate vs dxy");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);
   rh4->GetYaxis()->SetRangeUser(0.,0.1);
   sh4->GetYaxis()->SetRangeUser(0.,0.1);


   rdir->GetObject(collname1+"/duplicatesRate_dz",rh5);
   sdir->GetObject(collname2+"/duplicatesRate_dz",sh5);
   rh5->GetXaxis()->SetTitle("dz");
   rh5->GetYaxis()->SetTitle("duplicates rate vs dz");
   rh5->GetYaxis()->SetTitleSize(0.05);
   rh5->GetYaxis()->SetTitleOffset(1.2);
   rh5->GetYaxis()->SetRangeUser(0, 0.1);
   sh5->GetYaxis()->SetRangeUser(0, 0.1);
   rdir->GetObject(collname1+"/chargeMisIdRate_dz",rh6);
   sdir->GetObject(collname2+"/chargeMisIdRate_dz",sh6);
   rh6->GetYaxis()->SetRangeUser(0.,0.1);
   rh6->GetYaxis()->SetRangeUser(0.,0.1);
   rh6->GetXaxis()->SetTitle("dz");
   rh6->GetYaxis()->SetTitle("charge mis-id rate vs dz");
   rh6->GetYaxis()->SetTitleSize(0.05);
   rh6->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("Duplicates and Charge MisIDed Tracks 2","Tracks: duplicates and charge mis-id rate",1000,1400);

   TH1 * r[6]={rh1,rh2,rh3,rh4,rh5,rh6};
   TH1 * s[6]={sh1,sh2,sh3,sh4,sh5,sh6};

   plotBuilding(canvas,s, r,6,
        te,"UU",-1);

   canvas->cd();
   l = new TLegend(0.10,0.64,0.90,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print("duplicatesAndChargeMisId2.pdf");
   delete l;
   */

    //===== building 3
   rdir->GetObject(collname1+"/effic_vs_vertpos",rh1);
   sdir->GetObject(collname2+"/effic_vs_vertpos",sh1);
   //rh1 = getEfficiency((TH1*)rdir->Get(collname1+"/num_assoc(simToReco)_vertpos"), (TH1*)rdir->Get(collname1+"/num_simul_vertpos"), "effic_vs_vertpos_r", "efficiency vs vertpos", true);    
   //sh1 = getEfficiency((TH1*)sdir->Get(collname2+"/num_assoc(simToReco)_vertpos"), (TH1*)sdir->Get(collname2+"/num_simul_vertpos"), "effic_vs_vertpos_s", "efficiency vs vertpos", true);    
   rh1->GetYaxis()->SetRangeUser(0.,1.);
   sh1->GetYaxis()->SetRangeUser(0.,1.);
   rh1->GetXaxis()->SetTitle("TP vert xy pos");
   rh1->GetYaxis()->SetTitle("efficiency vs vert xy pos");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/effic_vs_zpos",rh2);
   sdir->GetObject(collname2+"/effic_vs_zpos",sh2);
   //rh2 = getEfficiency((TH1*)rdir->Get(collname1+"/num_assoc(simToReco)_zpos"), (TH1*)rdir->Get(collname1+"/num_simul_zpos"), "effic_vs_zpos_r", "efficiency vs zpos", true);    
   //sh2 = getEfficiency((TH1*)sdir->Get(collname2+"/num_assoc(simToReco)_zpos"), (TH1*)sdir->Get(collname2+"/num_simul_zpos"), "effic_vs_zpos_s", "efficiency vs zpos", true);    
   rh2->GetXaxis()->SetTitle("TP vert z pos");
   rh2->GetYaxis()->SetTitle("efficiency vs  vert z pos");
   rh2->GetYaxis()->SetRangeUser(0.,1.);
   sh2->GetYaxis()->SetRangeUser(0.,1.);
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);
   rh2->SetTitle("");

   canvas = new TCanvas("Tracks9","Tracks: efficiency & fakerate",1000,1400);

   TH1 * r[2]={rh1,rh2};
   TH1 * s[2]={sh1,sh2};

   plotBuilding(canvas,s, r,2,
		te,"UU",-1);

   canvas->cd();
   l = new TLegend(0.10,0.14,0.90,0.19);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print("effvspos.pdf");   
   delete l;

   //===== dE/dx
   rdir->GetObject(collname1+"/h_dedx_estim1",rh1);
   sdir->GetObject(collname2+"/h_dedx_estim1",sh1);
   rdir->GetObject(collname1+"/h_dedx_estim2",rh2);
   sdir->GetObject(collname2+"/h_dedx_estim2",sh2);
   rdir->GetObject(collname1+"/h_dedx_nom1",rh3);
   sdir->GetObject(collname2+"/h_dedx_nom1",sh3);
   rdir->GetObject(collname1+"/h_dedx_sat1",rh4);
   sdir->GetObject(collname2+"/h_dedx_sat1",sh4);
   // comment: for NOM and SAT, in principle there should be no difference between algorithms (although...)

   canvas = new TCanvas("Tracks10","Tracks: dE/dx",1000,1400);

   NormalizeHistograms(rh1,sh1);
   NormalizeHistograms(rh2,sh2);
   fixRangeY(rh1,sh1);
   fixRangeY(rh2,sh2);
   rh1->GetXaxis()->SetTitle("dE/dx, harm2");
   rh2->GetXaxis()->SetTitle("dE/dx, trunc40");
   
   plot4histos(canvas,
	       sh1,rh1,sh2,rh2,
	       sh3,rh3,sh4,rh4,    
	       te,"UU",-1);

   canvas->cd();
   l = new TLegend(0.10,0.14,0.90,0.19);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print("dedx.pdf");   
   delete l;

    /*
   //===== Pileup 1
   rdir->GetObject(collname1+"/effic_vertcount_barrel",rh1);
   sdir->GetObject(collname2+"/effic_vertcount_barrel",sh1);
   rdir->GetObject(collname1+"/fakerate_vertcount_barrel",rh2);
   sdir->GetObject(collname2+"/fakerate_vertcount_barrel",sh2);
   rdir->GetObject(collname1+"/effic_vertcount_fwdpos",rh3);
   sdir->GetObject(collname2+"/effic_vertcount_fwdpos",sh3);
   rdir->GetObject(collname1+"/fakerate_vertcount_fwdpos",rh4);
   sdir->GetObject(collname2+"/fakerate_vertcount_fwdpos",sh4);
   rdir->GetObject(collname1+"/effic_vertcount_fwdneg",rh5);
   sdir->GetObject(collname2+"/effic_vertcount_fwdneg",sh5);
   rdir->GetObject(collname1+"/fakerate_vertcount_fwdneg",rh6);
   sdir->GetObject(collname2+"/fakerate_vertcount_fwdneg",sh6);

   rh1->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   sh1->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   rh3->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   sh3->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   rh5->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   sh5->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   rh2->GetYaxis()->SetRangeUser(0.0,MAXFAKE);
   sh2->GetYaxis()->SetRangeUser(0.0,MAXFAKE);
   rh4->GetYaxis()->SetRangeUser(0.0,MAXFAKE);
   sh4->GetYaxis()->SetRangeUser(0.0,MAXFAKE);
   rh6->GetYaxis()->SetRangeUser(0.0,MAXFAKE);
   sh6->GetYaxis()->SetRangeUser(0.0,MAXFAKE);

   canvas = new TCanvas("Tracks11","Tracks: Pileup (1)",1000,1400);


   //fixRangeY(rh1,sh1);
   //fixRangeY(rh2,sh2);
   rh1->GetXaxis()->SetTitle("Number of Pileup Vertices");
   rh2->GetXaxis()->SetTitle("Number of Pileup Vertices");
   rh3->GetXaxis()->SetTitle("Number of Pileup Vertices");
   rh4->GetXaxis()->SetTitle("Number of Pileup Vertices");
   rh5->GetXaxis()->SetTitle("Number of Pileup Vertices");
   rh6->GetXaxis()->SetTitle("Number of Pileup Vertices");

   TH1 * r[6]={rh1,rh2,rh3,rh4,rh5,rh6};
   TH1 * s[6]={sh1,sh2,sh3,sh4,sh5,sh6};

   plotBuilding(canvas,s, r,6,
                te,"UU",-1);
   canvas->cd();
   //TPaveText* text = new TPaveText(0.25,0.72,0.75,0.77,"prova");
   //text->SetFillColor(0);
   //text->SetTextColor(1);
   //text->Draw();
   l = new TLegend(0.10,0.64,0.90,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print("pileup1.pdf");
   delete l;

   //===== Pileup 2
   rdir->GetObject(collname1+"/effic_vertz_barrel",rh1);
   sdir->GetObject(collname2+"/effic_vertz_barrel",sh1);
   rdir->GetObject(collname1+"/effic_vertz_fwdpos",rh2);
   sdir->GetObject(collname2+"/effic_vertz_fwdpos",sh2);
   rdir->GetObject(collname1+"/effic_vertz_fwdneg",rh3);
   sdir->GetObject(collname2+"/effic_vertz_fwdneg",sh3);

   rh1->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   sh1->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   rh2->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   sh2->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   rh3->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   sh3->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);

   canvas = new TCanvas("Tracks12","Tracks: Pileup (2)",1000,1400);

   //fixRangeY(rh1,sh1);
   //fixRangeY(rh2,sh2);
   rh1->GetXaxis()->SetTitle("Primary Interaction Vertex Z (cm)");
   rh2->GetXaxis()->SetTitle("Primary Interaction Vertex Z (cm)");
   rh3->GetXaxis()->SetTitle("Primary Interaction Vertex Z (cm)");

   TH1 * r[3]={rh1,rh2,rh3};
   TH1 * s[3]={sh1,sh2,sh3};

   plotBuilding(canvas,s, r,3,
                te,"UU",-1);
   canvas->cd();
   //TPaveText* text = new TPaveText(0.25,0.72,0.75,0.77,"prova");
   //text->SetFillColor(0);
   //text->SetTextColor(1);
   //text->Draw();
   l = new TLegend(0.10,0.64,0.90,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print("pileup2.pdf");
   delete l;

   //===== Pileup 3
   rdir->GetObject(collname1+"/fakerate_vertcount_entire",rh1);
   sdir->GetObject(collname2+"/fakerate_vertcount_entire",sh1);
   rdir->GetObject(collname1+"/fakerate_vertcount_barrel",rh2);
   sdir->GetObject(collname2+"/fakerate_vertcount_barrel",sh2);
   rdir->GetObject(collname1+"/fakerate_vertcount_fwdpos",rh3);
   sdir->GetObject(collname2+"/fakerate_vertcount_fwdpos",sh3);
   rdir->GetObject(collname1+"/fakerate_vertcount_fwdneg",rh4);
   sdir->GetObject(collname2+"/fakerate_vertcount_fwdneg",sh4);

   rh1->GetYaxis()->SetRangeUser(0.0,0.01);
   sh1->GetYaxis()->SetRangeUser(0.0,0.01);
   rh2->GetYaxis()->SetRangeUser(0.0,0.01);
   sh2->GetYaxis()->SetRangeUser(0.0,0.01);
   rh3->GetYaxis()->SetRangeUser(0.0,0.01);
   sh3->GetYaxis()->SetRangeUser(0.0,0.01);
   rh4->GetYaxis()->SetRangeUser(0.0,0.01);
   sh4->GetYaxis()->SetRangeUser(0.0,0.01);

   canvas = new TCanvas("Tracks13","Tracks: Pileup (1)",1000,1400);
   
   rh1->GetXaxis()->SetTitle("Number of Pileup Vertices (In time)");
   rh2->GetXaxis()->SetTitle("Number of Pileup Vertices (In time)");
   rh3->GetXaxis()->SetTitle("Number of Pileup Vertices (In time)");
   rh4->GetXaxis()->SetTitle("Number of Pileup Vertices (In time)");

   TH1 * r[4]={rh1,rh2,rh3,rh3};
   TH1 * s[4]={sh1,sh2,sh3,sh3};

   plotBuilding(canvas,s, r,4,
                te,"UU",-1);
   canvas->cd();
   l = new TLegend(0.10,0.64,0.90,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print("pileup3.pdf");
   delete l;
    */


 }
}


void NormalizeHistograms(TH1* h1, TH1* h2)
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



void plot4histos(TCanvas *canvas, 
		TH1 *s1,TH1 *r1, TH1 *s2,TH1 *r2, 
		TH1 *s3,TH1 *r3, TH1 *s4,TH1 *r4,
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
  setStats(s1,r1, 0.6, 0.65, false);
  r1->Draw();
  s1->Draw("sames");

  canvas->cd(2);
  setStats(s2,r2, 0.6, 0.65, false);
  r2->Draw();
  s2->Draw("sames");

  canvas->cd(3);
  setStats(s3,r3, 0.6, 0.65, false);
  r3->Draw();
  s3->Draw("sames");

  canvas->cd(4);
  setStats(s4,r4, 0.6, 0.65, false);
  r4->Draw();
  s4->Draw("sames");

}

void plotBuilding(TCanvas *canvas, TH1 **s, TH1 **r, int n,TText* te,
		  char * option, double startingY, double startingX = .1,bool fit = false, unsigned int logx=0){
  canvas->Divide(2,(n+1)/2); //this should work also for odd n
  for(int i=0; i<n;i++){
    s[i]->SetMarkerStyle(20);
    r[i]->SetMarkerStyle(21);
    s[i]->SetMarkerColor(2);
    r[i]->SetMarkerColor(4);
    s[i]->SetMarkerSize(0.7);
    r[i]->SetMarkerSize(0.7);
    s[i]->SetLineColor(1);
    r[i]->SetLineColor(1);
    s[i]->SetLineWidth(1);
    r[i]->SetLineWidth(1);

    TPad *pad=canvas->cd(i+1);
    setStats(s[i],r[i], -1, 0, false);
    if((logx>>i)&1)pad->SetLogx();
    r[i]->Draw();
    s[i]->Draw("sames");
  }

//   //setStats(r1,s1, startingY, startingX, fit);
//   canvas->cd(1);
//   setStats(s1,r1, -1, 0, false);
//   r1->Draw();
//   s1->Draw("sames");

//   canvas->cd(2);
//   setStats(s2,r2, -1, 0, false);
//   r2->Draw();
//   s2->Draw("sames");

//   canvas->cd(3);
//   setStats(r3,s3, -1, 0, false);
//   r3->Draw();
//   s3->Draw("sames");

//   canvas->cd(4);
//   if(logx)gPad->SetLogx();
//   setStats(s4,r4, 0.6, 0.65, false);
//   r4->Draw();
//   s4->Draw("sames");

//   canvas->cd(5);
//   setStats(r5,s5, -1, 0, false);
//   r5->Draw();
//   s5->Draw("sames");


//   canvas->cd(6);
//   setStats(s6,r6, 0.6, 0.65, false);
//   r6->Draw();
//   s6->Draw("sames");
}

void plotPulls(TCanvas *canvas, 
	       TH1 *s1,TH1 *r1, TH1 *s2,TH1 *r2, 
	       TH1 *s3,TH1 *r3, TH1 *s4,TH1 *r4,
	       TH1 *s5,TH1 *r5,TH1 *s6,TH1 *r6,
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
  s2->SetMarkerSize(0.7);
  r2->SetMarkerSize(0.7);
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
  s3->SetLineWidth(2);
  r3->SetLineWidth(2);

  s4->SetMarkerStyle(20);
  r4->SetMarkerStyle(21);
  s4->SetMarkerColor(2);
  r4->SetMarkerColor(4);
  s4->SetMarkerSize(0.7);
  r4->SetMarkerSize(0.7);
  s4->SetLineColor(2);
  r4->SetLineColor(4);
  s4->SetLineWidth(2);
  r4->SetLineWidth(2);


  s5->SetMarkerStyle(20);
  r5->SetMarkerStyle(21);
  s5->SetMarkerColor(2);
  r5->SetMarkerColor(4);
  s5->SetMarkerSize(0.7);
  r5->SetMarkerSize(0.7);
  s5->SetLineColor(2);
  r5->SetLineColor(4);
  s5->SetLineWidth(2);
  r5->SetLineWidth(2);


  s6->SetMarkerStyle(20);
  r6->SetMarkerStyle(21);
  s6->SetMarkerColor(2);
  r6->SetMarkerColor(4);
  s6->SetMarkerSize(0.7);
  r6->SetMarkerSize(0.7);
  s6->SetLineColor(2);
  r6->SetLineColor(4);
  s6->SetLineWidth(2);
  r6->SetLineWidth(2);



  //setStats(r1,s1, startingY, startingX, fit);


  canvas->cd(1);
  setStats(s1,r1, 0.6, 0.65, true);
  r1->Draw();
  s1->Draw("sames");

  canvas->cd(2);
  setStats(s2,r2, 0.6, 0.65, true);
  r2->Draw();
  s2->Draw("sames");

  canvas->cd(3);
  setStats(s3,r3, 0.6, 0.65, true);
  r3->Draw();
  s3->Draw("sames");

  canvas->cd(4);
  setStats(s4,r4, 0.6, 0.65, true);
  r4->Draw();
  s4->Draw("sames");

  canvas->cd(5);
  setStats(s5,r5, 0.6, 0.65, true);
  r5->Draw();
  s5->Draw("sames");


  canvas->cd(6);
  setStats(s6,r6, 0.6, 0.65, true);
  r6->Draw();
  s6->Draw("sames");
}

TH1* getEfficiency(TH1* numer, TH1* denom, const char* name, const char* title, bool isEff) {
    TH1* efficiencyHist = denom->Clone(name);
    efficiencyHist->SetTitle(title);

    for (int iBinX = 1; iBinX < denom->GetNbinsX()+1; iBinX++){
        float numerVal = numer->GetBinContent(iBinX);
        float denomVal = denom->GetBinContent(iBinX);

        float effVal = 0;
        if (!isEff) effVal = denomVal ? (1 - numerVal / denomVal) : 0;
        else effVal = denomVal ? numerVal / denomVal : 0;

        float errVal = (denomVal && (effVal <=1)) ? sqrt(effVal*(1-effVal)/denomVal) : 0;

        efficiencyHist->SetBinContent(iBinX, effVal);
        efficiencyHist->SetBinError(iBinX, errVal);
    }

    return efficiencyHist; 
}

TH1* getEfficiency(TH1* numer1, TH1* numer2, TH1* denom, const char* name, const char* title, bool isEff) {
    TH1* efficiencyHist = denom->Clone(name);
    efficiencyHist->SetTitle(title);

    for (int iBinX = 1; iBinX < denom->GetNbinsX()+1; iBinX++){
        float numerVal = numer1->GetBinContent(iBinX) - numer2->GetBinContent(iBinX);
        float denomVal = denom->GetBinContent(iBinX);

        float effVal = 0;
        if (!isEff) effVal = denomVal ? (1 - numerVal / denomVal) : 0;
        else effVal = denomVal ? numerVal / denomVal : 0;

        float errVal = (denomVal && (effVal <=1)) ? sqrt(effVal*(1-effVal)/denomVal) : 0;

        efficiencyHist->SetBinContent(iBinX, effVal);
        efficiencyHist->SetBinError(iBinX, errVal);
    }

    return efficiencyHist;
}


void plotResolutions(TCanvas *canvas, 
		     TH2F *s1_2,TH2F *r1_2, TH2F *s2_2,TH2F *r2_2, 
		     TH2F *s3_2,TH2F *r3_2, TH2F *s4_2,TH2F *r4_2,
		     TH2F *s5_2,TH2F *r5_2,
             const char* ystr1, const char* ystr2, const char* ystr3, const char* ystr4, const char* ystr5, const char* xstr,
		     TText* te,
		     char * option, double startingY, bool logx=false, double startingX = .1,bool fit = false){
  canvas->Divide(2,3);

  s1_2->FitSlicesY();  
  s2_2->FitSlicesY();  
  s3_2->FitSlicesY();  
  s4_2->FitSlicesY();  
  s5_2->FitSlicesY();  

  TH1* s1_sigma = (TH1*)gDirectory->Get((std::string(s1_2->GetName())+"_2").c_str());
  TH1* s2_sigma = (TH1*)gDirectory->Get((std::string(s2_2->GetName())+"_2").c_str());
  TH1* s3_sigma = (TH1*)gDirectory->Get((std::string(s3_2->GetName())+"_2").c_str());
  TH1* s4_sigma = (TH1*)gDirectory->Get((std::string(s4_2->GetName())+"_2").c_str());
  TH1* s5_sigma = (TH1*)gDirectory->Get((std::string(s5_2->GetName())+"_2").c_str());
   
  s1_sigma->SetName("s1_sigma");
  s2_sigma->SetName("s2_sigma");
  s3_sigma->SetName("s3_sigma");
  s4_sigma->SetName("s4_sigma");
  s5_sigma->SetName("s5_sigma");
   
  s1_sigma->SetTitle("");
  s2_sigma->SetTitle("");
  s3_sigma->SetTitle("");
  s4_sigma->SetTitle("");
  s5_sigma->SetTitle("");
   
  r1_2->FitSlicesY();  
  r2_2->FitSlicesY();  
  r3_2->FitSlicesY();  
  r4_2->FitSlicesY();  
  r5_2->FitSlicesY();  

  TH1* r1_sigma = (TH1*)gDirectory->Get((std::string(r1_2->GetName())+"_2").c_str());
  TH1* r2_sigma = (TH1*)gDirectory->Get((std::string(r2_2->GetName())+"_2").c_str());
  TH1* r3_sigma = (TH1*)gDirectory->Get((std::string(r3_2->GetName())+"_2").c_str());
  TH1* r4_sigma = (TH1*)gDirectory->Get((std::string(r4_2->GetName())+"_2").c_str());
  TH1* r5_sigma = (TH1*)gDirectory->Get((std::string(r5_2->GetName())+"_2").c_str());
   
  r1_sigma->SetName("r1_sigma");
  r2_sigma->SetName("r2_sigma");
  r3_sigma->SetName("r3_sigma");
  r4_sigma->SetName("r4_sigma");
  r5_sigma->SetName("r5_sigma");
   
  r1_sigma->SetTitle("");
  r2_sigma->SetTitle("");
  r3_sigma->SetTitle("");
  r4_sigma->SetTitle("");
  r5_sigma->SetTitle("");

  s1_sigma->GetYaxis()->SetTitle(ystr1);     
  s2_sigma->GetYaxis()->SetTitle(ystr2);     
  s3_sigma->GetYaxis()->SetTitle(ystr3);     
  s4_sigma->GetYaxis()->SetTitle(ystr4);     
  s5_sigma->GetYaxis()->SetTitle(ystr5);     
   
  r1_sigma->GetYaxis()->SetTitle(ystr1);     
  r2_sigma->GetYaxis()->SetTitle(ystr2);     
  r3_sigma->GetYaxis()->SetTitle(ystr3);     
  r4_sigma->GetYaxis()->SetTitle(ystr4);     
  r5_sigma->GetYaxis()->SetTitle(ystr5);     
   
  s1_sigma->GetXaxis()->SetTitle(xstr);     
  s2_sigma->GetXaxis()->SetTitle(xstr);     
  s3_sigma->GetXaxis()->SetTitle(xstr);     
  s4_sigma->GetXaxis()->SetTitle(xstr);     
  s5_sigma->GetXaxis()->SetTitle(xstr);     
   
  r1_sigma->GetXaxis()->SetTitle(xstr);     
  r2_sigma->GetXaxis()->SetTitle(xstr);     
  r3_sigma->GetXaxis()->SetTitle(xstr);     
  r4_sigma->GetXaxis()->SetTitle(xstr);     
  r5_sigma->GetXaxis()->SetTitle(xstr);     
   
  //TH1* s1_sigma = s1_2_2->Clone("s1_sigma");
  //TH1* s2_sigma = s1_2_2->Clone("s2_sigma");
  //TH1* s3_sigma = s1_2_2->Clone("s3_sigma");
  //TH1* s4_sigma = s1_2_2->Clone("s4_sigma");
  //TH1* s5_sigma = s1_2_2->Clone("s5_sigma");
  //
  //TH1* r1_sigma = r1_2_2->Clone("s1_sigma");
  //TH1* r2_sigma = r1_2_2->Clone("s2_sigma");
  //TH1* r3_sigma = r1_2_2->Clone("s3_sigma");
  //TH1* r4_sigma = r1_2_2->Clone("s4_sigma");
  //TH1* r5_sigma = r1_2_2->Clone("s5_sigma");

  s1_sigma->SetMarkerStyle(20);
  r1_sigma->SetMarkerStyle(21);
  s1_sigma->SetMarkerColor(2);
  r1_sigma->SetMarkerColor(4);
  s1_sigma->SetMarkerSize(0.7);
  r1_sigma->SetMarkerSize(0.7);
  s1_sigma->SetLineColor(1);
  r1_sigma->SetLineColor(1);

  s2_sigma->SetMarkerStyle(20);
  r2_sigma->SetMarkerStyle(21);
  s2_sigma->SetMarkerColor(2);
  r2_sigma->SetMarkerColor(4);
  s2_sigma->SetMarkerSize(0.7);
  r2_sigma->SetMarkerSize(0.7);
  s2_sigma->SetLineColor(1);
  r2_sigma->SetLineColor(1);

  s3_sigma->SetMarkerStyle(20);
  r3_sigma->SetMarkerStyle(21);
  s3_sigma->SetMarkerColor(2);
  r3_sigma->SetMarkerColor(4);
  s3_sigma->SetMarkerSize(0.7);
  r3_sigma->SetMarkerSize(0.7);
  s3_sigma->SetLineColor(1);
  r3_sigma->SetLineColor(1);

  s4_sigma->SetMarkerStyle(20);
  r4_sigma->SetMarkerStyle(21);
  s4_sigma->SetMarkerColor(2);
  r4_sigma->SetMarkerColor(4);
  s4_sigma->SetMarkerSize(0.7);
  r4_sigma->SetMarkerSize(0.7);
  s4_sigma->SetLineColor(1);
  r4_sigma->SetLineColor(1);


  s5_sigma->SetMarkerStyle(20);
  r5_sigma->SetMarkerStyle(21);
  s5_sigma->SetMarkerColor(2);
  r5_sigma->SetMarkerColor(4);
  s5_sigma->SetMarkerSize(0.7);
  r5_sigma->SetMarkerSize(0.7);
  s5_sigma->SetLineColor(1);
  r5_sigma->SetLineColor(1);

  r1_sigma->GetYaxis()->SetRangeUser(0.000009,0.01);
  s1_sigma->GetYaxis()->SetRangeUser(0.000009,0.01);
  r2_sigma->GetYaxis()->SetRangeUser(0.00009,0.03);
  s2_sigma->GetYaxis()->SetRangeUser(0.00009,0.03);
  r3_sigma->GetYaxis()->SetRangeUser(0.00009,0.05);  
  s3_sigma->GetYaxis()->SetRangeUser(0.00009,0.05);
  r4_sigma->GetYaxis()->SetRangeUser(0.0009,0.1);  
  s4_sigma->GetYaxis()->SetRangeUser(0.0009,0.1);

  r1_sigma->GetYaxis()->SetTitleSize(0.05);
  r1_sigma->GetYaxis()->SetTitleOffset(1.2);
  r1_sigma->GetXaxis()->SetTitleSize(0.07);
  r1_sigma->GetXaxis()->SetTitleOffset(0.6);

  r2_sigma->GetYaxis()->SetTitleSize(0.05);
  r2_sigma->GetYaxis()->SetTitleOffset(1.2);
  r2_sigma->GetXaxis()->SetTitleSize(0.07);
  r2_sigma->GetXaxis()->SetTitleOffset(0.6);

  r3_sigma->GetYaxis()->SetTitleSize(0.05);
  r3_sigma->GetYaxis()->SetTitleOffset(1.2);
  r3_sigma->GetXaxis()->SetTitleSize(0.07);
  r3_sigma->GetXaxis()->SetTitleOffset(0.6);

  r4_sigma->GetYaxis()->SetTitleSize(0.05);
  r4_sigma->GetYaxis()->SetTitleOffset(1.2);
  r4_sigma->GetXaxis()->SetTitleSize(0.07);
  r4_sigma->GetXaxis()->SetTitleOffset(0.6);

  r5_sigma->GetYaxis()->SetTitleSize(0.05);
  r5_sigma->GetYaxis()->SetTitleOffset(1.2);
  r5_sigma->GetXaxis()->SetTitleSize(0.07);
  r5_sigma->GetXaxis()->SetTitleOffset(0.6);


  //setStats(r1,s1, startingY, startingX, fit);
  canvas->cd(1);
  gPad->SetLogy(); 
  if(logx)gPad->SetLogx();
  setStats(r1_sigma,s1_sigma, -1, 0, false);
  r1_sigma->Draw();
  s1_sigma->Draw("sames");

  canvas->cd(2);
  gPad->SetLogy(); 
  if(logx)gPad->SetLogx();
  setStats(r2_sigma,s2_sigma, -1, 0, false);
  r2_sigma->Draw();
  s2_sigma->Draw("sames");

  canvas->cd(3);
  gPad->SetLogy(); 
  if(logx)gPad->SetLogx();
  setStats(r3_sigma,s3_sigma, -1, 0, false);
  r3_sigma->Draw();
  s3_sigma->Draw("sames");

  canvas->cd(4);
  gPad->SetLogy(); 
  if(logx)gPad->SetLogx();
  setStats(r4_sigma,s4_sigma, -1, 0, false);
  r4_sigma->Draw();
  s4_sigma->Draw("sames");

  canvas->cd(5);
  gPad->SetLogy(); 
  if(logx)gPad->SetLogx();
  setStats(r5_sigma,s5_sigma, -1, 0, false);
  r5_sigma->Draw();
  s5_sigma->Draw("sames");


  //canvas->cd(6);
  //r6->Draw();
  //s6->Draw("sames");





}

void plotResolutionsDirect(TCanvas *canvas, 
             TH1 *s1,TH1 *r1, TH1 *s2,TH1 *r2, 
             TH1 *s3,TH1 *r3, TH1 *s4,TH1 *r4,
             TH1 *s5,TH1 *r5,TH1 *s6,TH1 *r6,
             TText* te,
             char * option, double startingY, bool logx=false, double startingX = .1,bool fit = false){
  canvas->Divide(2,3);

  s1->SetMarkerStyle(20);
  r1->SetMarkerStyle(21);
  s1->SetMarkerColor(2);
  r1->SetMarkerColor(4);
  s1->SetMarkerSize(0.7);
  r1->SetMarkerSize(0.7);
  s1->SetLineColor(1);
  r1->SetLineColor(1);

  s2->SetMarkerStyle(20);
  r2->SetMarkerStyle(21);
  s2->SetMarkerColor(2);
  r2->SetMarkerColor(4);
  s2->SetMarkerSize(0.7);
  r2->SetMarkerSize(0.7);
  s2->SetLineColor(1);
  r2->SetLineColor(1);

  s3->SetMarkerStyle(20);
  r3->SetMarkerStyle(21);
  s3->SetMarkerColor(2);
  r3->SetMarkerColor(4);
  s3->SetMarkerSize(0.7);
  r3->SetMarkerSize(0.7);
  s3->SetLineColor(1);
  r3->SetLineColor(1);

  s4->SetMarkerStyle(20);
  r4->SetMarkerStyle(21);
  s4->SetMarkerColor(2);
  r4->SetMarkerColor(4);
  s4->SetMarkerSize(0.7);
  r4->SetMarkerSize(0.7);
  s4->SetLineColor(1);
  r4->SetLineColor(1);


  s5->SetMarkerStyle(20);
  r5->SetMarkerStyle(21);
  s5->SetMarkerColor(2);
  r5->SetMarkerColor(4);
  s5->SetMarkerSize(0.7);
  r5->SetMarkerSize(0.7);
  s5->SetLineColor(1);
  r5->SetLineColor(1);

  s6->SetMarkerStyle(20);
  r6->SetMarkerStyle(21);
  s6->SetMarkerColor(2);
  r6->SetMarkerColor(4);
  s6->SetMarkerSize(0.7);
  r6->SetMarkerSize(0.7);
  s6->SetLineColor(1);
  r6->SetLineColor(1);
  s6->SetLineWidth(2);
  r6->SetLineWidth(2);



  //setStats(r1,s1, startingY, startingX, fit);
  canvas->cd(1);
  gPad->SetLogy(); 
  if(logx)gPad->SetLogx();
  setStats(r1,s1, -1, 0, false);
  r1->Draw();
  s1->Draw("sames");

  canvas->cd(2);
  gPad->SetLogy(); 
  if(logx)gPad->SetLogx();
  setStats(r2,s2, -1, 0, false);
  r2->Draw();
  s2->Draw("sames");

  canvas->cd(3);
  gPad->SetLogy(); 
  if(logx)gPad->SetLogx();
  setStats(r3,s3, -1, 0, false);
  r3->Draw();
  s3->Draw("sames");

  canvas->cd(4);
  gPad->SetLogy(); 
  if(logx)gPad->SetLogx();
  setStats(r4,s4, -1, 0, false);
  r4->Draw();
  s4->Draw("sames");

  canvas->cd(5);
  gPad->SetLogy(); 
  if(logx)gPad->SetLogx();
  setStats(r5,s5, -1, 0, false);
  r5->Draw();
  s5->Draw("sames");


  //canvas->cd(6);
  //r6->Draw();
  //s6->Draw("sames");


}

void setStats(TH1* s,TH1* r, double startingY, double startingX = .1,bool fit){
  if (startingY<0){
    s->SetStats(0);
    r->SetStats(0);
  } else {
    //gStyle->SetOptStat(1001);
    s->SetStats(1);
    r->SetStats(1);

    if (fit){
      s->Fit("gaus");
      TF1* f1 = (TF1*) s->GetListOfFunctions()->FindObject("gaus");
      f1->SetLineColor(2);
      f1->SetLineWidth(1);
    }
    s->Draw();
    gPad->Update(); 
    TPaveStats* st1 = (TPaveStats*) s->GetListOfFunctions()->FindObject("stats");
    if (fit) {st1->SetOptFit(0010);    st1->SetOptStat(1001);}
    st1->SetX1NDC(startingX);
    st1->SetX2NDC(startingX+0.30);
    st1->SetY1NDC(startingY+0.20);
    st1->SetY2NDC(startingY+0.35);
    st1->SetTextColor(2);
    if (fit) {
      r->Fit("gaus");
      TF1* f2 = (TF1*) r->GetListOfFunctions()->FindObject("gaus");
      f2->SetLineColor(4);
      f2->SetLineWidth(1);    
    }
    r->Draw();
    gPad->Update(); 
    TPaveStats* st2 = (TPaveStats*) r->GetListOfFunctions()->FindObject("stats");
    if (fit) {st2->SetOptFit(0010);    st2->SetOptStat(1001);}
    st2->SetX1NDC(startingX);
    st2->SetX2NDC(startingX+0.30);
    st2->SetY1NDC(startingY);
    st2->SetY2NDC(startingY+0.15);
    st2->SetTextColor(4);
  }
}
  
void fixRangeY(TH1* r,TH1* s){
  double ymin = (r->GetBinContent(r->GetMinimumBin()) < s->GetBinContent(s->GetMinimumBin())) ?
    r->GetBinContent(r->GetMinimumBin()) : s->GetBinContent(s->GetMinimumBin());
  double ymax = (r->GetBinContent(r->GetMaximumBin()) > s->GetBinContent(s->GetMaximumBin())) ?
    r->GetBinContent(r->GetMaximumBin()) : s->GetBinContent(s->GetMaximumBin());
  r->GetYaxis()->SetRangeUser(ymin*0.9,ymax*1.1);
  s->GetYaxis()->SetRangeUser(ymin*0.9,ymax*1.1);
}


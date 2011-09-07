void HPIterTrackValHistoPublisher(char* newFile="NEW_FILE",char* refFile="REF_FILE")
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
 TString coll8hitname2 =sl->At(0)->GetName();

 // select high purity tracks to make the usual validation plots for
 int num_col = sl->GetSize();
 for(int i=0; i<num_col;i++){
   collname2 =sl->At(i)->GetName();
   if(collname2.Contains("cutsRecoHp")) break;
 }
 cout << "collname2 = " << collname2 << endl;
 for(int i=0; i<num_col;i++){
   coll8hitname2 =sl->At(i)->GetName();
   if(coll8hitname2.Contains("wbtagc")) break;
 }
 cout << "coll8hitname2 = " << coll8hitname2 << endl;

 if(rfile->GetDirectory("DQMData/Run 1/RecoTrackV")) rfile->cd("DQMData/Run 1/RecoTrackV/Run summary/Track");
 else if(rfile->GetDirectory("DQMData/RecoTrackV/Track"))rfile->cd("DQMData/RecoTrackV/Track");
 else if(rfile->GetDirectory("DQMData/Run 1/Tracking")) rfile->cd("DQMData/Run 1/Tracking/Run summary/Track");
 else if(rfile->GetDirectory("DQMData/Tracking/Track"))rfile->cd("DQMData/Tracking/Track");
 rdir=gDirectory;
 TList *rl= rdir->GetListOfKeys();
 TString collname1=rl->At(0)->GetName(); 
 TString coll8hitname1=rl->At(0)->GetName();

 num_col = rl->GetSize();
 for(int i=0; i<num_col;i++){
   collname1 =rl->At(i)->GetName();
   if(collname1.Contains("cutsRecoHp")) break;
 }
 cout << "collname1 = " << collname1 << endl;
 for(int i=0; i<num_col;i++){
   coll8hitname1 =rl->At(i)->GetName();
   if(coll8hitname1.Contains("wbtagc")) break;
 }
 cout << "coll8hitname1 = " << coll8hitname1 << endl;
 int num_col = rl->GetSize();
 for(int i=0; i<num_col;i++){
   cout << " >>>>>>>>>dir " << i << " name = " << rl->At(i)->GetName() << endl;
 }

 //HistoCompare_Tracks * myPV = new HistoCompare_Tracks();

 TCanvas *canvas;

 TH1F *sh1,*rh1;
 TH1F *sh2,*rh2;
 TH1F *sh3,*rh3;
 TH1F *sh4,*rh4;
 TH1F *sh5,*rh5;
 TH1F *sh6,*rh6;

 TH1F *sc1,*rc1;
 TH1F *sc2,*rc2;
 TH1F *sc3,*rc3;

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
   rdir->GetObject(collname1+"/fakerate",rh2);
   sdir->GetObject(collname2+"/fakerate",sh2);
   rh2->GetYaxis()->SetRangeUser(0.,MAXFAKEETA);
   sh2->GetYaxis()->SetRangeUser(0.,MAXFAKEETA);
   rh2->GetXaxis()->SetTitle("#eta");
   rh2->GetYaxis()->SetTitle("fakerate vs #eta");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);
//   rh2->GetYaxis()->SetRangeUser(0.,.70);
//   sh2->GetYaxis()->SetRangeUser(0.,.70);


   rdir->GetObject(collname1+"/efficPt",rh3);
   sdir->GetObject(collname2+"/efficPt",sh3);
   rh3->GetYaxis()->SetRangeUser(0.,1.0);
   sh3->GetYaxis()->SetRangeUser(0.,1.0);
   rh3->GetXaxis()->SetRangeUser(0,300);
   sh3->GetXaxis()->SetRangeUser(0,300);
   rh3->GetXaxis()->SetTitle("p_{t}");
   rh3->GetYaxis()->SetTitle("efficiency vs p_{t}");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rh3->SetTitle("");
   rdir->GetObject(collname1+"/fakeratePt",rh4);
   sdir->GetObject(collname2+"/fakeratePt",sh4);
   rh4->SetTitle("");
   rh4->GetXaxis()->SetTitle("p_{t}");
   rh4->GetYaxis()->SetTitle("fakrate vs p_{t}");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);
   rh4->GetYaxis()->SetRangeUser(0.,MAXFAKEPT);
   sh4->GetYaxis()->SetRangeUser(0.,MAXFAKEPT);
   rh4->GetXaxis()->SetRangeUser(0.2,300);
   sh4->GetXaxis()->SetRangeUser(0.2,300);


   rdir->GetObject(collname1+"/effic_vs_hit",rh5);
   sdir->GetObject(collname2+"/effic_vs_hit",sh5);
   rh5->GetXaxis()->SetTitle("hits");
   rh5->GetYaxis()->SetTitle("efficiency vs hits");
   rh5->GetYaxis()->SetTitleSize(0.05);
   rh5->GetYaxis()->SetTitleOffset(1.2);
   rh5->GetYaxis()->SetRangeUser(0.,1.0);
   sh5->GetYaxis()->SetRangeUser(0.,1.0);
   //rh3->GetXaxis()->SetRangeUser(0,30);
   //sh3->GetXaxis()->SetRangeUser(0,30);
   rdir->GetObject(collname1+"/fakerate_vs_hit",rh6);
   sdir->GetObject(collname2+"/fakerate_vs_hit",sh6);
   rh6->GetYaxis()->SetRangeUser(0.,MAXFAKEHIT);
   rh6->GetYaxis()->SetRangeUser(0.,MAXFAKEHIT);
   rh6->GetXaxis()->SetTitle("hits");
   rh6->GetYaxis()->SetTitle("fakerate vs hits");
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

   TH1F * r[6]={rh1,rh2,rh3,rh4,rh5,rh6};
   TH1F * s[6]={sh1,sh2,sh3,sh4,sh5,sh6};

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
   canvas->Print("building.pdf");   
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
   rh2->GetXaxis()->SetRangeUser(0,30);
   sh2->GetXaxis()->SetRangeUser(0,30);
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

   plotResolutions(canvas,
		   sh1,rh1,sh2,rh2,
		   sh3,rh3,sh4,rh4,
		   sh5,rh5,sh6,rh6,
		   te,"UU",-1);
   
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



   /* plotResolutions(canvas,
	     sh1,rh1,sh2,rh2,
	     sh3,rh3,sh4,rh4,
	     sh5,rh5,sh6,rh6,
	     te,"UU",-1);
   */
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
   rdir->GetObject(collname1+"/phires_vs_pt_Sigma",rh1);
   sdir->GetObject(collname2+"/phires_vs_pt_Sigma",sh1);

   rdir->GetObject(collname1+"/cotThetares_vs_pt_Sigma",rh2);
   sdir->GetObject(collname2+"/cotThetares_vs_pt_Sigma",sh2);


   rdir->GetObject(collname1+"/dxyres_vs_pt_Sigma",rh3);
   sdir->GetObject(collname2+"/dxyres_vs_pt_Sigma",sh3);

   rdir->GetObject(collname1+"/dzres_vs_pt_Sigma",rh4);
   sdir->GetObject(collname2+"/dzres_vs_pt_Sigma",sh4);

   rdir->GetObject(collname1+"/ptres_vs_pt_Sigma",rh5);
   sdir->GetObject(collname2+"/ptres_vs_pt_Sigma",sh5);

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
	     sh1,rh1,sh2,rh2,
	     sh3,rh3,sh4,rh4,
	     sh5,rh5,sh6,rh6,
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
   rdir->GetObject(collname1+"/fakerate_vs_phi",rh2);
   sdir->GetObject(collname2+"/fakerate_vs_phi",sh2);
   rh2->GetXaxis()->SetTitle("#phi");
   rh2->GetYaxis()->SetTitle("fakerate vs #phi");
   rh2->GetYaxis()->SetRangeUser(0.,MAXFAKEETA);
   sh2->GetYaxis()->SetRangeUser(0.,MAXFAKEETA);
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
   rdir->GetObject(collname1+"/fakerate_vs_dxy",rh4);
   sdir->GetObject(collname2+"/fakerate_vs_dxy",sh4);
   rh4->SetTitle("");
   rh4->GetXaxis()->SetTitle("dxy");
   rh4->GetYaxis()->SetTitle("fakerate vs dxy");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);
   rh4->GetYaxis()->SetRangeUser(0.,MAXFAKEETA);
   sh4->GetYaxis()->SetRangeUser(0.,MAXFAKEETA);


   rdir->GetObject(collname1+"/effic_vs_dz",rh5);
   sdir->GetObject(collname2+"/effic_vs_dz",sh5);
   rh5->GetXaxis()->SetTitle("dz");
   rh5->GetYaxis()->SetTitle("efficiency vs dz");
   rh5->GetYaxis()->SetTitleSize(0.05);
   rh5->GetYaxis()->SetTitleOffset(1.2);
   rh5->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   sh5->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   rdir->GetObject(collname1+"/fakerate_vs_dz",rh6);
   sdir->GetObject(collname2+"/fakerate_vs_dz",sh6);
   rh6->GetYaxis()->SetRangeUser(0.,1.0);
   rh6->GetYaxis()->SetRangeUser(0.,1.0);
   rh6->GetXaxis()->SetTitle("dz");
   rh6->GetYaxis()->SetTitle("fakerate vs dz");
   rh6->GetYaxis()->SetTitleSize(0.05);
   rh6->GetYaxis()->SetTitleOffset(1.2);

   canvas = new TCanvas("Tracks8","Tracks: efficiency & fakerate",1000,1400);

   TH1F * r[6]={rh1,rh2,rh3,rh4,rh5,rh6};
   TH1F * s[6]={sh1,sh2,sh3,sh4,sh5,sh6};

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
   canvas->Print("building2.pdf");   
   delete l;

    //===== building 3
   rdir->GetObject(collname1+"/effic_vs_vertpos",rh1);
   sdir->GetObject(collname2+"/effic_vs_vertpos",sh1);
   rh1->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   sh1->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   rh1->GetXaxis()->SetTitle("TP vert xy pos");
   rh1->GetYaxis()->SetTitle("efficiency vs vert xy pos");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/effic_vs_zpos",rh2);
   sdir->GetObject(collname2+"/effic_vs_zpos",sh2);
   rh2->GetXaxis()->SetTitle("TP vert z pos");
   rh2->GetYaxis()->SetTitle("efficiency vs  vert z pos");
   rh2->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   sh2->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);
   rh2->SetTitle("");

   canvas = new TCanvas("Tracks9","Tracks: efficiency & fakerate",1000,1400);

   TH1F * r[2]={rh1,rh2};
   TH1F * s[2]={sh1,sh2};

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
   canvas->Print("building3.pdf");   
   delete l;

/* doesn't work with 363 files
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
*/

   //======== plots of eff, fake rates for various iterative tracking steps
   //== we assume that the ref and new have the same histogram list! (and there are 5!)
   // HERE
   TH1F * rhp[5];
   TH1F * shp[5];
   // fill with the histos we want for eff
   fillStepHisto(rdir, sdir, rl, sl, rhp, shp, "/effic", "#eta", "efficiency vs #eta", 0., MAXEFF);
   // rearrange to a more natural order for plotting
   TH1F * rr[5]={rhp[3],rhp[0],rhp[4],rhp[1],rhp[2]};
   TH1F * ss[5]={shp[3],shp[0],shp[4],shp[1],shp[2]};

   canvas = new TCanvas("Tracks11","Tracks: efficiency for each step",1000,1400);

   plotBuilding(canvas,ss, rr,5,
		te,"UU",-69);

   canvas->cd();
   l = new TLegend(0.10,0.64,0.90,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(r[0],refLabel,"LPF");
   l->AddEntry(s[0],newLabel,"LPF");
   l->Draw();
   canvas->Print("steps_eff.pdf");   
   delete l;

   // fill with the histos we want for fake rate
   fillStepHisto(rdir, sdir, rl, sl, rhp, shp, "/fakerate", "#eta", "fakerate vs #eta", 0., MAXFAKEETA);
   // rearrange to a more natural order for plotting
   TH1F * rr[5]={rhp[3],rhp[0],rhp[4],rhp[1],rhp[2]};
   TH1F * ss[5]={shp[3],shp[0],shp[4],shp[1],shp[2]};

   canvas = new TCanvas("Tracks12","Tracks: fakerate for each step",1000,1400);

   plotBuilding(canvas,ss, rr,5,
		te,"UU",-69);

   canvas->cd();
   l = new TLegend(0.10,0.64,0.90,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(r[0],refLabel,"LPF");
   l->AddEntry(s[0],newLabel,"LPF");
   l->Draw();
   canvas->Print("steps_fakerate.pdf");   
   delete l;

   // fill with the histos we want for eff vs pT
   fillStepHisto(rdir, sdir, rl, sl, rhp, shp, "/efficPt", "p_{t}", "efficiency vs p_{t}", 0., MAXEFF);
   // rearrange to a more natural order for plotting
   TH1F * rr[5]={rhp[3],rhp[0],rhp[4],rhp[1],rhp[2]};
   TH1F * ss[5]={shp[3],shp[0],shp[4],shp[1],shp[2]};

   canvas = new TCanvas("Tracks13","Tracks: efficiency for each step",1000,1400);

   plotBuilding(canvas,ss, rr,5,
		te,"UU",-69, 1, false, 127);

   canvas->cd();
   l = new TLegend(0.10,0.64,0.90,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(r[0],refLabel,"LPF");
   l->AddEntry(s[0],newLabel,"LPF");
   l->Draw();
   canvas->Print("steps_effpt.pdf");   
   delete l;

   fillStepHisto(rdir, sdir, rl, sl, rhp, shp, "/fakeratePt", "p_{t}", "fakerate vs p_{t}", 0., MAXFAKEETA);
   // rearrange to a more natural order for plotting
   TH1F * rr[5]={rhp[3],rhp[0],rhp[4],rhp[1],rhp[2]};
   TH1F * ss[5]={shp[3],shp[0],shp[4],shp[1],shp[2]};

   canvas = new TCanvas("Tracks14","Tracks: fakerate for each step",1000,1400);

   plotBuilding(canvas,ss, rr,5,
		te,"UU",-69, 1, false, 127);

   canvas->cd();
   l = new TLegend(0.10,0.64,0.90,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(r[0],refLabel,"LPF");
   l->AddEntry(s[0],newLabel,"LPF");
   l->Draw();
   canvas->Print("steps_fakeratePt.pdf");   
   delete l;

   // fill with the histos we want for eff
   fillStepHisto(rdir, sdir, rl, sl, rhp, shp, "/effic_vs_hit", "hits", "efficiency vs hits", 0., MAXEFF);
   // rearrange to a more natural order for plotting
   TH1F * rr[5]={rhp[3],rhp[0],rhp[4],rhp[1],rhp[2]};
   TH1F * ss[5]={shp[3],shp[0],shp[4],shp[1],shp[2]};

   canvas = new TCanvas("Tracks15","Tracks: efficiency for each step",1000,1400);

   plotBuilding(canvas,ss, rr,5,
		te,"UU",-69);

   canvas->cd();
   l = new TLegend(0.10,0.64,0.90,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(r[0],refLabel,"LPF");
   l->AddEntry(s[0],newLabel,"LPF");
   l->Draw();
   canvas->Print("steps_effhits.pdf");   
   delete l;

   // fill with the histos we want for fake rate
   fillStepHisto(rdir, sdir, rl, sl, rhp, shp, "/fakerate_vs_hit", "hits", "fakerate vs hits", 0., MAXFAKEETA);
   // rearrange to a more natural order for plotting
   TH1F * rr[5]={rhp[3],rhp[0],rhp[4],rhp[1],rhp[2]};
   TH1F * ss[5]={shp[3],shp[0],shp[4],shp[1],shp[2]};

   canvas = new TCanvas("Tracks16","Tracks: fakerate for each step",1000,1400);

   plotBuilding(canvas,ss, rr,5,
		te,"UU",-69);

   canvas->cd();
   l = new TLegend(0.10,0.64,0.90,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(r[0],refLabel,"LPF");
   l->AddEntry(s[0],newLabel,"LPF");
   l->Draw();
   canvas->Print("steps_fakeratehits.pdf");   
   delete l;

   // fill with the histos we want for btag hits (8 hits and pT>1)

   rdir->GetObject(coll8hitname1+"/effic",rh1);
   sdir->GetObject(coll8hitname2+"/effic",sh1);
   rh1->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   sh1->GetYaxis()->SetRangeUser(MINEFF,MAXEFF);
   rh1->GetXaxis()->SetTitle("#eta");
   rh1->GetYaxis()->SetTitle("efficiency vs #eta");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);
   rdir->GetObject(coll8hitname1+"/fakerate",rh2);
   sdir->GetObject(coll8hitname2+"/fakerate",sh2);
   rh2->GetYaxis()->SetRangeUser(0.,MAXFAKEETA);
   sh2->GetYaxis()->SetRangeUser(0.,MAXFAKEETA);
   rh2->GetXaxis()->SetTitle("#eta");
   rh2->GetYaxis()->SetTitle("fakerate vs #eta");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(coll8hitname1+"/efficPt",rh3);
   sdir->GetObject(coll8hitname2+"/efficPt",sh3);
   rh3->GetXaxis()->SetRangeUser(0,300);
   sh3->GetXaxis()->SetRangeUser(0,300);
   rh3->GetXaxis()->SetTitle("p_{t}");
   rh3->GetYaxis()->SetTitle("efficiency vs p_{t}");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rh3->SetTitle("");
   rdir->GetObject(coll8hitname1+"/fakeratePt",rh4);
   sdir->GetObject(coll8hitname2+"/fakeratePt",sh4);
   rh4->SetTitle("");
   rh4->GetXaxis()->SetTitle("p_{t}");
   rh4->GetYaxis()->SetTitle("fakrate vs p_{t}");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);
   rh4->GetYaxis()->SetRangeUser(0.,MAXFAKEPT);
   sh4->GetYaxis()->SetRangeUser(0.,MAXFAKEPT);
   rh4->GetXaxis()->SetRangeUser(0.2,300);
   sh4->GetXaxis()->SetRangeUser(0.2,300);

   canvas = new TCanvas("Tracks18","Tracks: efficiency & fakerate 8 hits pT gt 1",1000,1400);

   TH1F * r[4]={rh1,rh2,rh3,rh4};
   TH1F * s[4]={sh1,sh2,sh3,sh4};

   plotBuilding(canvas,s, r,4,
		te,"UU",-1, 1, false, 0xC);

   canvas->cd();
   //l = new TLegend(0.10,0.64,0.90,0.69);
   l = new TLegend(0.20,0.47,0.90,0.52);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print("buildingbtag.pdf");   
   delete l;
 }
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

void fillStepHisto(TDirectory *rdir, TDirectory *sdir, TList *rl, TList *sl, TH1F **r, TH1F **s, 
                   TString histotype, TString xtext, TString ytext, double ymin, double ymax) {
   TString refcollname=rl->At(0)->GetName();
   TString newcollname=sl->At(0)->GetName();
   int num_col = rl->GetSize();
   for(int i=0; i<num_col;i++){
     refcollname =rl->At(i)->GetName();
     newcollname =sl->At(i)->GetName();
     rdir->GetObject(refcollname+histotype,r[i]);
     sdir->GetObject(newcollname+histotype,s[i]);
     r[i]->GetYaxis()->SetRangeUser(ymin, ymax);
     s[i]->GetYaxis()->SetRangeUser(ymin, ymax);
     r[i]->GetXaxis()->SetTitle(xtext);
     r[i]->GetYaxis()->SetTitle(ytext);
     r[i]->GetYaxis()->SetTitleSize(0.05);
     r[i]->GetYaxis()->SetTitleOffset(1.2);
     r[i]->SetTitle(refcollname);
   }
}

void fillHitsHisto(TDirectory *rdir, TDirectory *sdir, TList *rl, TList *sl, TH1F **r, TH1F **s, 
                   TString histotype, TString xtext, TString ytext, double ymin, double ymax) {
   TString refcollname=rl->At(0)->GetName();
   TString newcollname=sl->At(0)->GetName();
   int num_col = rl->GetSize();
   int j=-1;
   for(int i=0; i<num_col;i++){
     refcollname =rl->At(i)->GetName();
     newcollname =sl->At(i)->GetName();
     
     if(refcollname.Contains("cutsRecoHp") || refcollname.Contains("w6hits")
        || refcollname.Contains("w8hits")) {
       ++j;
       rdir->GetObject(refcollname+histotype,r[j]);
       sdir->GetObject(newcollname+histotype,s[j]);
       r[j]->GetYaxis()->SetRangeUser(ymin, ymax);
       s[j]->GetYaxis()->SetRangeUser(ymin, ymax);
       r[j]->GetXaxis()->SetTitle(xtext);
       r[j]->GetYaxis()->SetTitle(ytext);
       r[j]->GetYaxis()->SetTitleSize(0.05);
       r[j]->GetYaxis()->SetTitleOffset(1.2);
       r[j]->SetTitle(refcollname);
     }
   }
}

void plotBuilding(TCanvas *canvas, TH1F **s, TH1F **r, int n,TText* te,
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
    // mov histo title for steps histos
    if(startingY == -69 && (i==2 || i==3)) {
      if (i==2) TPaveText *pt = new TPaveText(0.01,0.00,0.7,0.06,"NDC");
      else      TPaveText *pt = new TPaveText(0.01,0.00,0.6,0.06,"NDC");
      pt->SetName("title");  // This tells the histogram to NOT display its title at all
      pt->SetBorderSize(1);
      pt->SetFillColor(0);
      text = pt->AddText(r[i]->GetTitle());
      pt->Draw();
    }
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
	       TH1F *s1,TH1F *r1, TH1F *s2,TH1F *r2, 
	       TH1F *s3,TH1F *r3, TH1F *s4,TH1F *r4,
	       TH1F *s5,TH1F *r5,TH1F *s6,TH1F *r6,
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

void plotResolutions(TCanvas *canvas, 
		     TH1F *s1,TH1F *r1, TH1F *s2,TH1F *r2, 
		     TH1F *s3,TH1F *r3, TH1F *s4,TH1F *r4,
		     TH1F *s5,TH1F *r5,TH1F *s6,TH1F *r6,
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

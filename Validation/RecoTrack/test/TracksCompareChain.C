void TracksCompareChain()
{

 gROOT ->Reset();
 // gROOT ->SetBatch();
 gROOT->SetStyle("Plain");

 char*  rfilename = "validationPlots.root";//new release is in red
 char*  sfilename = "../data/validationPlots.root";//reference is in blue

 delete gROOT->GetListOfFiles()->FindObject(rfilename);
 delete gROOT->GetListOfFiles()->FindObject(sfilename); 

 TText* te = new TText();
 TFile * rfile = new TFile(rfilename);
 TFile * sfile = new TFile(sfilename);

 // create an iterator to loop through all objects(keys) in the  file
 //get release name
 TIter nextkey(rfile -> GetListOfKeys());
 TObjString *newRel;
 while (key = (TKey*)nextkey()) {
   obj = key -> ReadObj();  //use ReadObj, not Read
   
   if (obj->InheritsFrom("TObjString") ) {  //instead of obj->IsA
     newRel = (TObjString *) obj;
     newRel->Print();
   }
 }
 
 TIter nextkey(sfile -> GetListOfKeys());
 TObjString *refRel;
 while (key = (TKey*)nextkey()) {
   obj = key -> ReadObj();  //use ReadObj, not Read
   
   if (obj->InheritsFrom("TObjString") ) {  //instead of obj->IsA
     refRel = (TObjString *) obj;
     refRel->Print();
   }
 }
 TLatex thistext;
 thistext.SetTextSize(0.02);
 char relinfo[200];
 sprintf(relinfo,"RED histograms = %s *** BLUE Histograms = %s", newRel->GetName(), refRel->GetName());
 
 gROOT->ProcessLine(".x HistoCompare_Tracks.C");
 HistoCompare_Tracks * myPV = new HistoCompare_Tracks();

 TCanvas *canvas;

 TH1F *sh1,*rh1;
 TH1F *sc1,*rc1;
 TH1F *sh2,*rh2;
 TH1F *sc2,*rc2;
 TH1F *sh3,*rh3;
 TH1F *sc3,*rc3;

 bool hit=1;
 bool chi2=1;
 bool ctf=1;
 bool rs=1;

 //////////////////////////////////////
 /////////// CTF //////////////////////
 //////////////////////////////////////
 if (ctf){
   //efficiency&fakerate
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/effic",rh1);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/effic",sh1);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/effic",rc1);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/effic",sc1);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/fakerate",rh2);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/fakerate",sh2);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/fakerate",rc2);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/fakerate",sc2);

   canvas = new TCanvas("Tracks1","Tracks: efficiency & fakerate",1000,1000);
   TPaveLabel* title = new TPaveLabel(0.1,0.96,0.9,0.99,relinfo);
   title->Draw();
   TPad* graphPad = new TPad("Graphs","Graphs",0.01,0.05,0.95,0.95);
   graphPad->Draw();
   graphPad->cd();

   if (hit) rh1->GetYaxis()->SetRangeUser(0.7,1.025);
   if (hit) sh1->GetYaxis()->SetRangeUser(0.7,1.025);
   if (chi2)rc1->GetYaxis()->SetRangeUser(0.7,1.025);
   if (chi2)sc1->GetYaxis()->SetRangeUser(0.7,1.025);

   if (hit&&chi2){
     char * option = "UU";
     double  startingY = -1;
     double  startingX = .1;
     bool fit = false;

     graphPad->Divide(2,2);
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(1);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rc1->SetLineColor(2);
     sc1->SetLineColor(4);
     sc1->SetLineStyle(2);
     setStats(rc1,sc1, startingY, startingX, fit);
     rc1->Draw();
     sc1->Draw("sames");
     myPV->PVCompute(rc1, sc1, te, option );
     
     graphPad->cd(3);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(4);
     rc2->SetLineColor(2);
     sc2->SetLineColor(4);
     sc2->SetLineStyle(2);
     setStats(rc2,sc2, startingY, startingX, fit);
     rc2->Draw();
     sc2->Draw("sames");
     myPV->PVCompute(rc2, sc2, te, option );
     
     //     canvas->cd(0);
     //    thistext.DrawLatex(0.1, 0.01, mytext);
     
   }else if (hit){  
     char * option = "UU";
     double  startingY = -1;
     double  startingX = .1;
     bool fit = false;

     graphPad->Divide(1,2);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );  
  
   }else if (chi2){ 
     char * option = "UU";
     double  startingY = -1;
     double  startingX = .1;
     bool fit = false;

     graphPad->Divide(1,2);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );  
     
   }

   canvas->Print("ctf_effic_fake.eps");
   canvas->Print("ctf_effic_fake.gif");

   //chi2&chi2 probability
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/chi2",rh1);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/chi2",sh1);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/chi2",rc1);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/chi2",sc1);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/chi2_prob",rh2);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/chi2_prob",sh2);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/chi2_prob",rc2);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/chi2_prob",sc2);

   canvas = new TCanvas("Tracks2","Tracks: chi2 & chi2 probability",1000,1000);
   TPaveLabel* title = new TPaveLabel(0.1,0.96,0.9,0.99,relinfo);
   title->Draw();
   TPad* graphPad = new TPad("Graphs","Graphs",0.01,0.05,0.95,0.95);
   graphPad->Draw();
   graphPad->cd();

   if (hit){
     NormalizeHistograms(rh1,sh1);
     NormalizeHistograms(rh2,sh2);
     fixRangeY(rh1,sh1);
     fixRangeY(rh2,sh2);
   }
   if (chi2){
     NormalizeHistograms(rc1,sc1);
     NormalizeHistograms(rc2,sc2);
     fixRangeY(rc1,sc1);
     fixRangeY(rc2,sc2);
   }
   
   if (hit&&chi2){
     char * option = "UUNORM";
     double  startingY = 0.4;
     double  startingX = 0.7;
     bool fit = false;

     graphPad->Divide(2,2);

     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(1);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rc1->SetLineColor(2);
     sc1->SetLineColor(4);
     sc1->SetLineStyle(2);
     setStats(rc1,sc1, startingY, startingX, fit);
     rc1->Draw();
     sc1->Draw("sames");
     myPV->PVCompute(rc1, sc1, te, option );
     
     graphPad->cd(3);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(4);
     rc2->SetLineColor(2);
     sc2->SetLineColor(4);
     sc2->SetLineStyle(2);
     setStats(rc2,sc2, startingY, startingX, fit);
     rc2->Draw();
     sc2->Draw("sames");
     myPV->PVCompute(rc2, sc2, te, option );
     
   }else if (hit){  
     char * option = "UUNORM";
     double  startingY = 0.4;
     double  startingX = 0.7;
     bool fit = false;

     graphPad->Divide(1,2);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );  
  
   }else if (chi2){ 
     char * option = "UUNORM";
     double  startingY = 0.4;
     double  startingX = 0.7;
     bool fit = false;

     graphPad->Divide(1,2);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );  
     
   }

   canvas->Print("ctf_chi2_chi2prob.eps");
   canvas->Print("ctf_chi2_chi2prob.gif");

   //meanchi2 and #hits vs eta
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/hits_eta",rh1);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/hits_eta",sh1);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/hits_eta",rc1);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/hits_eta",sc1);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/chi2mean",rh2);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/chi2mean",sh2);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/chi2mean",rc2);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/chi2mean",sc2);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/losthits_eta",rh3);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/losthits_eta",sh3);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/losthits_eta",rc3);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/losthits_eta",sc3);

   canvas = new TCanvas("Tracks3","Tracks: chi2 and #hits vs eta",1000,1000);
   TPaveLabel* title = new TPaveLabel(0.1,0.96,0.9,0.99,relinfo);
   title->Draw();
   TPad* graphPad = new TPad("Graphs","Graphs",0.01,0.05,0.95,0.95);
   graphPad->Draw();
   graphPad->cd();

   //fixRangeY(rh1,sh1);
   //fixRangeY(rc1,sc1);
   if (hit) fixRangeY(rh2,sh2);
   if (chi2)fixRangeY(rc2,sc2);


   if (hit&&chi2){
     char * option = "UU";
     double  startingY = -1;
     double  startingX = .1;
     bool fit = false;

     graphPad->Divide(2,3);

     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(1);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rc1->SetLineColor(2);
     sc1->SetLineColor(4);
     sc1->SetLineStyle(2);
     setStats(rc1,sc1, startingY, startingX, fit);
     rc1->Draw();
     sc1->Draw("sames");
     myPV->PVCompute(rc1, sc1, te, option );
     
     graphPad->cd(3);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(4);
     rc2->SetLineColor(2);
     sc2->SetLineColor(4);
     sc2->SetLineStyle(2);
     setStats(rc2,sc2, startingY, startingX, fit);
     rc2->Draw();
     sc2->Draw("sames");
     myPV->PVCompute(rc2, sc2, te, option );

     graphPad->cd(5);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
  
     graphPad->cd(6);
     rc3->SetLineColor(2);
     sc3->SetLineColor(4);
     sc3->SetLineStyle(2);
     setStats(rc3,sc3, startingY, startingX, fit);
     rc3->Draw();
     sc3->Draw("sames");
     myPV->PVCompute(rc3, sc3, te, option );
     
   }else if (hit){  
     char * option = "UU";
     double  startingY = -1;
     double  startingX = .1;
     bool fit = false;

     graphPad->Divide(1,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );  

     graphPad->cd(3);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
     
   }else if (chi2){ 
     char * option = "UU";
     double  startingY = -1;
     double  startingX = .1;
     bool fit = false;

     graphPad->Divide(1,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );  
     
     graphPad->cd(3);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
     
   }
   
   canvas->Print("ctf_hitseta_chi2mean.eps");
   canvas->Print("ctf_hitseta_chi2mean.gif");

   //pull Pt, Qoverp, Phi
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/pullPt",rh1);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/pullPt",sh1);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/pullPt",rc1);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/pullPt",sc1);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/pullQoverp",rh2);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/pullQoverp",sh2);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/pullQoverp",rc2);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/pullQoverp",sc2);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/pullPhi0",rh3);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/pullPhi0",sh3);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/pullPhi0",rc3);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/pullPhi0",sc3);

   canvas = new TCanvas("Tracks4","Tracks: pull of Pt, Qoverp and Phi",1000,1000);
   TPaveLabel* title = new TPaveLabel(0.1,0.96,0.9,0.99,relinfo);
   title->Draw();
   TPad* graphPad = new TPad("Graphs","Graphs",0.01,0.05,0.95,0.95);
   graphPad->Draw();
   graphPad->cd();

   if (hit){
     NormalizeHistograms(rh1,sh1);
     NormalizeHistograms(rh2,sh2);
     NormalizeHistograms(rh3,sh3);
   }
   if (chi2){
     NormalizeHistograms(rc1,sc1);
     NormalizeHistograms(rc2,sc2);
     NormalizeHistograms(rc3,sc3);
   }

   if (hit&&chi2){   
     char * option = "UUNORM";
     double  startingY = 0.4;
     double  startingX = 0.1;
     bool fit = true;

     graphPad->Divide(2,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rc1->SetLineColor(2);
     sc1->SetLineColor(4);
     sc1->SetLineStyle(2);
     setStats(rc1,sc1, startingY, startingX, fit);
     rc1->Draw();
     sc1->Draw("sames");
     myPV->PVCompute(rc1, sc1, te, option );
     
     graphPad->cd(3);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(4);
     rc2->SetLineColor(2);
     sc2->SetLineColor(4);
     sc2->SetLineStyle(2);
     setStats(rc2,sc2, startingY, startingX, fit);
     rc2->Draw();
     sc2->Draw("sames");
     myPV->PVCompute(rc2, sc2, te, option );
     
     graphPad->cd(5);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
     
     graphPad->cd(6);
     rc3->SetLineColor(2);
     sc3->SetLineColor(4);
     sc3->SetLineStyle(2);
     setStats(rc3,sc3, startingY, startingX, fit);
     rc3->Draw();
     sc3->Draw("sames");
     myPV->PVCompute(rc3, sc3, te, option );
   }else if (hit){ 
     char * option = "UUNORM";
     double  startingY = 0.4;
     double  startingX = 0.1;
     bool fit = true;

     graphPad->Divide(1,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(3);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
   }else if (chi2){
     char * option = "UUNORM";
     double  startingY = 0.4;
     double  startingX = 0.1;
     bool fit = true;
     
     graphPad->Divide(1,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(3);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
   }
   
   canvas->Print("ctf_pullPt_Qoverp_Phi.eps");
   canvas->Print("ctf_pullPt_Qoverp_Phi.gif");

   //pull D0, Z0, Theta
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/pullD0",rh1);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/pullD0",sh1);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/pullD0",rc1);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/pullD0",sc1);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/pullDz",rh2);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/pullDz",sh2);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/pullDz",rc2);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/pullDz",sc2);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/pullTheta",rh3);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/pullTheta",sh3);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/pullTheta",rc3);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/pullTheta",sc3);

   canvas = new TCanvas("Tracks5","Tracks: pull of D0, Z0, Theta",1000,1000);
   TPaveLabel* title = new TPaveLabel(0.1,0.96,0.9,0.99,relinfo);
   title->Draw();
   TPad* graphPad = new TPad("Graphs","Graphs",0.01,0.05,0.95,0.95);
   graphPad->Draw();
   graphPad->cd();

   if (hit){
     NormalizeHistograms(rh1,sh1);
     NormalizeHistograms(rh2,sh2);
     NormalizeHistograms(rh3,sh3);
   }
   if (chi2){
     NormalizeHistograms(rc1,sc1);
     NormalizeHistograms(rc2,sc2);
     NormalizeHistograms(rc3,sc3);
   }

   if (hit&&chi2){   
     char * option = "UUNORM";
     double  startingY = 0.4;
     double  startingX = 0.1;
     bool fit = true;

     graphPad->Divide(2,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rc1->SetLineColor(2);
     sc1->SetLineColor(4);
     sc1->SetLineStyle(2);
     setStats(rc1,sc1, startingY, startingX, fit);
     rc1->Draw();
     sc1->Draw("sames");
     myPV->PVCompute(rc1, sc1, te, option );
     
     graphPad->cd(3);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(4);
     rc2->SetLineColor(2);
     sc2->SetLineColor(4);
     sc2->SetLineStyle(2);
     setStats(rc2,sc2, startingY, startingX, fit);
     rc2->Draw();
     sc2->Draw("sames");
     myPV->PVCompute(rc2, sc2, te, option );
     
     graphPad->cd(5);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
     
     graphPad->cd(6);
     rc3->SetLineColor(2);
     sc3->SetLineColor(4);
     sc3->SetLineStyle(2);
     setStats(rc3,sc3, startingY, startingX, fit);
     rc3->Draw();
     sc3->Draw("sames");
     myPV->PVCompute(rc3, sc3, te, option );
   }else if (hit){ 
     char * option = "UUNORM";
     double  startingY = 0.4;
     double  startingX = 0.1;
     bool fit = true;

     graphPad->Divide(1,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(3);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
   }else if (chi2){
     char * option = "UUNORM";
     double  startingY = 0.4;
     double  startingX = 0.1;
     bool fit = true;
     
     graphPad->Divide(1,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(3);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
   }
   
   canvas->Print("ctf_pullD0_Z0_Theta.eps");
   canvas->Print("ctf_pullD0_Z0_Theta.gif");

   //resolution Pt, Phi
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/sigmapt",rh1);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/sigmapt",sh1);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/sigmapt",rc1);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/sigmapt",sc1);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/sigmaphi",rh2);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/sigmaphi",sh2);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/sigmaphi",rc2);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/sigmaphi",sc2);

   canvas = new TCanvas("Tracks6","Tracks: Pt and Phi resolution",1000,1000);
   TPaveLabel* title = new TPaveLabel(0.1,0.96,0.9,0.99,relinfo);
   title->Draw();
   TPad* graphPad = new TPad("Graphs","Graphs",0.01,0.05,0.95,0.95);
   graphPad->Draw();
   graphPad->cd();

   if (hit&&chi2){
     char * option = "UU";
     double  startingY = -1;
     double  startingX = .1;
     bool fit = false;

     graphPad->Divide(2,2);

     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(1);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rc1->SetLineColor(2);
     sc1->SetLineColor(4);
     sc1->SetLineStyle(2);
     setStats(rc1,sc1, startingY, startingX, fit);
     rc1->Draw();
     sc1->Draw("sames");
     myPV->PVCompute(rc1, sc1, te, option );
     
     graphPad->cd(3);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(4);
     rc2->SetLineColor(2);
     sc2->SetLineColor(4);
     sc2->SetLineStyle(2);
     setStats(rc2,sc2, startingY, startingX, fit);
     rc2->Draw();
     sc2->Draw("sames");
     myPV->PVCompute(rc2, sc2, te, option );
     
   }else if (hit){  
     char * option = "UU";
     double  startingY = -1;
     double  startingX = .1;
     bool fit = false;

     graphPad->Divide(1,2);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );  
  
   }else if (chi2){ 
     char * option = "UU";
     double  startingY = -1;
     double  startingX = .1;
     bool fit = false;

     graphPad->Divide(1,2);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );  
     
   }

   canvas->Print("ctf_resolPt_Phi.eps");
   canvas->Print("ctf_resolPt_Phi.gif");

   //resolution D0, Z0, Theta
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/sigmad0",rh1);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/sigmad0",sh1);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/sigmad0",rc1);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/sigmad0",sc1);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/sigmaz0",rh2);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/sigmaz0",sh2);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/sigmaz0",rc2);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/sigmaz0",sc2);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/sigmacotTheta",rh3);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByHits/sigmacotTheta",sh3);
   rfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/sigmacotTheta",rc3);
   sfile->GetObject("DQMData/Track/cutsCKF_AssociatorByChi2/sigmacotTheta",sc3);

   canvas = new TCanvas("Tracks7","Tracks: D0, Z0, Theta resolution",1000,1000);
   TPaveLabel* title = new TPaveLabel(0.1,0.96,0.9,0.99,relinfo);
   title->Draw();
   TPad* graphPad = new TPad("Graphs","Graphs",0.01,0.05,0.95,0.95);
   graphPad->Draw();
   graphPad->cd();

   if (hit&&chi2){   
     char * option = "UU";
     double  startingY = -1;
     double  startingX = 0.1;
     bool fit = true;

     graphPad->Divide(2,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rc1->SetLineColor(2);
     sc1->SetLineColor(4);
     sc1->SetLineStyle(2);
     setStats(rc1,sc1, startingY, startingX, fit);
     rc1->Draw();
     sc1->Draw("sames");
     myPV->PVCompute(rc1, sc1, te, option );
     
     graphPad->cd(3);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(4);
     rc2->SetLineColor(2);
     sc2->SetLineColor(4);
     sc2->SetLineStyle(2);
     setStats(rc2,sc2, startingY, startingX, fit);
     rc2->Draw();
     sc2->Draw("sames");
     myPV->PVCompute(rc2, sc2, te, option );
     
     graphPad->cd(5);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
     
     graphPad->cd(6);
     rc3->SetLineColor(2);
     sc3->SetLineColor(4);
     sc3->SetLineStyle(2);
     setStats(rc3,sc3, startingY, startingX, fit);
     rc3->Draw();
     sc3->Draw("sames");
     myPV->PVCompute(rc3, sc3, te, option );
   }else if (hit){ 
     char * option = "UU";
     double  startingY = -1;
     double  startingX = 0.1;
     bool fit = true;

     graphPad->Divide(1,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(3);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
   }else if (chi2){
     char * option = "UU";
     double  startingY = -1;
     double  startingX = 0.1;
     bool fit = true;
     
     graphPad->Divide(1,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(3);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
   }
   
   canvas->Print("ctf_resolD0_Z0_Theta.eps");
   canvas->Print("ctf_resolD0_Z0_Theta.gif");
 }



 //////////////////////////////////////
 /////////// RS //////////////////////
 //////////////////////////////////////
 if (rs){
   //efficiency&fakerate
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/effic",rh1);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/effic",sh1);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/effic",rc1);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/effic",sc1);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/fakerate",rh2);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/fakerate",sh2);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/fakerate",rc2);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/fakerate",sc2);

   canvas = new TCanvas("Tracks8","Tracks: efficiency & fakerate",1000,1000);
   TPaveLabel* title = new TPaveLabel(0.1,0.96,0.9,0.99,relinfo);
   title->Draw();
   TPad* graphPad = new TPad("Graphs","Graphs",0.01,0.05,0.95,0.95);
   graphPad->Draw();
   graphPad->cd();

   if (hit) rh1->GetYaxis()->SetRangeUser(0.7,1.025);
   if (hit) sh1->GetYaxis()->SetRangeUser(0.7,1.025);
   if (chi2)rc1->GetYaxis()->SetRangeUser(0.7,1.025);
   if (chi2)sc1->GetYaxis()->SetRangeUser(0.7,1.025);

   if (hit&&chi2){
     char * option = "UU";
     double  startingY = -1;
     double  startingX = .1;
     bool fit = false;

     graphPad->Divide(2,2);

     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(1);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rc1->SetLineColor(2);
     sc1->SetLineColor(4);
     sc1->SetLineStyle(2);
     setStats(rc1,sc1, startingY, startingX, fit);
     rc1->Draw();
     sc1->Draw("sames");
     myPV->PVCompute(rc1, sc1, te, option );
     
     graphPad->cd(3);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(4);
     rc2->SetLineColor(2);
     sc2->SetLineColor(4);
     sc2->SetLineStyle(2);
     setStats(rc2,sc2, startingY, startingX, fit);
     rc2->Draw();
     sc2->Draw("sames");
     myPV->PVCompute(rc2, sc2, te, option );
     
   }else if (hit){  
     char * option = "UU";
     double  startingY = -1;
     double  startingX = .1;
     bool fit = false;

     graphPad->Divide(1,2);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );  
  
   }else if (chi2){ 
     char * option = "UU";
     double  startingY = -1;
     double  startingX = .1;
     bool fit = false;

     graphPad->Divide(1,2);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );  
     
   }

   canvas->Print("rs_effic_fake.eps");
   canvas->Print("rs_effic_fake.gif");

   //chi2&chi2 probability
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/chi2",rh1);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/chi2",sh1);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/chi2",rc1);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/chi2",sc1);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/chi2_prob",rh2);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/chi2_prob",sh2);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/chi2_prob",rc2);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/chi2_prob",sc2);

   canvas = new TCanvas("Tracks9","Tracks: chi2 & chi2 probability",1000,1000);
   TPaveLabel* title = new TPaveLabel(0.1,0.96,0.9,0.99,relinfo);
   title->Draw();
   TPad* graphPad = new TPad("Graphs","Graphs",0.01,0.05,0.95,0.95);
   graphPad->Draw();
   graphPad->cd();

   if (hit) { 
     NormalizeHistograms(rh1,sh1);
     NormalizeHistograms(rh2,sh2);
     fixRangeY(rh1,sh1);
     fixRangeY(rh2,sh2);
   }
   if (chi2) {
     NormalizeHistograms(rc1,sc1);
     NormalizeHistograms(rc2,sc2);
     fixRangeY(rc1,sc1);
     fixRangeY(rc2,sc2);
   }

   fixRangeY(rh1,sh1);
   fixRangeY(rc1,sc1);
   fixRangeY(rh2,sh2);
   fixRangeY(rc2,sc2);

   if (hit&&chi2){
     char * option = "UUNORM";
     double  startingY = 0.4;
     double  startingX = 0.7;
     bool fit = false;

     graphPad->Divide(2,2);

     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(1);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rc1->SetLineColor(2);
     sc1->SetLineColor(4);
     sc1->SetLineStyle(2);
     setStats(rc1,sc1, startingY, startingX, fit);
     rc1->Draw();
     sc1->Draw("sames");
     myPV->PVCompute(rc1, sc1, te, option );
     
     graphPad->cd(3);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(4);
     rc2->SetLineColor(2);
     sc2->SetLineColor(4);
     sc2->SetLineStyle(2);
     setStats(rc2,sc2, startingY, startingX, fit);
     rc2->Draw();
     sc2->Draw("sames");
     myPV->PVCompute(rc2, sc2, te, option );
     
   }else if (hit){  
     char * option = "UUNORM";
     double  startingY = 0.4;
     double  startingX = 0.7;
     bool fit = false;

     graphPad->Divide(1,2);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );  
  
   }else if (chi2){ 
     char * option = "UUNORM";
     double  startingY = 0.4;
     double  startingX = 0.7;
     bool fit = false;

     graphPad->Divide(1,2);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );  
     
   }

   canvas->Print("rs_chi2_chi2prob.eps");
   canvas->Print("rs_chi2_chi2prob.gif");

   //meanchi2 and #hits vs eta
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/hits_eta",rh1);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/hits_eta",sh1);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/hits_eta",rc1);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/hits_eta",sc1);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/chi2mean",rh2);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/chi2mean",sh2);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/chi2mean",rc2);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/chi2mean",sc2);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/losthits_eta",rh3);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/losthits_eta",sh3);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/losthits_eta",rc3);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/losthits_eta",sc3);

   canvas = new TCanvas("Tracks10","Tracks: chi2 and #hits vs eta",1000,1000);
   TPaveLabel* title = new TPaveLabel(0.1,0.96,0.9,0.99,relinfo);
   title->Draw();
   TPad* graphPad = new TPad("Graphs","Graphs",0.01,0.05,0.95,0.95);
   graphPad->Draw();
   graphPad->cd();

   if (hit) fixRangeY(rh2,sh2);
   if (chi2) fixRangeY(rc2,sc2);

   if (hit&&chi2){
     char * option = "UU";
     double  startingY = -1;
     double  startingX = .1;
     bool fit = false;
     
     graphPad->Divide(2,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(1);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rc1->SetLineColor(2);
     sc1->SetLineColor(4);
     sc1->SetLineStyle(2);
     setStats(rc1,sc1, startingY, startingX, fit);
     rc1->Draw();
     sc1->Draw("sames");
     myPV->PVCompute(rc1, sc1, te, option );
     
     graphPad->cd(3);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(4);
     rc2->SetLineColor(2);
     sc2->SetLineColor(4);
     sc2->SetLineStyle(2);
     setStats(rc2,sc2, startingY, startingX, fit);
     rc2->Draw();
     sc2->Draw("sames");
     myPV->PVCompute(rc2, sc2, te, option );
     
     graphPad->cd(5);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
  
     graphPad->cd(6);
     rc3->SetLineColor(2);
     sc3->SetLineColor(4);
     sc3->SetLineStyle(2);
     setStats(rc3,sc3, startingY, startingX, fit);
     rc3->Draw();
     sc3->Draw("sames");
     myPV->PVCompute(rc3, sc3, te, option );
     
   }else if (hit){  
     char * option = "UU";
     double  startingY = -1;
     double  startingX = .1;
     bool fit = false;
     
     graphPad->Divide(1,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );  

     graphPad->cd(3);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );     
     
   }else if (chi2){ 
     char * option = "UU";
     double  startingY = -1;
     double  startingX = .1;
     bool fit = false;
     
     graphPad->Divide(1,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );  

     graphPad->cd(3);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
          
   }

   canvas->Print("rs_hitseta_chi2mean.eps");
   canvas->Print("rs_hitseta_chi2mean.gif");
   
   //pull Pt, Qoverp, Phi
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/pullPt",rh1);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/pullPt",sh1);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/pullPt",rc1);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/pullPt",sc1);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/pullQoverp",rh2);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/pullQoverp",sh2);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/pullQoverp",rc2);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/pullQoverp",sc2);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/pullPhi0",rh3);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/pullPhi0",sh3);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/pullPhi0",rc3);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/pullPhi0",sc3);

   canvas = new TCanvas("Tracks11","Tracks: pull of Pt, Qoverp and Phi",1000,1000);
   TPaveLabel* title = new TPaveLabel(0.1,0.96,0.9,0.99,relinfo);
   title->Draw();
   TPad* graphPad = new TPad("Graphs","Graphs",0.01,0.05,0.95,0.95);
   graphPad->Draw();
   graphPad->cd();

   if (hit) { 
     NormalizeHistograms(rh1,sh1);
     NormalizeHistograms(rh2,sh2);
     NormalizeHistograms(rh3,sh3);
   }
   if (chi2) { 
     NormalizeHistograms(rc1,sc1);
     NormalizeHistograms(rc2,sc2);
     NormalizeHistograms(rc3,sc3);
   }

   if (hit&&chi2){   
     char * option = "UUNORM";
     double  startingY = 0.4;
     double  startingX = 0.1;
     bool fit = true;

     graphPad->Divide(2,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rc1->SetLineColor(2);
     sc1->SetLineColor(4);
     sc1->SetLineStyle(2);
     setStats(rc1,sc1, startingY, startingX, fit);
     rc1->Draw();
     sc1->Draw("sames");
     myPV->PVCompute(rc1, sc1, te, option );
     
     graphPad->cd(3);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(4);
     rc2->SetLineColor(2);
     sc2->SetLineColor(4);
     sc2->SetLineStyle(2);
     setStats(rc2,sc2, startingY, startingX, fit);
     rc2->Draw();
     sc2->Draw("sames");
     myPV->PVCompute(rc2, sc2, te, option );
     
     graphPad->cd(5);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
     
     graphPad->cd(6);
     rc3->SetLineColor(2);
     sc3->SetLineColor(4);
     sc3->SetLineStyle(2);
     setStats(rc3,sc3, startingY, startingX, fit);
     rc3->Draw();
     sc3->Draw("sames");
     myPV->PVCompute(rc3, sc3, te, option );
   }else if (hit){ 
     char * option = "UUNORM";
     double  startingY = 0.4;
     double  startingX = 0.1;
     bool fit = true;

     graphPad->Divide(1,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(3);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
   }else if (chi2){
     char * option = "UUNORM";
     double  startingY = 0.4;
     double  startingX = 0.1;
     bool fit = true;
     
     graphPad->Divide(1,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(3);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
   }
   
   canvas->Print("rs_pullPt_Qoverp_Phi.eps");
   canvas->Print("rs_pullPt_Qoverp_Phi.gif");

   //pull D0, Z0, Theta
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/pullD0",rh1);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/pullD0",sh1);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/pullD0",rc1);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/pullD0",sc1);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/pullDz",rh2);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/pullDz",sh2);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/pullDz",rc2);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/pullDz",sc2);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/pullTheta",rh3);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/pullTheta",sh3);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/pullTheta",rc3);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/pullTheta",sc3);

   canvas = new TCanvas("Tracks12","Tracks: pull of D0, Z0, Theta",1000,1000);
   TPaveLabel* title = new TPaveLabel(0.1,0.96,0.9,0.99,relinfo);
   title->Draw();
   TPad* graphPad = new TPad("Graphs","Graphs",0.01,0.05,0.95,0.95);
   graphPad->Draw();
   graphPad->cd();

   if (hit) { 
     NormalizeHistograms(rh1,sh1);
     NormalizeHistograms(rh2,sh2);
     NormalizeHistograms(rh3,sh3);
   }
   if (chi2) { 
     NormalizeHistograms(rc1,sc1);
     NormalizeHistograms(rc2,sc2);
     NormalizeHistograms(rc3,sc3);
   }

   if (hit&&chi2){   
     char * option = "UUNORM";
     double  startingY = 0.4;
     double  startingX = 0.1;
     bool fit = true;

     graphPad->Divide(2,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rc1->SetLineColor(2);
     sc1->SetLineColor(4);
     sc1->SetLineStyle(2);
     setStats(rc1,sc1, startingY, startingX, fit);
     rc1->Draw();
     sc1->Draw("sames");
     myPV->PVCompute(rc1, sc1, te, option );
     
     graphPad->cd(3);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(4);
     rc2->SetLineColor(2);
     sc2->SetLineColor(4);
     sc2->SetLineStyle(2);
     setStats(rc2,sc2, startingY, startingX, fit);
     rc2->Draw();
     sc2->Draw("sames");
     myPV->PVCompute(rc2, sc2, te, option );
     
     graphPad->cd(5);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
     
     graphPad->cd(6);
     rc3->SetLineColor(2);
     sc3->SetLineColor(4);
     sc3->SetLineStyle(2);
     setStats(rc3,sc3, startingY, startingX, fit);
     rc3->Draw();
     sc3->Draw("sames");
     myPV->PVCompute(rc3, sc3, te, option );
   }else if (hit){ 
     char * option = "UUNORM";
     double  startingY = 0.4;
     double  startingX = 0.1;
     bool fit = true;

     graphPad->Divide(1,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(3);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
   }else if (chi2){
     char * option = "UUNORM";
     double  startingY = 0.4;
     double  startingX = 0.1;
     bool fit = true;
     
     graphPad->Divide(1,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(3);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
   }
   
   canvas->Print("rs_pullD0_Z0_Theta.eps");
   canvas->Print("rs_pullD0_Z0_Theta.gif");

   //resolution Pt, Phi
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/sigmapt",rh1);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/sigmapt",sh1);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/sigmapt",rc1);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/sigmapt",sc1);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/sigmaphi",rh2);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/sigmaphi",sh2);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/sigmaphi",rc2);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/sigmaphi",sc2);

   canvas = new TCanvas("Tracks13","Tracks: Pt and Phi resolution",1000,1000);
   TPaveLabel* title = new TPaveLabel(0.1,0.96,0.9,0.99,relinfo);
   title->Draw();
   TPad* graphPad = new TPad("Graphs","Graphs",0.01,0.05,0.95,0.95);
   graphPad->Draw();
   graphPad->cd();

   if (hit&&chi2){
     char * option = "UU";
     double  startingY = -1;
     double  startingX = .1;
     bool fit = false;

     graphPad->Divide(2,2);

     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(1);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rc1->SetLineColor(2);
     sc1->SetLineColor(4);
     sc1->SetLineStyle(2);
     setStats(rc1,sc1, startingY, startingX, fit);
     rc1->Draw();
     sc1->Draw("sames");
     myPV->PVCompute(rc1, sc1, te, option );
     
     graphPad->cd(3);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(4);
     rc2->SetLineColor(2);
     sc2->SetLineColor(4);
     sc2->SetLineStyle(2);
     setStats(rc2,sc2, startingY, startingX, fit);
     rc2->Draw();
     sc2->Draw("sames");
     myPV->PVCompute(rc2, sc2, te, option );
     
   }else if (hit){  
     char * option = "UU";
     double  startingY = -1;
     double  startingX = .1;
     bool fit = false;

     graphPad->Divide(1,2);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );  
  
   }else if (chi2){ 
     char * option = "UU";
     double  startingY = -1;
     double  startingX = .1;
     bool fit = false;

     graphPad->Divide(1,2);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );  
     
   }

   canvas->Print("rs_resolPt_Phi.eps");
   canvas->Print("rs_resolPt_Phi.gif");

   //resolution D0, Z0, Theta
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/sigmad0",rh1);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/sigmad0",sh1);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/sigmad0",rc1);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/sigmad0",sc1);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/sigmaz0",rh2);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/sigmaz0",sh2);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/sigmaz0",rc2);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/sigmaz0",sc2);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/sigmacotTheta",rh3);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByHits/sigmacotTheta",sh3);
   rfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/sigmacotTheta",rc3);
   sfile->GetObject("DQMData/Track/cutsRS_AssociatorByChi2/sigmacotTheta",sc3);

   canvas = new TCanvas("Tracks14","Tracks: D0, Z0, Theta resolution",1000,1000);
   TPaveLabel* title = new TPaveLabel(0.1,0.96,0.9,0.99,relinfo);
   title->Draw();
   TPad* graphPad = new TPad("Graphs","Graphs",0.01,0.05,0.95,0.95);
   graphPad->Draw();
   graphPad->cd();

   if (hit&&chi2){   
     char * option = "UU";
     double  startingY = -1;
     double  startingX = 0.1;
     bool fit = true;

     graphPad->Divide(2,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rc1->SetLineColor(2);
     sc1->SetLineColor(4);
     sc1->SetLineStyle(2);
     setStats(rc1,sc1, startingY, startingX, fit);
     rc1->Draw();
     sc1->Draw("sames");
     myPV->PVCompute(rc1, sc1, te, option );
     
     graphPad->cd(3);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(4);
     rc2->SetLineColor(2);
     sc2->SetLineColor(4);
     sc2->SetLineStyle(2);
     setStats(rc2,sc2, startingY, startingX, fit);
     rc2->Draw();
     sc2->Draw("sames");
     myPV->PVCompute(rc2, sc2, te, option );
     
     graphPad->cd(5);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
     
     graphPad->cd(6);
     rc3->SetLineColor(2);
     sc3->SetLineColor(4);
     sc3->SetLineStyle(2);
     setStats(rc3,sc3, startingY, startingX, fit);
     rc3->Draw();
     sc3->Draw("sames");
     myPV->PVCompute(rc3, sc3, te, option );
   }else if (hit){ 
     char * option = "UU";
     double  startingY = -1;
     double  startingX = 0.1;
     bool fit = true;

     graphPad->Divide(1,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(3);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
   }else if (chi2){
     char * option = "UU";
     double  startingY = -1;
     double  startingX = 0.1;
     bool fit = true;
     
     graphPad->Divide(1,3);
     
     graphPad->cd(1);
     rh1->SetLineColor(2);
     sh1->SetLineColor(4);
     sh1->SetLineStyle(2);
     setStats(rh1,sh1, startingY, startingX, fit);
     rh1->Draw();
     sh1->Draw("sames");
     myPV->PVCompute(rh1, sh1, te, option );
     
     graphPad->cd(2);
     rh2->SetLineColor(2);
     sh2->SetLineColor(4);
     sh2->SetLineStyle(2);
     setStats(rh2,sh2, startingY, startingX, fit);
     rh2->Draw();
     sh2->Draw("sames");
     myPV->PVCompute(rh2, sh2, te, option );
     
     graphPad->cd(3);
     rh3->SetLineColor(2);
     sh3->SetLineColor(4);
     sh3->SetLineStyle(2);
     setStats(rh3,sh3, startingY, startingX, fit);
     rh3->Draw();
     sh3->Draw("sames");
     myPV->PVCompute(rh3, sh3, te, option );
   }
   
   canvas->Print("rs_resolD0_Z0_Theta.eps");
   canvas->Print("rs_resolD0_Z0_Theta.gif");
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
    
      h1->Sumw2();
      h2->Sumw2();
      h1->Scale(scale1);
      h2->Scale(scale2);
    }
}

void setStats(TH1* r,TH1* s, double startingY, double startingX = .1,bool fit){
  if (startingY<0){
    r->SetStats(0);
    s->SetStats(0);
  } else {
    if (fit){
      r->Fit("gaus");
      TF1* f1 = (TF1*) r->GetListOfFunctions()->FindObject("gaus");
      f1->SetLineColor(2);
      f1->SetLineWidth(1);
    }
    r->Draw();
    gPad->Update(); 
    TPaveStats* st1 = (TPaveStats*) r->GetListOfFunctions()->FindObject("stats");
    if (fit) st1->SetOptFit();
    st1->SetX1NDC(startingX);
    st1->SetX2NDC(startingX+0.2);
    st1->SetY1NDC(startingY+0.15);
    st1->SetY2NDC(startingY+0.3);
    st1->SetTextColor(2);
    if (fit) {
      s->Fit("gaus");
      TF1* f2 = (TF1*) s->GetListOfFunctions()->FindObject("gaus");
      f2->SetLineColor(4);
      f2->SetLineWidth(1);    
    }
    s->Draw();
    gPad->Update(); 
    TPaveStats* st2 = (TPaveStats*) s->GetListOfFunctions()->FindObject("stats");
    if (fit) st2->SetOptFit();
    st2->SetX1NDC(startingX);
    st2->SetX2NDC(startingX+0.2);
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

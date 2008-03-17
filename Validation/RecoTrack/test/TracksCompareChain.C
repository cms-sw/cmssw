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
 TDirectory * rdir=gDirectory; 
 TFile * sfile = new TFile(sfilename);
 TDirectory * sdir=gDirectory; 

 if(rfile->cd("DQMData/RecoTrackV"))rfile->cd("DQMData/RecoTrackV/Track");
 else rfile->cd("DQMData/Track");
 rdir=gDirectory;

 if(sfile->cd("DQMData/RecoTrackV"))sfile->cd("DQMData/RecoTrackV/Track");
 else sfile->cd("DQMData/Track");
 sdir=gDirectory; 

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
   string rdirName,sdirName;
   if(rdir->cd("cutsReco_AssociatorByHits"))rdirName="cutsReco";
   else rdirName="cutsCKF";
   if(sdir->cd("cutsReco_AssociatorByHits"))sdirName="cutsReco";
   else sdirName="cutsCKF";
   //efficiency&fakerate
   cout<<rdir->GetPath()<<endl;
   cout<<sdir->GetPath()<<endl;
   cout<<sdir->ls()<<endl;
   rdir->GetObject((rdirName+"_AssociatorByHits/effic").c_str(),rh1);
   cout<<(sdirName+"_AssociatorByHits/effic").c_str()<<endl;
   sdir->GetObject((sdirName+"_AssociatorByHits/effic").c_str(),sh1);
   rdir->GetObject((rdirName+"_AssociatorByChi2/effic").c_str(),rc1);
   sdir->GetObject((sdirName+"_AssociatorByChi2/effic").c_str(),sc1);
   rdir->GetObject((rdirName+"_AssociatorByHits/fakerate").c_str(),rh2);
   sdir->GetObject((sdirName+"_AssociatorByHits/fakerate").c_str(),sh2);
   rdir->GetObject((rdirName+"_AssociatorByChi2/fakerate").c_str(),rc2);
   sdir->GetObject((sdirName+"_AssociatorByChi2/fakerate").c_str(),sc2);

   canvas = new TCanvas("Tracks1","Tracks: efficiency & fakerate",1000,1000);
   TPaveLabel* title = new TPaveLabel(0.1,0.96,0.9,0.99,relinfo);
   title->Draw();
   TPad* graphPad = new TPad("Graphs","Graphs",0.01,0.05,0.95,0.95);
   graphPad->Draw();
   graphPad->cd();

   if (hit) rh1->GetYaxis()->SetRangeUser(0.7,1.025);
   //   if (hit) sh1->GetYaxis()->SetRangeUser(0.7,1.025);
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
   rdir->GetObject((rdirName+"_AssociatorByHits/chi2").c_str(),rh1);
   sdir->GetObject((sdirName+"_AssociatorByHits/chi2").c_str(),sh1);
   rdir->GetObject((rdirName+"_AssociatorByChi2/chi2").c_str(),rc1);
   sdir->GetObject((sdirName+"_AssociatorByChi2/chi2").c_str(),sc1);
   rdir->GetObject((rdirName+"_AssociatorByHits/chi2_prob").c_str(),rh2);
   sdir->GetObject((sdirName+"_AssociatorByHits/chi2_prob").c_str(),sh2);
   rdir->GetObject((rdirName+"_AssociatorByChi2/chi2_prob").c_str(),rc2);
   sdir->GetObject((sdirName+"_AssociatorByChi2/chi2_prob").c_str(),sc2);

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
   rdir->GetObject((rdirName+"_AssociatorByHits/hits_eta").c_str(),rh1);
   sdir->GetObject((sdirName+"_AssociatorByHits/hits_eta").c_str(),sh1);
   rdir->GetObject((rdirName+"_AssociatorByChi2/hits_eta").c_str(),rc1);
   sdir->GetObject((sdirName+"_AssociatorByChi2/hits_eta").c_str(),sc1);
   rdir->GetObject((rdirName+"_AssociatorByHits/chi2mean").c_str(),rh2);
   sdir->GetObject((sdirName+"_AssociatorByHits/chi2mean").c_str(),sh2);
   rdir->GetObject((rdirName+"_AssociatorByChi2/chi2mean").c_str(),rc2);
   sdir->GetObject((sdirName+"_AssociatorByChi2/chi2mean").c_str(),sc2);
   rdir->GetObject((rdirName+"_AssociatorByHits/losthits_eta").c_str(),rh3);
   sdir->GetObject((sdirName+"_AssociatorByHits/losthits_eta").c_str(),sh3);
   rdir->GetObject((rdirName+"_AssociatorByChi2/losthits_eta").c_str(),rc3);
   sdir->GetObject((sdirName+"_AssociatorByChi2/losthits_eta").c_str(),sc3);

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
   rdir->GetObject((rdirName+"_AssociatorByHits/pullPt").c_str(),rh1);
   sdir->GetObject((sdirName+"_AssociatorByHits/pullPt").c_str(),sh1);
   rdir->GetObject((rdirName+"_AssociatorByChi2/pullPt").c_str(),rc1);
   sdir->GetObject((sdirName+"_AssociatorByChi2/pullPt").c_str(),sc1);
   rdir->GetObject((rdirName+"_AssociatorByHits/pullQoverp").c_str(),rh2);
   sdir->GetObject((sdirName+"_AssociatorByHits/pullQoverp").c_str(),sh2);
   rdir->GetObject((rdirName+"_AssociatorByChi2/pullQoverp").c_str(),rc2);
   sdir->GetObject((sdirName+"_AssociatorByChi2/pullQoverp").c_str(),sc2);
   rdir->GetObject((rdirName+"_AssociatorByHits/pullPhi").c_str(),rh3);
   sdir->GetObject((sdirName+"_AssociatorByHits/pullPhi").c_str(),sh3);
   rdir->GetObject((rdirName+"_AssociatorByChi2/pullPhi").c_str(),rc3);
   sdir->GetObject((sdirName+"_AssociatorByChi2/pullPhi").c_str(),sc3);

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

   //pull Dxy, Dz, Theta
   rdir->GetObject((rdirName+"_AssociatorByHits/pullDxy").c_str(),rh1);
   sdir->GetObject((sdirName+"_AssociatorByHits/pullDxy").c_str(),sh1);
   rdir->GetObject((rdirName+"_AssociatorByChi2/pullDxy").c_str(),rc1);
   sdir->GetObject((sdirName+"_AssociatorByChi2/pullDxy").c_str(),sc1);
   rdir->GetObject((rdirName+"_AssociatorByHits/pullDz").c_str(),rh2);
   sdir->GetObject((sdirName+"_AssociatorByHits/pullDz").c_str(),sh2);
   rdir->GetObject((rdirName+"_AssociatorByChi2/pullDz").c_str(),rc2);
   sdir->GetObject((sdirName+"_AssociatorByChi2/pullDz").c_str(),sc2);
   rdir->GetObject((rdirName+"_AssociatorByHits/pullTheta").c_str(),rh3);
   sdir->GetObject((sdirName+"_AssociatorByHits/pullTheta").c_str(),sh3);
   rdir->GetObject((rdirName+"_AssociatorByChi2/pullTheta").c_str(),rc3);
   sdir->GetObject((sdirName+"_AssociatorByChi2/pullTheta").c_str(),sc3);

   canvas = new TCanvas("Tracks5","Tracks: pull of Dxy, Dz, Theta",1000,1000);
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
   
   canvas->Print("ctf_pullDxy_Dz_Theta.eps");
   canvas->Print("ctf_pullDxy_Dz_Theta.gif");

   //resolution Pt, Phi
   rdir->GetObject((rdirName+"_AssociatorByHits/sigmapt").c_str(),rh1);
   sdir->GetObject((sdirName+"_AssociatorByHits/sigmapt").c_str(),sh1);
   rdir->GetObject((rdirName+"_AssociatorByChi2/sigmapt").c_str(),rc1);
   sdir->GetObject((sdirName+"_AssociatorByChi2/sigmapt").c_str(),sc1);
   rdir->GetObject((rdirName+"_AssociatorByHits/sigmaphi").c_str(),rh2);
   sdir->GetObject((sdirName+"_AssociatorByHits/sigmaphi").c_str(),sh2);
   rdir->GetObject((rdirName+"_AssociatorByChi2/sigmaphi").c_str(),rc2);
   sdir->GetObject((sdirName+"_AssociatorByChi2/sigmaphi").c_str(),sc2);

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

   //resolution Dxy, Dz, Theta
   rdir->GetObject((rdirName+"_AssociatorByHits/sigmadxy").c_str(),rh1);
   sdir->GetObject((sdirName+"_AssociatorByHits/sigmadxy").c_str(),sh1);
   rdir->GetObject((rdirName+"_AssociatorByChi2/sigmadxy").c_str(),rc1);
   sdir->GetObject((sdirName+"_AssociatorByChi2/sigmadxy").c_str(),sc1);
   rdir->GetObject((rdirName+"_AssociatorByHits/sigmadz").c_str(),rh2);
   sdir->GetObject((sdirName+"_AssociatorByHits/sigmadz").c_str(),sh2);
   rdir->GetObject((rdirName+"_AssociatorByChi2/sigmadz").c_str(),rc2);
   sdir->GetObject((sdirName+"_AssociatorByChi2/sigmadz").c_str(),sc2);
   rdir->GetObject((rdirName+"_AssociatorByHits/sigmacotTheta").c_str(),rh3);
   sdir->GetObject((sdirName+"_AssociatorByHits/sigmacotTheta").c_str(),sh3);
   rdir->GetObject((rdirName+"_AssociatorByChi2/sigmacotTheta").c_str(),rc3);
   sdir->GetObject((sdirName+"_AssociatorByChi2/sigmacotTheta").c_str(),sc3);

   canvas = new TCanvas("Tracks7","Tracks: Dxy, Dz, Theta resolution",1000,1000);
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
   
   canvas->Print("ctf_resolDxy_Dz_Theta.eps");
   canvas->Print("ctf_resolDxy_Dz_Theta.gif");
 }



 //////////////////////////////////////
 /////////// RS //////////////////////
 //////////////////////////////////////
 if (rs){
   //efficiency&fakerate
   rdir->GetObject("cutsRS_AssociatorByHits/effic",rh1);
   sdir->GetObject("cutsRS_AssociatorByHits/effic",sh1);
   rdir->GetObject("cutsRS_AssociatorByChi2/effic",rc1);
   sdir->GetObject("cutsRS_AssociatorByChi2/effic",sc1);
   rdir->GetObject("cutsRS_AssociatorByHits/fakerate",rh2);
   sdir->GetObject("cutsRS_AssociatorByHits/fakerate",sh2);
   rdir->GetObject("cutsRS_AssociatorByChi2/fakerate",rc2);
   sdir->GetObject("cutsRS_AssociatorByChi2/fakerate",sc2);

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
   rdir->GetObject("cutsRS_AssociatorByHits/chi2",rh1);
   sdir->GetObject("cutsRS_AssociatorByHits/chi2",sh1);
   rdir->GetObject("cutsRS_AssociatorByChi2/chi2",rc1);
   sdir->GetObject("cutsRS_AssociatorByChi2/chi2",sc1);
   rdir->GetObject("cutsRS_AssociatorByHits/chi2_prob",rh2);
   sdir->GetObject("cutsRS_AssociatorByHits/chi2_prob",sh2);
   rdir->GetObject("cutsRS_AssociatorByChi2/chi2_prob",rc2);
   sdir->GetObject("cutsRS_AssociatorByChi2/chi2_prob",sc2);

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
   rdir->GetObject("cutsRS_AssociatorByHits/hits_eta",rh1);
   sdir->GetObject("cutsRS_AssociatorByHits/hits_eta",sh1);
   rdir->GetObject("cutsRS_AssociatorByChi2/hits_eta",rc1);
   sdir->GetObject("cutsRS_AssociatorByChi2/hits_eta",sc1);
   rdir->GetObject("cutsRS_AssociatorByHits/chi2mean",rh2);
   sdir->GetObject("cutsRS_AssociatorByHits/chi2mean",sh2);
   rdir->GetObject("cutsRS_AssociatorByChi2/chi2mean",rc2);
   sdir->GetObject("cutsRS_AssociatorByChi2/chi2mean",sc2);
   rdir->GetObject("cutsRS_AssociatorByHits/losthits_eta",rh3);
   sdir->GetObject("cutsRS_AssociatorByHits/losthits_eta",sh3);
   rdir->GetObject("cutsRS_AssociatorByChi2/losthits_eta",rc3);
   sdir->GetObject("cutsRS_AssociatorByChi2/losthits_eta",sc3);

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
   rdir->GetObject("cutsRS_AssociatorByHits/pullPt",rh1);
   sdir->GetObject("cutsRS_AssociatorByHits/pullPt",sh1);
   rdir->GetObject("cutsRS_AssociatorByChi2/pullPt",rc1);
   sdir->GetObject("cutsRS_AssociatorByChi2/pullPt",sc1);
   rdir->GetObject("cutsRS_AssociatorByHits/pullQoverp",rh2);
   sdir->GetObject("cutsRS_AssociatorByHits/pullQoverp",sh2);
   rdir->GetObject("cutsRS_AssociatorByChi2/pullQoverp",rc2);
   sdir->GetObject("cutsRS_AssociatorByChi2/pullQoverp",sc2);
   rdir->GetObject("cutsRS_AssociatorByHits/pullPhi",rh3);
   sdir->GetObject("cutsRS_AssociatorByHits/pullPhi",sh3);
   rdir->GetObject("cutsRS_AssociatorByChi2/pullPhi",rc3);
   sdir->GetObject("cutsRS_AssociatorByChi2/pullPhi",sc3);

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

   //pull Dxy, Dz, Theta
   rdir->GetObject("cutsRS_AssociatorByHits/pullDxy",rh1);
   sdir->GetObject("cutsRS_AssociatorByHits/pullDxy",sh1);
   rdir->GetObject("cutsRS_AssociatorByChi2/pullDxy",rc1);
   sdir->GetObject("cutsRS_AssociatorByChi2/pullDxy",sc1);
   rdir->GetObject("cutsRS_AssociatorByHits/pullDz",rh2);
   sdir->GetObject("cutsRS_AssociatorByHits/pullDz",sh2);
   rdir->GetObject("cutsRS_AssociatorByChi2/pullDz",rc2);
   sdir->GetObject("cutsRS_AssociatorByChi2/pullDz",sc2);
   rdir->GetObject("cutsRS_AssociatorByHits/pullTheta",rh3);
   sdir->GetObject("cutsRS_AssociatorByHits/pullTheta",sh3);
   rdir->GetObject("cutsRS_AssociatorByChi2/pullTheta",rc3);
   sdir->GetObject("cutsRS_AssociatorByChi2/pullTheta",sc3);

   canvas = new TCanvas("Tracks12","Tracks: pull of Dxy, Dz, Theta",1000,1000);
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
   
   canvas->Print("rs_pullDxy_Dz_Theta.eps");
   canvas->Print("rs_pullDxy_Dz_Theta.gif");

   //resolution Pt, Phi
   rdir->GetObject("cutsRS_AssociatorByHits/sigmapt",rh1);
   sdir->GetObject("cutsRS_AssociatorByHits/sigmapt",sh1);
   rdir->GetObject("cutsRS_AssociatorByChi2/sigmapt",rc1);
   sdir->GetObject("cutsRS_AssociatorByChi2/sigmapt",sc1);
   rdir->GetObject("cutsRS_AssociatorByHits/sigmaphi",rh2);
   sdir->GetObject("cutsRS_AssociatorByHits/sigmaphi",sh2);
   rdir->GetObject("cutsRS_AssociatorByChi2/sigmaphi",rc2);
   sdir->GetObject("cutsRS_AssociatorByChi2/sigmaphi",sc2);

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

   //resolution Dxy, Dz, Theta
   rdir->GetObject("cutsRS_AssociatorByHits/sigmadxy",rh1);
   sdir->GetObject("cutsRS_AssociatorByHits/sigmadxy",sh1);
   rdir->GetObject("cutsRS_AssociatorByChi2/sigmadxy",rc1);
   sdir->GetObject("cutsRS_AssociatorByChi2/sigmadxy",sc1);
   rdir->GetObject("cutsRS_AssociatorByHits/sigmadz",rh2);
   sdir->GetObject("cutsRS_AssociatorByHits/sigmadz",sh2);
   rdir->GetObject("cutsRS_AssociatorByChi2/sigmadz",rc2);
   sdir->GetObject("cutsRS_AssociatorByChi2/sigmadz",sc2);
   rdir->GetObject("cutsRS_AssociatorByHits/sigmacotTheta",rh3);
   sdir->GetObject("cutsRS_AssociatorByHits/sigmacotTheta",sh3);
   rdir->GetObject("cutsRS_AssociatorByChi2/sigmacotTheta",rc3);
   sdir->GetObject("cutsRS_AssociatorByChi2/sigmacotTheta",sc3);

   canvas = new TCanvas("Tracks14","Tracks: Dxy, Dz, Theta resolution",1000,1000);
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
   
   canvas->Print("rs_resolDxy_Dz_Theta.eps");
   canvas->Print("rs_resolDxy_Dz_Theta.gif");
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

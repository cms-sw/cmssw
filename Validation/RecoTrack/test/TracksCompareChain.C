void TracksCompareChain()
{

 gROOT ->Reset();
 // gROOT ->SetBatch();
 gROOT->SetStyle("Plain");

 char*  rfilename = "validationPlots.root";//new release is in red
 char*  sfilename = "../validationPlots.root";//reference is in blue

 delete gROOT->GetListOfFiles()->FindObject(rfilename);
 delete gROOT->GetListOfFiles()->FindObject(sfilename); 

 TText* te = new TText();
 TFile * rfile = new TFile(rfilename);
 TDirectory * rdir=gDirectory; 
 TFile * sfile = new TFile(sfilename);
 TDirectory * sdir=gDirectory; 

 if(rfile->cd("DQMData/Run 1/RecoTrackV"))rfile->cd("DQMData/Run 1/RecoTrackV/Run summary/Track");
 else rfile->cd("DQMData/RecoTrackV/Track");
 rdir=gDirectory;

 if(sfile->cd("DQMData/Run 1/RecoTrackV"))sfile->cd("DQMData/Run 1/RecoTrackV/Run summary/Track");
 else sfile->cd("DQMData/RecoTrackV/Track");
 sdir=gDirectory; 


 TString collnamerCTF = "general";//cutsReco
 TString collnamesCTF = "general";//cutsReco
 // TString collnamesCTF = "cutsReco";
 TString collnamerRS = "cutsRS";
 TString collnamesRS = "cutsRS";
 
 TString assocnamesCTF1 = "trackingParticleRecoAsssociation";//AssociatorByHits
 //TString assocnamesCTF1 = "trackingParticleRecoAsssociation";//AssociatorByHits
 // TString assocnamerCTF1 = "AssociatorByHits";
  TString assocnamerCTF1 = "AssociatorByHits";
 TString assocnamerRS1 = "AssociatorByHits";
 TString assocnamesRS1 = "AssociatorByHits";
 TString assocnamerCTF2 = "AssociatorByChi2";
 TString assocnamesCTF2 = "AssociatorByChi2";
 TString assocnamerRS2 = "AssociatorByChi2";
 TString assocnamesRS2 = "AssociatorByChi2";

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
 bool chi2=0;
 bool ctf=1;
 bool rs=0;

 //////////////////////////////////////
 /////////// CTF //////////////////////
 //////////////////////////////////////
 if (ctf){
//    string rdirName,sdirName;
//    if(rdir->cd("cutsReco_AssociatorByHits"))rdirName="cutsReco";
//    else if(rdir->cd("general_AssociatorByHits"))rdirName="general";
//    else rdirName="cutsCKF";
//    if(sdir->cd("cutsReco_AssociatorByHits"))sdirName="cutsReco";
//    else if(sdir->cd("general_AssociatorByHits"))sdirName="general";
//    else sdirName="cutsCKF";
//    //efficiency&fakerate
//    cout<<rdir->GetPath()<<endl;
//    cout<<sdir->GetPath()<<endl;
//    cout<<sdir->ls()<<endl;
   cout<<collnamerCTF+"_"+assocnamerCTF1+"/effic"<<endl;
   rdir->ls();
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF1+"/effic",rh1);
   sdir->ls();
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF1+"/effic",sh1);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF2+"/effic",rc1);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF2+"/effic",sc1);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF1+"/fakerate",rh2);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF1+"/fakerate",sh2);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF2+"/fakerate",rc2);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF2+"/fakerate",sc2);

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
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF1+"/chi2",rh1);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF1+"/chi2",sh1);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF2+"/chi2",rc1);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF2+"/chi2",sc1);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF1+"/chi2_prob",rh2);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF1+"/chi2_prob",sh2);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF2+"/chi2_prob",rc2);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF2+"/chi2_prob",sc2);

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
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF1+"/nhits_vs_eta_pfx",(TProfile *)rh1);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF1+"/hits_eta",sh1);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF2+"/nhits_vs_eta_pfx",(TProfile *)rc1);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF2+"/hits_eta",sc1);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF1+"/chi2_vs_eta_pfx",(TProfile *)rh2);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF1+"/chi2mean",sh2);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF2+"/chi2_vs_eta_pfx",(TProfile *)rc2);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF2+"/chi2mean",sc2);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF1+"/nlosthits_vs_eta_pfx",(TProfile *)rh3);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF1+"/losthits_eta",sh3);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF2+"/nlosthits_vs_eta_pfx",(TProfile *)rc3);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF2+"/losthits_eta",sc3);

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
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF1+"/pullPt",rh1);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF1+"/pullPt",sh1);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF2+"/pullPt",rc1);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF2+"/pullPt",sc1);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF1+"/pullQoverp",rh2);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF1+"/pullQoverp",sh2);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF2+"/pullQoverp",rc2);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF2+"/pullQoverp",sc2);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF1+"/pullPhi",rh3);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF1+"/pullPhi",sh3);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF2+"/pullPhi",rc3);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF2+"/pullPhi",sc3);

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
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF1+"/pullDxy",rh1);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF1+"/pullDxy",sh1);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF2+"/pullDxy",rc1);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF2+"/pullDxy",sc1);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF1+"/pullDz",rh2);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF1+"/pullDz",sh2);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF2+"/pullDz",rc2);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF2+"/pullDz",sc2);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF1+"/pullTheta",rh3);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF1+"/pullTheta",sh3);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF2+"/pullTheta",rc3);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF2+"/pullTheta",sc3);

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
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF1+"/ptres_vs_eta_Sigma",rh1);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF1+"/sigmapt",sh1);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF2+"/ptres_vs_eta_Sigma",rc1);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF2+"/sigmapt",sc1);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF1+"/phires_vs_eta_Sigma",rh2);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF1+"/sigmaphi",sh2);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF2+"/phires_vs_eta_Sigma",rc2);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF2+"/sigmaphi",sc2);

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
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF1+"/dxyres_vs_eta_Sigma",rh1);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF1+"/sigmadxy",sh1);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF2+"/dxyres_vs_eta_Sigma",rc1);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF2+"/sigmadxy",sc1);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF1+"/dzres_vs_eta_Sigma",rh2);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF1+"/sigmadz",sh2);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF2+"/dzres_vs_eta_Sigma",rc2);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF2+"/sigmadz",sc2);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF1+"/cotThetares_vs_eta_Sigma",rh3);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF1+"/sigmacotTheta",sh3);
   rdir->GetObject(collnamerCTF+"_"+assocnamerCTF2+"/cotThetares_vs_eta_Sigma",rc3);
   sdir->GetObject(collnamesCTF+"_"+assocnamesCTF2+"/sigmacotTheta",sc3);

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
   rdir->GetObject(collnamerRS+"_"+assocnamerRS1+"/effic",rh1);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS1+"/effic",sh1);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS2+"/effic",rc1);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS2+"/effic",sc1);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS1+"/fakerate",rh2);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS1+"/fakerate",sh2);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS2+"/fakerate",rc2);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS2+"/fakerate",sc2);

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
   rdir->GetObject(collnamerRS+"_"+assocnamerRS1+"/chi2",rh1);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS1+"/chi2",sh1);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS2+"/chi2",rc1);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS2+"/chi2",sc1);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS1+"/chi2_prob",rh2);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS1+"/chi2_prob",sh2);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS2+"/chi2_prob",rc2);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS2+"/chi2_prob",sc2);

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
   rdir->GetObject(collnamerRS+"_"+assocnamerRS1+"/hits_eta",rh1);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS1+"/hits_eta",sh1);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS2+"/hits_eta",rc1);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS2+"/hits_eta",sc1);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS1+"/chi2mean",rh2);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS1+"/chi2mean",sh2);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS2+"/chi2mean",rc2);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS2+"/chi2mean",sc2);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS1+"/losthits_eta",rh3);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS1+"/losthits_eta",sh3);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS2+"/losthits_eta",rc3);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS2+"/losthits_eta",sc3);

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
   rdir->GetObject(collnamerRS+"_"+assocnamerRS1+"/pullPt",rh1);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS1+"/pullPt",sh1);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS2+"/pullPt",rc1);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS2+"/pullPt",sc1);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS1+"/pullQoverp",rh2);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS1+"/pullQoverp",sh2);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS2+"/pullQoverp",rc2);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS2+"/pullQoverp",sc2);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS1+"/pullPhi",rh3);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS1+"/pullPhi",sh3);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS2+"/pullPhi",rc3);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS2+"/pullPhi",sc3);

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
   rdir->GetObject(collnamerRS+"_"+assocnamerRS1+"/pullDxy",rh1);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS1+"/pullDxy",sh1);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS2+"/pullDxy",rc1);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS2+"/pullDxy",sc1);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS1+"/pullDz",rh2);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS1+"/pullDz",sh2);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS2+"/pullDz",rc2);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS2+"/pullDz",sc2);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS1+"/pullTheta",rh3);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS1+"/pullTheta",sh3);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS2+"/pullTheta",rc3);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS2+"/pullTheta",sc3);

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
   rdir->GetObject(collnamerRS+"_"+assocnamerRS1+"/sigmapt",rh1);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS1+"/sigmapt",sh1);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS2+"/sigmapt",rc1);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS2+"/sigmapt",sc1);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS1+"/sigmaphi",rh2);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS1+"/sigmaphi",sh2);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS2+"/sigmaphi",rc2);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS2+"/sigmaphi",sc2);

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
   rdir->GetObject(collnamerRS+"_"+assocnamerRS1+"/sigmadxy",rh1);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS1+"/sigmadxy",sh1);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS2+"/sigmadxy",rc1);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS2+"/sigmadxy",sc1);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS1+"/sigmadz",rh2);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS1+"/sigmadz",sh2);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS2+"/sigmadz",rc2);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS2+"/sigmadz",sc2);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS1+"/sigmacotTheta",rh3);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS1+"/sigmacotTheta",sh3);
   rdir->GetObject(collnamerRS+"_"+assocnamerRS2+"/sigmacotTheta",rc3);
   sdir->GetObject(collnamesRS+"_"+assocnamesRS2+"/sigmacotTheta",sc3);

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

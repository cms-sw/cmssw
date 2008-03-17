void TracksCompare()
{

 gROOT ->Reset();
 gROOT ->SetBatch();
 gROOT->SetStyle("Plain");

 // gROOT->ProcessLine(".L HistoCompare_Tracks.C");
 // gROOT->ProcessLine(".L TracksCompare.C");

 char*  rfilename = "file05.root";//new release is in red
 char*  sfilename = "file15.root";//reference is in blue

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
   rdir->GetObject("cutsCKF_AssociatorByHits/effic",rh1);
   sdir->GetObject("cutsCKF_AssociatorByHits/effic",sh1);
   rdir->GetObject("cutsCKF_AssociatorByChi2/effic",rc1);
   sdir->GetObject("cutsCKF_AssociatorByChi2/effic",sc1);
   rdir->GetObject("cutsCKF_AssociatorByHits/fakerate",rh2);
   sdir->GetObject("cutsCKF_AssociatorByHits/fakerate",sh2);
   rdir->GetObject("cutsCKF_AssociatorByChi2/fakerate",rc2);
   sdir->GetObject("cutsCKF_AssociatorByChi2/fakerate",sc2);

   canvas = new TCanvas("Tracks1","Tracks: efficiency & fakerate",1000,1000);

   if (hit) rh1->GetYaxis()->SetRangeUser(0.7,1.025);
   if (hit) sh1->GetYaxis()->SetRangeUser(0.7,1.025);
   if (chi2)rc1->GetYaxis()->SetRangeUser(0.7,1.025);
   if (chi2)sc1->GetYaxis()->SetRangeUser(0.7,1.025);

   if (hit&&chi2) plotHist22(canvas,sh1,rh1,sc1,rc1,sh2,rh2,sc2,rc2,myPV,te,"UU",-1);
   else if (hit)  plotHist12(canvas,sh1,rh1,sh2,rh2,myPV,te,"UU",-1);
   else if (chi2) plotHist12(canvas,sc1,rc1,sc2,rc2,myPV,te,"UU",-1);

   canvas->Print("ctf_effic_fake.eps");
   canvas->Print("ctf_effic_fake.gif");

   //chi2&chi2 probability
   rdir->GetObject("cutsCKF_AssociatorByHits/chi2",rh1);
   sdir->GetObject("cutsCKF_AssociatorByHits/chi2",sh1);
   rdir->GetObject("cutsCKF_AssociatorByChi2/chi2",rc1);
   sdir->GetObject("cutsCKF_AssociatorByChi2/chi2",sc1);
   rdir->GetObject("cutsCKF_AssociatorByHits/chi2_prob",rh2);
   sdir->GetObject("cutsCKF_AssociatorByHits/chi2_prob",sh2);
   rdir->GetObject("cutsCKF_AssociatorByChi2/chi2_prob",rc2);
   sdir->GetObject("cutsCKF_AssociatorByChi2/chi2_prob",sc2);

   canvas = new TCanvas("Tracks2","Tracks: chi2 & chi2 probability",1000,1000);

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

   if (hit&&chi2) plotHist22(canvas,sh1,rh1,sc1,rc1,sh2,rh2,sc2,rc2,myPV,te,"UUNORM",0.4,0.7);
   else if (hit)  plotHist12(canvas,sh1,rh1,sh2,rh2,myPV,te,"UUNORM",0.4,0.7);
   else if (chi2) plotHist12(canvas,sc1,rc1,sc2,rc2,myPV,te,"UUNORM",0.4,0.7);

   canvas->Print("ctf_chi2_chi2prob.eps");
   canvas->Print("ctf_chi2_chi2prob.gif");

   //meanchi2 and #hits vs eta
   rdir->GetObject("cutsCKF_AssociatorByHits/hits_eta",rh1);
   sdir->GetObject("cutsCKF_AssociatorByHits/hits_eta",sh1);
   rdir->GetObject("cutsCKF_AssociatorByChi2/hits_eta",rc1);
   sdir->GetObject("cutsCKF_AssociatorByChi2/hits_eta",sc1);
   rdir->GetObject("cutsCKF_AssociatorByHits/chi2mean",rh2);
   sdir->GetObject("cutsCKF_AssociatorByHits/chi2mean",sh2);
   rdir->GetObject("cutsCKF_AssociatorByChi2/chi2mean",rc2);
   sdir->GetObject("cutsCKF_AssociatorByChi2/chi2mean",sc2);
   rdir->GetObject("cutsCKF_AssociatorByHits/losthits_eta",rh3);
   sdir->GetObject("cutsCKF_AssociatorByHits/losthits_eta",sh3);
   rdir->GetObject("cutsCKF_AssociatorByChi2/losthits_eta",rc3);
   sdir->GetObject("cutsCKF_AssociatorByChi2/losthits_eta",sc3);

   canvas = new TCanvas("Tracks3","Tracks: chi2 and #hits vs eta",1000,1000);

   //fixRangeY(rh1,sh1);
   //fixRangeY(rc1,sc1);
   if (hit)  fixRangeY(rh2,sh2);
   if (chi2) fixRangeY(rc2,sc2);

   if (hit&&chi2) plotHist23(canvas,sh1,rh1,sc1,rc1,sh2,rh2,sc2,rc2,sh3,rh3,sc3,rc3,myPV,te, "UU",-1);
   else if (hit)  plotHist13(canvas,sh1,rh1,sh2,rh2,sh3,rh3,myPV,te,"UU",-1);
   else if (chi2) plotHist13(canvas,sc1,rc1,sc2,rc2,sc3,rc3,myPV,te,"UU",-1);
 
   canvas->Print("ctf_hitseta_chi2mean.eps");
   canvas->Print("ctf_hitseta_chi2mean.gif");

   //pull Pt, Qoverp, Phi
   rdir->GetObject("cutsCKF_AssociatorByHits/pullPt",rh1);
   sdir->GetObject("cutsCKF_AssociatorByHits/pullPt",sh1);
   rdir->GetObject("cutsCKF_AssociatorByChi2/pullPt",rc1);
   sdir->GetObject("cutsCKF_AssociatorByChi2/pullPt",sc1);
   rdir->GetObject("cutsCKF_AssociatorByHits/pullQoverp",rh2);
   sdir->GetObject("cutsCKF_AssociatorByHits/pullQoverp",sh2);
   rdir->GetObject("cutsCKF_AssociatorByChi2/pullQoverp",rc2);
   sdir->GetObject("cutsCKF_AssociatorByChi2/pullQoverp",sc2);
   rdir->GetObject("cutsCKF_AssociatorByHits/pullPhi",rh3);
   sdir->GetObject("cutsCKF_AssociatorByHits/pullPhi",sh3);
   rdir->GetObject("cutsCKF_AssociatorByChi2/pullPhi",rc3);
   sdir->GetObject("cutsCKF_AssociatorByChi2/pullPhi",sc3);

   canvas = new TCanvas("Tracks4","Tracks: pull of Pt, Qoverp and Phi",1000,1000);

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

   if (hit&&chi2) plotHist23(canvas,sh1,rh1,sc1,rc1,sh2,rh2,sc2,rc2,sh3,rh3,sc3,rc3,myPV,te,"UUNORM",0.4,0.1,true);
   else if (hit)  plotHist13(canvas,sh1,rh1,sh2,rh2,sh3,rh3,myPV,te,"UUNORM",0.4,0.1,true);
   else if (chi2) plotHist13(canvas,sc1,rc1,sc2,rc2,sc3,rc3,myPV,te,"UUNORM",0.4,0.1,true);

   canvas->Print("ctf_pullPt_Qoverp_Phi.eps");
   canvas->Print("ctf_pullPt_Qoverp_Phi.gif");

   //pull Dxy, Dz, Theta
   rdir->GetObject("cutsCKF_AssociatorByHits/pullDxy",rh1);
   sdir->GetObject("cutsCKF_AssociatorByHits/pullDxy",sh1);
   rdir->GetObject("cutsCKF_AssociatorByChi2/pullDxy",rc1);
   sdir->GetObject("cutsCKF_AssociatorByChi2/pullDxy",sc1);
   rdir->GetObject("cutsCKF_AssociatorByHits/pullDz",rh2);
   sdir->GetObject("cutsCKF_AssociatorByHits/pullDz",sh2);
   rdir->GetObject("cutsCKF_AssociatorByChi2/pullDz",rc2);
   sdir->GetObject("cutsCKF_AssociatorByChi2/pullDz",sc2);
   rdir->GetObject("cutsCKF_AssociatorByHits/pullTheta",rh3);
   sdir->GetObject("cutsCKF_AssociatorByHits/pullTheta",sh3);
   rdir->GetObject("cutsCKF_AssociatorByChi2/pullTheta",rc3);
   sdir->GetObject("cutsCKF_AssociatorByChi2/pullTheta",sc3);

   canvas = new TCanvas("Tracks5","Tracks: pull of Dxy, Dz, Theta",1000,1000);

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

   if (hit&&chi2) plotHist23(canvas,sh1,rh1,sc1,rc1,sh2,rh2,sc2,rc2,sh3,rh3,sc3,rc3,myPV,te,"UUNORM",0.4,0.1,true);
   else if (hit)  plotHist13(canvas,sh1,rh1,sh2,rh2,sh3,rh3,myPV,te,"UUNORM",0.4,0.1,true);
   else if (chi2) plotHist13(canvas,sc1,rc1,sc2,rc2,sc3,rc3,myPV,te,"UUNORM",0.4,0.1,true);
 
   canvas->Print("ctf_pullDxy_Dz_Theta.eps");
   canvas->Print("ctf_pullDxy_Dz_Theta.gif");

   //resolution Pt, Phi
   rdir->GetObject("cutsCKF_AssociatorByHits/sigmapt",rh1);
   sdir->GetObject("cutsCKF_AssociatorByHits/sigmapt",sh1);
   rdir->GetObject("cutsCKF_AssociatorByChi2/sigmapt",rc1);
   sdir->GetObject("cutsCKF_AssociatorByChi2/sigmapt",sc1);
   rdir->GetObject("cutsCKF_AssociatorByHits/sigmaphi",rh2);
   sdir->GetObject("cutsCKF_AssociatorByHits/sigmaphi",sh2);
   rdir->GetObject("cutsCKF_AssociatorByChi2/sigmaphi",rc2);
   sdir->GetObject("cutsCKF_AssociatorByChi2/sigmaphi",sc2);

   canvas = new TCanvas("Tracks6","Tracks: Pt and Phi resolution",1000,1000);

   if (hit&&chi2) plotHist22(canvas,sh1,rh1,sc1,rc1,sh2,rh2,sc2,rc2,myPV,te,"UU",-1);
   else if (hit)  plotHist12(canvas,sh1,rh1,sh2,rh2,myPV,te,"UU",-1);
   else if (chi2) plotHist12(canvas,sc1,rc1,sc2,rc2,myPV,te,"UU",-1);
 
   canvas->Print("ctf_resolPt_Phi.eps");
   canvas->Print("ctf_resolPt_Phi.gif");

   //resolution Dxy, Dz, Theta
   rdir->GetObject("cutsCKF_AssociatorByHits/sigmadxy",rh1);
   sdir->GetObject("cutsCKF_AssociatorByHits/sigmadxy",sh1);
   rdir->GetObject("cutsCKF_AssociatorByChi2/sigmadxy",rc1);
   sdir->GetObject("cutsCKF_AssociatorByChi2/sigmadxy",sc1);
   rdir->GetObject("cutsCKF_AssociatorByHits/sigmadz",rh2);
   sdir->GetObject("cutsCKF_AssociatorByHits/sigmadz",sh2);
   rdir->GetObject("cutsCKF_AssociatorByChi2/sigmadz",rc2);
   sdir->GetObject("cutsCKF_AssociatorByChi2/sigmadz",sc2);
   rdir->GetObject("cutsCKF_AssociatorByHits/sigmacotTheta",rh3);
   sdir->GetObject("cutsCKF_AssociatorByHits/sigmacotTheta",sh3);
   rdir->GetObject("cutsCKF_AssociatorByChi2/sigmacotTheta",rc3);
   sdir->GetObject("cutsCKF_AssociatorByChi2/sigmacotTheta",sc3);

   canvas = new TCanvas("Tracks7","Tracks: Dxy, Dz, Theta resolution",1000,1000);

   if (hit&&chi2) plotHist23(canvas,sh1,rh1,sc1,rc1,sh2,rh2,sc2,rc2,sh3,rh3,sc3,rc3,myPV,te,"UU",-1);
   else if (hit)  plotHist13(canvas,sh1,rh1,sh2,rh2,sh3,rh3,myPV,te,"UU",-1);
   else if (chi2) plotHist13(canvas,sc1,rc1,sc2,rc2,sc3,rc3,myPV,te,"UU",-1);
 
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

   if (hit) rh1->GetYaxis()->SetRangeUser(0.7,1.025);
   if (hit) sh1->GetYaxis()->SetRangeUser(0.7,1.025);
   if (chi2)rc1->GetYaxis()->SetRangeUser(0.7,1.025);
   if (chi2)sc1->GetYaxis()->SetRangeUser(0.7,1.025);

   if (hit&&chi2) plotHist22(canvas,sh1,rh1,sc1,rc1,sh2,rh2,sc2,rc2,myPV,te,"UU",-1);
   else if (hit)  plotHist12(canvas,sh1,rh1,sh2,rh2,myPV,te,"UU",-1);
   else if (chi2) plotHist12(canvas,sc1,rc1,sc2,rc2,myPV,te,"UU",-1);

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

   if (hit&&chi2) plotHist22(canvas,sh1,rh1,sc1,rc1,sh2,rh2,sc2,rc2,myPV,te,"UUNORM",0.4,0.7);
   else if (hit)  plotHist12(canvas,sh1,rh1,sh2,rh2,myPV,te,"UUNORM",0.4,0.7);
   else if (chi2) plotHist12(canvas,sc1,rc1,sc2,rc2,myPV,te,"UUNORM",0.4,0.7);

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

   if (hit) fixRangeY(rh2,sh2);
   if (chi2) fixRangeY(rc2,sc2);

   if (hit&&chi2) plotHist23(canvas,sh1,rh1,sc1,rc1,sh2,rh2,sc2,rc2,sh3,rh3,sc3,rc3,myPV,te, "UU",-1);
   else if (hit)  plotHist13(canvas,sh1,rh1,sh2,rh2,sh3,rh3,myPV,te,"UU",-1);
   else if (chi2) plotHist13(canvas,sc1,rc1,sc2,rc2,sc3,rc3,myPV,te,"UU",-1);
 
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
   
   if (hit&&chi2) plotHist23(canvas,sh1,rh1,sc1,rc1,sh2,rh2,sc2,rc2,sh3,rh3,sc3,rc3,myPV,te,"UUNORM",0.4,0.1,true);
   else if (hit)  plotHist13(canvas,sh1,rh1,sh2,rh2,sh3,rh3,myPV,te,"UUNORM",0.4,0.1,true);
   else if (chi2) plotHist13(canvas,sc1,rc1,sc2,rc2,sc3,rc3,myPV,te,"UUNORM",0.4,0.1,true);
 
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

   if (hit&&chi2) plotHist23(canvas,sh1,rh1,sc1,rc1,sh2,rh2,sc2,rc2,sh3,rh3,sc3,rc3,myPV,te,"UUNORM",0.4,0.1,true);
   else if (hit)  plotHist13(canvas,sh1,rh1,sh2,rh2,sh3,rh3,myPV,te,"UUNORM",0.4,0.1,true);
   else if (chi2) plotHist13(canvas,sc1,rc1,sc2,rc2,sc3,rc3,myPV,te,"UUNORM",0.4,0.1,true);
 
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

   if (hit&&chi2) plotHist22(canvas,sh1,rh1,sc1,rc1,sh2,rh2,sc2,rc2,myPV,te,"UU",-1);
   else if (hit)  plotHist12(canvas,sh1,rh1,sh2,rh2,myPV,te,"UU",-1);
   else if (chi2) plotHist12(canvas,sc1,rc1,sc2,rc2,myPV,te,"UU",-1);
 
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

   if (hit&&chi2) plotHist23(canvas,sh1,rh1,sc1,rc1,sh2,rh2,sc2,rc2,sh3,rh3,sc3,rc3,myPV,te,"UU",-1);
   else if (hit)  plotHist13(canvas,sh1,rh1,sh2,rh2,sh3,rh3,myPV,te,"UU",-1);
   else if (chi2) plotHist13(canvas,sc1,rc1,sc2,rc2,sc3,rc3,myPV,te,"UU",-1);
 
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


void plotHist12(TCanvas *canvas, 
		TH1F *sh1,TH1F *rh1,
		TH1F *sh2,TH1F *rh2,
		HistoCompare_Tracks * myPV, TText* te,
		char * option, double startingY, double startingX = .1,bool fit = false){
  canvas->Divide(1,2);

  canvas->cd(1);
  rh1->SetLineColor(2);
  sh1->SetLineColor(4);
  sh1->SetLineStyle(2);
  setStats(rh1,sh1, startingY, startingX, fit);
  rh1->Draw();
  sh1->Draw("sames");
  myPV->PVCompute(rh1, sh1, te, option );
  
  canvas->cd(2);
  rh2->SetLineColor(2);
  sh2->SetLineColor(4);
  sh2->SetLineStyle(2);
  setStats(rh2,sh2, startingY, startingX, fit);
  rh2->Draw();
  sh2->Draw("sames");
  myPV->PVCompute(rh2, sh2, te, option );  
}

void plotHist22(TCanvas *canvas, 
		TH1F *sh1,TH1F *rh1, TH1F *sc1,TH1F *rc1, 
		TH1F *sh2,TH1F *rh2, TH1F *sc2,TH1F *rc2,
		HistoCompare_Tracks * myPV, TText* te,
		char * option, double startingY, double startingX = .1,bool fit = false){
  canvas->Divide(2,2);

  canvas->cd(1);
  rh1->SetLineColor(2);
  sh1->SetLineColor(4);
  sh1->SetLineStyle(1);
  setStats(rh1,sh1, startingY, startingX, fit);
  rh1->Draw();
  sh1->Draw("sames");
  myPV->PVCompute(rh1, sh1, te, option );
  
  canvas->cd(2);
  rc1->SetLineColor(2);
  sc1->SetLineColor(4);
  sc1->SetLineStyle(2);
  setStats(rc1,sc1, startingY, startingX, fit);
  rc1->Draw();
  sc1->Draw("sames");
  myPV->PVCompute(rc1, sc1, te, option );
  
  canvas->cd(3);
  rh2->SetLineColor(2);
  sh2->SetLineColor(4);
  sh2->SetLineStyle(2);
  setStats(rh2,sh2, startingY, startingX, fit);
  rh2->Draw();
  sh2->Draw("sames");
  myPV->PVCompute(rh2, sh2, te, option );
  
  canvas->cd(4);
  rc2->SetLineColor(2);
  sc2->SetLineColor(4);
  sc2->SetLineStyle(2);
  setStats(rc2,sc2, startingY, startingX, fit);
  rc2->Draw();
  sc2->Draw("sames");
  myPV->PVCompute(rc2, sc2, te, option );
  
}

void plotHist13(TCanvas *canvas, 
		TH1F *sh1,TH1F *rh1,
 		TH1F *sh2,TH1F *rh2,
 		TH1F *sh3,TH1F *rh3,
		HistoCompare_Tracks * myPV, TText* te,
		char * option, double startingY, double startingX = .1,bool fit = false){
  canvas->Divide(1,3);

  canvas->cd(1);
  rh1->SetLineColor(2);
  sh1->SetLineColor(4);
  sh1->SetLineStyle(2);
  setStats(rh1,sh1, startingY, startingX, fit);
  rh1->Draw();
  sh1->Draw("sames");
  myPV->PVCompute(rh1, sh1, te, option );
  
  canvas->cd(2);
  rh2->SetLineColor(2);
  sh2->SetLineColor(4);
  sh2->SetLineStyle(2);
  setStats(rh2,sh2, startingY, startingX, fit);
  rh2->Draw();
  sh2->Draw("sames");
  myPV->PVCompute(rh2, sh2, te, option );

  canvas->cd(3);
  rh3->SetLineColor(2);
  sh3->SetLineColor(4);
  sh3->SetLineStyle(2);
  setStats(rh3,sh3, startingY, startingX, fit);
  rh3->Draw();
  sh3->Draw("sames");
  myPV->PVCompute(rh3, sh3, te, option );
}

void plotHist23(TCanvas *canvas, 
		TH1F *sh1,TH1F *rh1, TH1F *sc1,TH1F *rc1, 
 		TH1F *sh2,TH1F *rh2, TH1F *sc2,TH1F *rc2,
 		TH1F *sh3,TH1F *rh3, TH1F *sc3,TH1F *rc3,
		HistoCompare_Tracks * myPV, TText* te,
		char * option, double startingY, double startingX = .1,bool fit = false){
  canvas->Divide(2,3);

  canvas->cd(1);
  rh1->SetLineColor(2);
  sh1->SetLineColor(4);
  sh1->SetLineStyle(2);
  setStats(rh1,sh1, startingY, startingX, fit);
  rh1->Draw();
  sh1->Draw("sames");
  myPV->PVCompute(rh1, sh1, te, option );
  
  canvas->cd(2);
  rc1->SetLineColor(2);
  sc1->SetLineColor(4);
  sc1->SetLineStyle(2);
  setStats(rc1,sc1, startingY, startingX, fit);
  rc1->Draw();
  sc1->Draw("sames");
  myPV->PVCompute(rc1, sc1, te, option );
  
  canvas->cd(3);
  rh2->SetLineColor(2);
  sh2->SetLineColor(4);
  sh2->SetLineStyle(2);
  setStats(rh2,sh2, startingY, startingX, fit);
  rh2->Draw();
  sh2->Draw("sames");
  myPV->PVCompute(rh2, sh2, te, option );
  
  canvas->cd(4);
  rc2->SetLineColor(2);
  sc2->SetLineColor(4);
  sc2->SetLineStyle(2);
  setStats(rc2,sc2, startingY, startingX, fit);
  rc2->Draw();
  sc2->Draw("sames");
  myPV->PVCompute(rc2, sc2, te, option );

  canvas->cd(5);
  rh3->SetLineColor(2);
  sh3->SetLineColor(4);
  sh3->SetLineStyle(2);
  setStats(rh3,sh3, startingY, startingX, fit);
  rh3->Draw();
  sh3->Draw("sames");
  myPV->PVCompute(rh3, sh3, te, option );
  
  canvas->cd(6);
  rc3->SetLineColor(2);
  sc3->SetLineColor(4);
  sc3->SetLineStyle(2);
  setStats(rc3,sc3, startingY, startingX, fit);
  rc3->Draw();
  sc3->Draw("sames");
  myPV->PVCompute(rc3, sc3, te, option );
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

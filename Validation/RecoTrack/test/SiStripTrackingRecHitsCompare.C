void SetUpHistograms(TH1F* h1, TH1F* h2)
  //, const char* xtitle, TLegend* leg = 0)
{
  float scale1 = -9999.9;
  float scale2 = -9999.9;

  if ( h1->Integral() != 0 && h2->Integral() != 0 )
    {
      scale1 = 1.0/(float)h1->Integral();
      scale2 = 1.0/(float)h2->Integral();
      
      h1->Sumw2();
      h2->Sumw2();
      h1->Scale(scale1);
      h2->Scale(scale2);
  
      h1->SetLineWidth(1);
      h2->SetLineWidth(1);
      h1->SetLineColor(2);
      h2->SetLineColor(4);
      h2->SetLineStyle(2);  
    }
  /*
  h1->SetXTitle(xtitle);
  if ( leg != 0 )
    {
      leg->SetBorderSize(0);
      leg->AddEntry(h1, "reference  ", "l");
      leg->AddEntry(h2, "new release", "l");
    }
  */
}

void SiStripTrackingRecHitsCompare()
{
  //color 2 = red  = rfile = new file
  //color 4 = blue = sfile = reference file


 gROOT ->Reset();

 char*  rfilename = "striptrackingrechitshisto.root";
 //char*  sfilename = "striptrackingrechitshisto.root";
 char*  sfilename = "../data/striptrackingrechitshisto_REF.root";

 delete gROOT->GetListOfFiles()->FindObject(rfilename);
 delete gROOT->GetListOfFiles()->FindObject(sfilename); 

 TText* te = new TText();
 TFile * rfile = new TFile(rfilename);
 TFile * sfile = new TFile(sfilename);

 rfile->cd("DQMData/TrackingRecHits/Strip");
 sfile->cd("DQMData/TrackingRecHits/Strip");

 Char_t histo[200];

 gROOT->ProcessLine(".x HistoCompare_Strips.C");
 HistoCompare_Strips * myPV = new HistoCompare_Strips();

 TCanvas *Strip;


 //=============================================================== 
 // TIB

 TH1F* refplotsTIB[6];
 TH1F* newplotsTIB[6];
 
 TProfile* PullTrackangleProfiletib[6];
 TProfile* PullTrackwidthProfiletib[6];
 TProfile* PullTrackwidthProfileCategory1tib[6];
 TProfile* PullTrackwidthProfileCategory2tib[6];
 TProfile* PullTrackwidthProfileCategory3tib[6];
 TProfile* PullTrackwidthProfileCategory4tib[6];
 TH1F* matchedtib[16];

 TProfile* newPullTrackangleProfiletib[6];
 TProfile* newPullTrackwidthProfiletib[6];
 TProfile* newPullTrackwidthProfileCategory1tib[6];
 TProfile* newPullTrackwidthProfileCategory2tib[6];
 TProfile* newPullTrackwidthProfileCategory3tib[6];
 TProfile* newPullTrackwidthProfileCategory4tib[6];
 TH1F* newmatchedtib[16];

 //
 
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Adc_rphi_layer1tib",refplotsTIB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Adc_rphi_layer2tib",refplotsTIB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Adc_rphi_layer3tib",refplotsTIB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Adc_rphi_layer4tib",refplotsTIB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Adc_sas_layer1tib",refplotsTIB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Adc_sas_layer2tib",refplotsTIB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Adc_rphi_layer1tib",newplotsTIB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Adc_rphi_layer2tib",newplotsTIB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Adc_rphi_layer3tib",newplotsTIB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Adc_rphi_layer4tib",newplotsTIB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Adc_sas_layer1tib",newplotsTIB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Adc_sas_layer2tib",newplotsTIB[5]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   if (refplotsTIB[i]->GetEntries() == 0 || newplotsTIB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTIB[i],newplotsTIB[i]);
   refplotsTIB[i]->Draw();
   newplotsTIB[i]->Draw("sames");
   myPV->PVCompute(refplotsTIB[i] , newplotsTIB[i] , te );
 }

 Strip->Print("AdcTIBCompare.eps");
 
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_LF_rphi_layer1tib",refplotsTIB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_LF_rphi_layer2tib",refplotsTIB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_LF_rphi_layer3tib",refplotsTIB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_LF_rphi_layer4tib",refplotsTIB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_LF_sas_layer1tib",refplotsTIB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_LF_sas_layer2tib",refplotsTIB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_LF_rphi_layer1tib",newplotsTIB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_LF_rphi_layer2tib",newplotsTIB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_LF_rphi_layer3tib",newplotsTIB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_LF_rphi_layer4tib",newplotsTIB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_LF_sas_layer1tib",newplotsTIB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_LF_sas_layer2tib",newplotsTIB[5]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   if (refplotsTIB[i]->GetEntries() == 0 || newplotsTIB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTIB[i],newplotsTIB[i]);
   refplotsTIB[i]->Draw();
   newplotsTIB[i]->Draw("sames");
   myPV->PVCompute(refplotsTIB[i] , newplotsTIB[i] , te );
 }
 
 Strip->Print("PullLFTIBCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_MF_rphi_layer1tib",refplotsTIB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_MF_rphi_layer2tib",refplotsTIB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_MF_rphi_layer3tib",refplotsTIB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_MF_rphi_layer4tib",refplotsTIB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_MF_sas_layer1tib",refplotsTIB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_MF_sas_layer2tib",refplotsTIB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_MF_rphi_layer1tib",newplotsTIB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_MF_rphi_layer2tib",newplotsTIB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_MF_rphi_layer3tib",newplotsTIB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_MF_rphi_layer4tib",newplotsTIB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_MF_sas_layer1tib",newplotsTIB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pull_MF_sas_layer2tib",newplotsTIB[5]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   if (refplotsTIB[i]->GetEntries() == 0 || newplotsTIB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTIB[i],newplotsTIB[i]);
   refplotsTIB[i]->Draw();
   newplotsTIB[i]->Draw("sames");
   myPV->PVCompute(refplotsTIB[i] , newplotsTIB[i] , te );
 }
 
 Strip->Print("PullMFTIBCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackangle_rphi_layer1tib",refplotsTIB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackangle_rphi_layer2tib",refplotsTIB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackangle_rphi_layer3tib",refplotsTIB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackangle_rphi_layer4tib",refplotsTIB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackangle_sas_layer1tib",refplotsTIB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackangle_sas_layer2tib",refplotsTIB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackangle_rphi_layer1tib",newplotsTIB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackangle_rphi_layer2tib",newplotsTIB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackangle_rphi_layer3tib",newplotsTIB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackangle_rphi_layer4tib",newplotsTIB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackangle_sas_layer1tib",newplotsTIB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackangle_sas_layer2tib",newplotsTIB[5]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   if (refplotsTIB[i]->GetEntries() == 0 || newplotsTIB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTIB[i],newplotsTIB[i]);
   refplotsTIB[i]->Draw();
   newplotsTIB[i]->Draw("sames");
   myPV->PVCompute(refplotsTIB[i] , newplotsTIB[i] , te );
 }
 
 Strip->Print("TrackangleTIBCompare.eps");
 
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackwidth_rphi_layer1tib",refplotsTIB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackwidth_rphi_layer2tib",refplotsTIB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackwidth_rphi_layer3tib",refplotsTIB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackwidth_rphi_layer4tib",refplotsTIB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackwidth_sas_layer1tib",refplotsTIB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackwidth_sas_layer2tib",refplotsTIB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackwidth_rphi_layer1tib",newplotsTIB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackwidth_rphi_layer2tib",newplotsTIB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackwidth_rphi_layer3tib",newplotsTIB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackwidth_rphi_layer4tib",newplotsTIB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackwidth_sas_layer1tib",newplotsTIB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Trackwidth_sas_layer2tib",newplotsTIB[5]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   if (refplotsTIB[i]->GetEntries() == 0 || newplotsTIB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTIB[i],newplotsTIB[i]);
   refplotsTIB[i]->Draw();
   newplotsTIB[i]->Draw("sames");
   myPV->PVCompute(refplotsTIB[i] , newplotsTIB[i] , te );
 }
 
 Strip->Print("TrackwidthTIBCompare.eps");
 
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Expectedwidth_rphi_layer1tib",refplotsTIB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Expectedwidth_rphi_layer2tib",refplotsTIB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Expectedwidth_rphi_layer3tib",refplotsTIB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Expectedwidth_rphi_layer4tib",refplotsTIB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Expectedwidth_sas_layer1tib",refplotsTIB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Expectedwidth_sas_layer2tib",refplotsTIB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Expectedwidth_rphi_layer1tib",newplotsTIB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Expectedwidth_rphi_layer2tib",newplotsTIB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Expectedwidth_rphi_layer3tib",newplotsTIB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Expectedwidth_rphi_layer4tib",newplotsTIB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Expectedwidth_sas_layer1tib",newplotsTIB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Expectedwidth_sas_layer2tib",newplotsTIB[5]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   if (refplotsTIB[i]->GetEntries() == 0 || newplotsTIB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTIB[i],newplotsTIB[i]);
   refplotsTIB[i]->Draw();
   newplotsTIB[i]->Draw("sames");
   myPV->PVCompute(refplotsTIB[i] , newplotsTIB[i] , te );
 }
 
 Strip->Print("ExpectedwidthTIBCompare.eps");
 
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Category_rphi_layer1tib",refplotsTIB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Category_rphi_layer2tib",refplotsTIB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Category_rphi_layer3tib",refplotsTIB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Category_rphi_layer4tib",refplotsTIB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Category_sas_layer1tib",refplotsTIB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Category_sas_layer2tib",refplotsTIB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Category_rphi_layer1tib",newplotsTIB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Category_rphi_layer2tib",newplotsTIB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Category_rphi_layer3tib",newplotsTIB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Category_rphi_layer4tib",newplotsTIB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Category_sas_layer1tib",newplotsTIB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Category_sas_layer2tib",newplotsTIB[5]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   if (refplotsTIB[i]->GetEntries() == 0 || newplotsTIB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTIB[i],newplotsTIB[i]);
   refplotsTIB[i]->Draw();
   newplotsTIB[i]->Draw("sames");
   myPV->PVCompute(refplotsTIB[i] , newplotsTIB[i] , te );
 }
 
 Strip->Print("CategoryTIBCompare.eps");
 
 /*
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackangleProfile_rphi_layer1tib",PullTrackangleProfiletib[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackangleProfile_rphi_layer2tib",PullTrackangleProfiletib[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackangleProfile_rphi_layer3tib",PullTrackangleProfiletib[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackangleProfile_rphi_layer4tib",PullTrackangleProfiletib[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackangleProfile_sas_layer1tib",PullTrackangleProfiletib[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackangleProfile_sas_layer2tib",PullTrackangleProfiletib[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackangleProfile_rphi_layer1tib",newPullTrackangleProfiletib[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackangleProfile_rphi_layer2tib",newPullTrackangleProfiletib[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackangleProfile_rphi_layer3tib",newPullTrackangleProfiletib[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackangleProfile_rphi_layer4tib",newPullTrackangleProfiletib[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackangleProfile_sas_layer1tib",newPullTrackangleProfiletib[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackangleProfile_sas_layer2tib",newPullTrackangleProfiletib[5]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   Strip->cd(i+1);
   PullTrackangleProfiletib[i]->SetLineColor(2);
   newPullTrackangleProfiletib[i]->SetLineColor(4);
   newPullTrackangleProfiletib[i]->SetLineStyle(2);
   PullTrackangleProfiletib[i]->Draw();
   newPullTrackangleProfiletib[i]->Draw("sames");
   //    myPV->PVCompute(PullTrackangleProfiletib[i] , newPullTrackangleProfiletib[i] , te );
 }
 
 Strip->Print("PullTrackangleProfileTIBCompare.eps");


 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_rphi_layer1tib",PullTrackwidthProfiletib[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_rphi_layer2tib",PullTrackwidthProfiletib[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_rphi_layer3tib",PullTrackwidthProfiletib[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_rphi_layer4tib",PullTrackwidthProfiletib[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_sas_layer1tib",PullTrackwidthProfiletib[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_sas_layer2tib",PullTrackwidthProfiletib[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_rphi_layer1tib",newPullTrackwidthProfiletib[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_rphi_layer2tib",newPullTrackwidthProfiletib[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_rphi_layer3tib",newPullTrackwidthProfiletib[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_rphi_layer4tib",newPullTrackwidthProfiletib[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_sas_layer1tib",newPullTrackwidthProfiletib[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_sas_layer2tib",newPullTrackwidthProfiletib[5]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   Strip->cd(i+1);
   PullTrackwidthProfiletib[i]->SetLineColor(2);
   newPullTrackwidthProfiletib[i]->SetLineColor(4);
   newPullTrackwidthProfiletib[i]->SetLineStyle(2);
   PullTrackwidthProfiletib[i]->Draw();
   newPullTrackwidthProfiletib[i]->Draw("sames");
   //    myPV->PVCompute(PullTrackwidthProfiletib[i] , newPullTrackwidthProfiletib[i] , te );
 }
 
 Strip->Print("PullTrackwidthProfileTIBCompare.eps");


 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category1_rphi_layer1tib",PullTrackwidthProfileCategory1tib[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category1_rphi_layer2tib",PullTrackwidthProfileCategory1tib[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category1_rphi_layer3tib",PullTrackwidthProfileCategory1tib[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category1_rphi_layer4tib",PullTrackwidthProfileCategory1tib[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category1_sas_layer1tib",PullTrackwidthProfileCategory1tib[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category1_sas_layer2tib",PullTrackwidthProfileCategory1tib[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category1_rphi_layer1tib",newPullTrackwidthProfileCategory1tib[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category1_rphi_layer2tib",newPullTrackwidthProfileCategory1tib[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category1_rphi_layer3tib",newPullTrackwidthProfileCategory1tib[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category1_rphi_layer4tib",newPullTrackwidthProfileCategory1tib[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category1_sas_layer1tib",newPullTrackwidthProfileCategory1tib[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category1_sas_layer2tib",newPullTrackwidthProfileCategory1tib[5]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   Strip->cd(i+1);
   PullTrackwidthProfileCategory1tib[i]->SetLineColor(2);
   newPullTrackwidthProfileCategory1tib[i]->SetLineColor(4);
   newPullTrackwidthProfileCategory1tib[i]->SetLineStyle(2);
   PullTrackwidthProfileCategory1tib[i]->Draw();
   newPullTrackwidthProfileCategory1tib[i]->Draw("sames");
   //    myPV->PVCompute(PullTrackwidthProfileCategory1tib[i] , newPullTrackwidthProfileCategory1tib[i] , te );
 }
 
 Strip->Print("PullTrackwidthProfileCategory1TIBCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category2_rphi_layer1tib",PullTrackwidthProfileCategory2tib[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category2_rphi_layer2tib",PullTrackwidthProfileCategory2tib[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category2_rphi_layer3tib",PullTrackwidthProfileCategory2tib[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category2_rphi_layer4tib",PullTrackwidthProfileCategory2tib[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category2_sas_layer1tib",PullTrackwidthProfileCategory2tib[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category2_sas_layer2tib",PullTrackwidthProfileCategory2tib[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category2_rphi_layer1tib",newPullTrackwidthProfileCategory2tib[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category2_rphi_layer2tib",newPullTrackwidthProfileCategory2tib[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category2_rphi_layer3tib",newPullTrackwidthProfileCategory2tib[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category2_rphi_layer4tib",newPullTrackwidthProfileCategory2tib[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category2_sas_layer1tib",newPullTrackwidthProfileCategory2tib[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category2_sas_layer2tib",newPullTrackwidthProfileCategory2tib[5]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   Strip->cd(i+1);
   PullTrackwidthProfileCategory2tib[i]->SetLineColor(2);
   newPullTrackwidthProfileCategory2tib[i]->SetLineColor(4);
   newPullTrackwidthProfileCategory2tib[i]->SetLineStyle(2);
   PullTrackwidthProfileCategory2tib[i]->Draw();
   newPullTrackwidthProfileCategory2tib[i]->Draw("sames");
   //    myPV->PVCompute(PullTrackwidthProfileCategory2tib[i] , newPullTrackwidthProfileCategory2tib[i] , te );
 }
 
 Strip->Print("PullTrackwidthProfileCategory2TIBCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category3_rphi_layer1tib",PullTrackwidthProfileCategory3tib[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category3_rphi_layer2tib",PullTrackwidthProfileCategory3tib[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category3_rphi_layer3tib",PullTrackwidthProfileCategory3tib[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category3_rphi_layer4tib",PullTrackwidthProfileCategory3tib[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category3_sas_layer1tib",PullTrackwidthProfileCategory3tib[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category3_sas_layer2tib",PullTrackwidthProfileCategory3tib[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category3_rphi_layer1tib",newPullTrackwidthProfileCategory3tib[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category3_rphi_layer2tib",newPullTrackwidthProfileCategory3tib[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category3_rphi_layer3tib",newPullTrackwidthProfileCategory3tib[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category3_rphi_layer4tib",newPullTrackwidthProfileCategory3tib[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category3_sas_layer1tib",newPullTrackwidthProfileCategory3tib[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category3_sas_layer2tib",newPullTrackwidthProfileCategory3tib[5]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   Strip->cd(i+1);
   PullTrackwidthProfileCategory3tib[i]->SetLineColor(2);
   newPullTrackwidthProfileCategory3tib[i]->SetLineColor(4);
   newPullTrackwidthProfileCategory3tib[i]->SetLineStyle(2);
   PullTrackwidthProfileCategory3tib[i]->Draw();
   newPullTrackwidthProfileCategory3tib[i]->Draw("sames");
   //    myPV->PVCompute(PullTrackwidthProfileCategory3tib[i] , newPullTrackwidthProfileCategory3tib[i] , te );
 }
 
 Strip->Print("PullTrackwidthProfileCategory3TIBCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category4_rphi_layer1tib",PullTrackwidthProfileCategory4tib[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category4_rphi_layer2tib",PullTrackwidthProfileCategory4tib[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category4_rphi_layer3tib",PullTrackwidthProfileCategory4tib[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category4_rphi_layer4tib",PullTrackwidthProfileCategory4tib[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category4_sas_layer1tib",PullTrackwidthProfileCategory4tib[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category4_sas_layer2tib",PullTrackwidthProfileCategory4tib[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category4_rphi_layer1tib",newPullTrackwidthProfileCategory4tib[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category4_rphi_layer2tib",newPullTrackwidthProfileCategory4tib[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category4_rphi_layer3tib",newPullTrackwidthProfileCategory4tib[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category4_rphi_layer4tib",newPullTrackwidthProfileCategory4tib[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category4_sas_layer1tib",newPullTrackwidthProfileCategory4tib[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/PullTrackwidthProfile_Category4_sas_layer2tib",newPullTrackwidthProfileCategory4tib[5]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   Strip->cd(i+1);
   PullTrackwidthProfileCategory4tib[i]->SetLineColor(2);
   newPullTrackwidthProfileCategory4tib[i]->SetLineColor(4);
   newPullTrackwidthProfileCategory4tib[i]->SetLineStyle(2);
   PullTrackwidthProfileCategory4tib[i]->Draw();
   newPullTrackwidthProfileCategory4tib[i]->Draw("sames");
   //    myPV->PVCompute(PullTrackwidthProfileCategory4tib[i] , newPullTrackwidthProfileCategory4tib[i] , te );
 }
 
 Strip->Print("PullTrackwidthProfileCategory4TIBCompare.eps");
 */

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Nstp_rphi_layer1tib",refplotsTIB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Nstp_rphi_layer2tib",refplotsTIB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Nstp_rphi_layer3tib",refplotsTIB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Nstp_rphi_layer4tib",refplotsTIB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Nstp_sas_layer1tib",refplotsTIB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Nstp_sas_layer2tib",refplotsTIB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Nstp_rphi_layer1tib",newplotsTIB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Nstp_rphi_layer2tib",newplotsTIB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Nstp_rphi_layer3tib",newplotsTIB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Nstp_rphi_layer4tib",newplotsTIB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Nstp_sas_layer1tib",newplotsTIB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Nstp_sas_layer2tib",newplotsTIB[5]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   if (refplotsTIB[i]->GetEntries() == 0 || newplotsTIB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTIB[i],newplotsTIB[i]);
   refplotsTIB[i]->Draw();
   newplotsTIB[i]->Draw("sames");
   myPV->PVCompute(refplotsTIB[i] , newplotsTIB[i] , te );
 }
 
 Strip->Print("NstpTIBCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posx_rphi_layer1tib",refplotsTIB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posx_rphi_layer2tib",refplotsTIB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posx_rphi_layer3tib",refplotsTIB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posx_rphi_layer4tib",refplotsTIB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posx_sas_layer1tib",refplotsTIB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posx_sas_layer2tib",refplotsTIB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posx_rphi_layer1tib",newplotsTIB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posx_rphi_layer2tib",newplotsTIB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posx_rphi_layer3tib",newplotsTIB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posx_rphi_layer4tib",newplotsTIB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posx_sas_layer1tib",newplotsTIB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posx_sas_layer2tib",newplotsTIB[5]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   if (refplotsTIB[i]->GetEntries() == 0 || newplotsTIB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTIB[i],newplotsTIB[i]);
   refplotsTIB[i]->Draw();
   newplotsTIB[i]->Draw("sames");
   myPV->PVCompute(refplotsTIB[i] , newplotsTIB[i] , te );
 }
  
 Strip->Print("PosTIBCompare.eps");
  

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_LF_rphi_layer1tib",refplotsTIB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_LF_rphi_layer2tib",refplotsTIB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_LF_rphi_layer3tib",refplotsTIB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_LF_rphi_layer4tib",refplotsTIB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_LF_sas_layer1tib",refplotsTIB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_LF_sas_layer2tib",refplotsTIB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_LF_rphi_layer1tib",newplotsTIB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_LF_rphi_layer2tib",newplotsTIB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_LF_rphi_layer3tib",newplotsTIB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_LF_rphi_layer4tib",newplotsTIB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_LF_sas_layer1tib",newplotsTIB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_LF_sas_layer2tib",newplotsTIB[5]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   if (refplotsTIB[i]->GetEntries() == 0 || newplotsTIB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTIB[i],newplotsTIB[i]);
   refplotsTIB[i]->Draw();
   newplotsTIB[i]->Draw("sames");
   myPV->PVCompute(refplotsTIB[i] , newplotsTIB[i] , te );
 }
  
 Strip->Print("ErrxLFTIBCompare.eps");
  
 
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_MF_rphi_layer1tib",refplotsTIB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_MF_rphi_layer2tib",refplotsTIB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_MF_rphi_layer3tib",refplotsTIB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_MF_rphi_layer4tib",refplotsTIB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_MF_sas_layer1tib",refplotsTIB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_MF_sas_layer2tib",refplotsTIB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_MF_rphi_layer1tib",newplotsTIB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_MF_rphi_layer2tib",newplotsTIB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_MF_rphi_layer3tib",newplotsTIB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_MF_rphi_layer4tib",newplotsTIB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_MF_sas_layer1tib",newplotsTIB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_MF_sas_layer2tib",newplotsTIB[5]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   if (refplotsTIB[i]->GetEntries() == 0 || newplotsTIB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTIB[i],newplotsTIB[i]);
   refplotsTIB[i]->Draw();
   newplotsTIB[i]->Draw("sames");
   myPV->PVCompute(refplotsTIB[i] , newplotsTIB[i] , te );
 }
 Strip->Print("ErrxMFTIBCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_LF_rphi_layer1tib",refplotsTIB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_LF_rphi_layer2tib",refplotsTIB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_LF_rphi_layer3tib",refplotsTIB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_LF_rphi_layer4tib",refplotsTIB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_LF_sas_layer1tib",refplotsTIB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_LF_sas_layer2tib",refplotsTIB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_LF_rphi_layer1tib",newplotsTIB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_LF_rphi_layer2tib",newplotsTIB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_LF_rphi_layer3tib",newplotsTIB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_LF_rphi_layer4tib",newplotsTIB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_LF_sas_layer1tib",newplotsTIB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_LF_sas_layer2tib",newplotsTIB[5]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   if (refplotsTIB[i]->GetEntries() == 0 || newplotsTIB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTIB[i],newplotsTIB[i]);
   refplotsTIB[i]->Draw();
   newplotsTIB[i]->Draw("sames");
   myPV->PVCompute(refplotsTIB[i] , newplotsTIB[i] , te );
 }
  
 Strip->Print("ResLFTIBCompare.eps");


 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_MF_rphi_layer1tib",refplotsTIB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_MF_rphi_layer2tib",refplotsTIB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_MF_rphi_layer3tib",refplotsTIB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_MF_rphi_layer4tib",refplotsTIB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_MF_sas_layer1tib",refplotsTIB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_MF_sas_layer2tib",refplotsTIB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_MF_rphi_layer1tib",newplotsTIB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_MF_rphi_layer2tib",newplotsTIB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_MF_rphi_layer3tib",newplotsTIB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_MF_rphi_layer4tib",newplotsTIB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_MF_sas_layer1tib",newplotsTIB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Res_MF_sas_layer2tib",newplotsTIB[5]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   if (refplotsTIB[i]->GetEntries() == 0 || newplotsTIB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTIB[i],newplotsTIB[i]);
   refplotsTIB[i]->Draw();
   newplotsTIB[i]->Draw("sames");
   myPV->PVCompute(refplotsTIB[i] , newplotsTIB[i] , te );
 }
  
 Strip->Print("ResMFTIBCompare.eps");


  /*
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Chi2_rphi_layer1tib",chi2tib[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Chi2_rphi_layer2tib",chi2tib[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Chi2_rphi_layer3tib",chi2tib[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Chi2_rphi_layer4tib",chi2tib[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Chi2_sas_layer1tib",chi2tib[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Chi2_sas_layer2tib",chi2tib[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Chi2_rphi_layer1tib",newchi2tib[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Chi2_rphi_layer2tib",newchi2tib[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Chi2_rphi_layer3tib",newchi2tib[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Chi2_rphi_layer4tib",newchi2tib[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Chi2_sas_layer1tib",newchi2tib[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Chi2_sas_layer2tib",newchi2tib[5]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<6; i++) {
    Strip->cd(i+1);
    chi2tib[i]->SetLineColor(2);
   refplotsTIB[i]->Add( refplotsTIB[i],refplotsTIB[i], 1/(refplotsTIB[i]->GetEntries()),0.);
    newchi2tib[i]->SetLineColor(4);
   newplotsTIB[i]->Add( newplotsTIB[i],newplotsTIB[i], 1/(newplotsTIB[i]->GetEntries()),0.);
    newchi2tib[i]->SetLineStyle(2);
    chi2tib[i]->Draw();
    newchi2tib[i]->Draw("sames");
    myPV->PVCompute(chi2tib[i] , newchi2tib[i] , te );
  }
  
  Strip->Print("Chi2TIBCompare.eps");
  */
  
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posx_matched_layer1tib",matchedtib[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posy_matched_layer1tib",matchedtib[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posx_matched_layer2tib",matchedtib[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posy_matched_layer2tib",matchedtib[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_matched_layer1tib",matchedtib[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Erry_matched_layer1tib",matchedtib[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_matched_layer2tib",matchedtib[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Erry_matched_layer2tib",matchedtib[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Resx_matched_layer1tib",matchedtib[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Resy_matched_layer1tib",matchedtib[9]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Resx_matched_layer2tib",matchedtib[10]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Resy_matched_layer2tib",matchedtib[11]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pullx_matched_layer1tib",matchedtib[12]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pully_matched_layer1tib",matchedtib[13]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pullx_matched_layer2tib",matchedtib[14]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pully_matched_layer2tib",matchedtib[15]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posx_matched_layer1tib",newmatchedtib[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posy_matched_layer1tib",newmatchedtib[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posx_matched_layer2tib",newmatchedtib[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Posy_matched_layer2tib",newmatchedtib[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_matched_layer1tib",newmatchedtib[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Erry_matched_layer1tib",newmatchedtib[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Errx_matched_layer2tib",newmatchedtib[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Erry_matched_layer2tib",newmatchedtib[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Resx_matched_layer1tib",newmatchedtib[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Resy_matched_layer1tib",newmatchedtib[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Resx_matched_layer2tib",newmatchedtib[10]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Resy_matched_layer2tib",newmatchedtib[11]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pullx_matched_layer1tib",newmatchedtib[12]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pully_matched_layer1tib",newmatchedtib[13]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pullx_matched_layer2tib",newmatchedtib[14]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TIB/Pully_matched_layer2tib",newmatchedtib[15]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,4);
 for (Int_t i=0; i<16; i++) {
   if (matchedtib[i]->GetEntries() == 0 || newmatchedtib[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(matchedtib[i],newmatchedtib[i]);
   matchedtib[i]->Draw();
   newmatchedtib[i]->Draw("sames");
   myPV->PVCompute(matchedtib[i] , newmatchedtib[i] , te );
 }
 
 Strip->Print("MatchedTIBCompare.eps");

 
 //======================================================================================================
// TOB

 TH1F* refplotsTOB[8];
 TH1F* newplotsTOB[8];

 TProfile* PullTrackangleProfiletob[8];
 TProfile* PullTrackwidthProfiletob[8];
 TProfile* PullTrackwidthProfileCategory1tob[8];
 TProfile* PullTrackwidthProfileCategory2tob[8];
 TProfile* PullTrackwidthProfileCategory3tob[8];
 TProfile* PullTrackwidthProfileCategory4tob[8];
 TH1F* matchedtob[16];
 TProfile* newPullTrackangleProfiletob[8];
 TProfile* newPullTrackwidthProfiletob[8];
 TProfile* newPullTrackwidthProfileCategory1tob[8];
 TProfile* newPullTrackwidthProfileCategory2tob[8];
 TProfile* newPullTrackwidthProfileCategory3tob[8];
 TProfile* newPullTrackwidthProfileCategory4tob[8];
 TH1F* newmatchedtob[16];
 
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Adc_rphi_layer1tob",refplotsTOB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Adc_rphi_layer2tob",refplotsTOB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Adc_rphi_layer3tob",refplotsTOB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Adc_rphi_layer4tob",refplotsTOB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Adc_rphi_layer5tob",refplotsTOB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Adc_rphi_layer6tob",refplotsTOB[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Adc_sas_layer1tob",refplotsTOB[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Adc_sas_layer2tob",refplotsTOB[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Adc_rphi_layer1tob",newplotsTOB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Adc_rphi_layer2tob",newplotsTOB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Adc_rphi_layer3tob",newplotsTOB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Adc_rphi_layer4tob",newplotsTOB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Adc_rphi_layer5tob",newplotsTOB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Adc_rphi_layer6tob",newplotsTOB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Adc_sas_layer1tob",newplotsTOB[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Adc_sas_layer2tob",newplotsTOB[7]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   if (refplotsTOB[i]->GetEntries() == 0 || newplotsTOB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTOB[i],newplotsTOB[i]);
   refplotsTOB[i]->Draw();
   newplotsTOB[i]->Draw("sames");
   myPV->PVCompute(refplotsTOB[i] , newplotsTOB[i] , te );
 }
 
 Strip->Print("AdcTOBCompare.eps");
 
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_LF_rphi_layer1tob",refplotsTOB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_LF_rphi_layer2tob",refplotsTOB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_LF_rphi_layer3tob",refplotsTOB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_LF_rphi_layer4tob",refplotsTOB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_LF_rphi_layer5tob",refplotsTOB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_LF_rphi_layer6tob",refplotsTOB[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_LF_sas_layer1tob",refplotsTOB[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_LF_sas_layer2tob",refplotsTOB[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_LF_rphi_layer1tob",newplotsTOB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_LF_rphi_layer2tob",newplotsTOB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_LF_rphi_layer3tob",newplotsTOB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_LF_rphi_layer4tob",newplotsTOB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_LF_rphi_layer5tob",newplotsTOB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_LF_rphi_layer6tob",newplotsTOB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_LF_sas_layer1tob",newplotsTOB[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_LF_sas_layer2tob",newplotsTOB[7]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   if (refplotsTOB[i]->GetEntries() == 0 || newplotsTOB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTOB[i],newplotsTOB[i]);
   refplotsTOB[i]->Draw();
   newplotsTOB[i]->Draw("sames");
   myPV->PVCompute(refplotsTOB[i] , newplotsTOB[i] , te );
 }
 
 Strip->Print("PullLFTOBCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_MF_rphi_layer1tob",refplotsTOB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_MF_rphi_layer2tob",refplotsTOB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_MF_rphi_layer3tob",refplotsTOB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_MF_rphi_layer4tob",refplotsTOB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_MF_rphi_layer5tob",refplotsTOB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_MF_rphi_layer6tob",refplotsTOB[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_MF_sas_layer1tob",refplotsTOB[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_MF_sas_layer2tob",refplotsTOB[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_MF_rphi_layer1tob",newplotsTOB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_MF_rphi_layer2tob",newplotsTOB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_MF_rphi_layer3tob",newplotsTOB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_MF_rphi_layer4tob",newplotsTOB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_MF_rphi_layer5tob",newplotsTOB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_MF_rphi_layer6tob",newplotsTOB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_MF_sas_layer1tob",newplotsTOB[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pull_MF_sas_layer2tob",newplotsTOB[7]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   if (refplotsTOB[i]->GetEntries() == 0 || newplotsTOB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTOB[i],newplotsTOB[i]);
   refplotsTOB[i]->Draw();
   newplotsTOB[i]->Draw("sames");
   myPV->PVCompute(refplotsTOB[i] , newplotsTOB[i] , te );
 }
 
 Strip->Print("PullMFTOBCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackangle_rphi_layer1tob",refplotsTOB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackangle_rphi_layer2tob",refplotsTOB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackangle_rphi_layer3tob",refplotsTOB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackangle_rphi_layer4tob",refplotsTOB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackangle_rphi_layer5tob",refplotsTOB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackangle_rphi_layer6tob",refplotsTOB[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackangle_sas_layer1tob",refplotsTOB[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackangle_sas_layer2tob",refplotsTOB[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackangle_rphi_layer1tob",newplotsTOB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackangle_rphi_layer2tob",newplotsTOB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackangle_rphi_layer3tob",newplotsTOB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackangle_rphi_layer4tob",newplotsTOB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackangle_rphi_layer5tob",newplotsTOB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackangle_rphi_layer6tob",newplotsTOB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackangle_sas_layer1tob",newplotsTOB[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackangle_sas_layer2tob",newplotsTOB[7]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   if (refplotsTOB[i]->GetEntries() == 0 || newplotsTOB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTOB[i],newplotsTOB[i]);
   refplotsTOB[i]->Draw();
   newplotsTOB[i]->Draw("sames");
   myPV->PVCompute(refplotsTOB[i] , newplotsTOB[i] , te );
 }
 
 Strip->Print("TrackangleTOBCompare.eps");


 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackwidth_rphi_layer1tob",refplotsTOB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackwidth_rphi_layer2tob",refplotsTOB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackwidth_rphi_layer3tob",refplotsTOB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackwidth_rphi_layer4tob",refplotsTOB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackwidth_rphi_layer5tob",refplotsTOB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackwidth_rphi_layer6tob",refplotsTOB[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackwidth_sas_layer1tob",refplotsTOB[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackwidth_sas_layer2tob",refplotsTOB[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackwidth_rphi_layer1tob",newplotsTOB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackwidth_rphi_layer2tob",newplotsTOB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackwidth_rphi_layer3tob",newplotsTOB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackwidth_rphi_layer4tob",newplotsTOB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackwidth_rphi_layer5tob",newplotsTOB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackwidth_rphi_layer6tob",newplotsTOB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackwidth_sas_layer1tob",newplotsTOB[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Trackwidth_sas_layer2tob",newplotsTOB[7]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   if (refplotsTOB[i]->GetEntries() == 0 || newplotsTOB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTOB[i],newplotsTOB[i]);
   refplotsTOB[i]->Draw();
   newplotsTOB[i]->Draw("sames");
   myPV->PVCompute(refplotsTOB[i] , newplotsTOB[i] , te );
 }
 
 Strip->Print("TrackwidthTOBCompare.eps");


 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Expectedwidth_rphi_layer1tob",refplotsTOB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Expectedwidth_rphi_layer2tob",refplotsTOB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Expectedwidth_rphi_layer3tob",refplotsTOB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Expectedwidth_rphi_layer4tob",refplotsTOB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Expectedwidth_rphi_layer5tob",refplotsTOB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Expectedwidth_rphi_layer6tob",refplotsTOB[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Expectedwidth_sas_layer1tob",refplotsTOB[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Expectedwidth_sas_layer2tob",refplotsTOB[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Expectedwidth_rphi_layer1tob",newplotsTOB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Expectedwidth_rphi_layer2tob",newplotsTOB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Expectedwidth_rphi_layer3tob",newplotsTOB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Expectedwidth_rphi_layer4tob",newplotsTOB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Expectedwidth_rphi_layer5tob",newplotsTOB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Expectedwidth_rphi_layer6tob",newplotsTOB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Expectedwidth_sas_layer1tob",newplotsTOB[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Expectedwidth_sas_layer2tob",newplotsTOB[7]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   if (refplotsTOB[i]->GetEntries() == 0 || newplotsTOB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTOB[i],newplotsTOB[i]);
   refplotsTOB[i]->Draw();
   newplotsTOB[i]->Draw("sames");
   myPV->PVCompute(refplotsTOB[i] , newplotsTOB[i] , te );
 }
 
 Strip->Print("ExpectedwidthTOBCompare.eps");


 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Category_rphi_layer1tob",refplotsTOB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Category_rphi_layer2tob",refplotsTOB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Category_rphi_layer3tob",refplotsTOB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Category_rphi_layer4tob",refplotsTOB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Category_rphi_layer5tob",refplotsTOB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Category_rphi_layer6tob",refplotsTOB[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Category_sas_layer1tob",refplotsTOB[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Category_sas_layer2tob",refplotsTOB[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Category_rphi_layer1tob",newplotsTOB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Category_rphi_layer2tob",newplotsTOB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Category_rphi_layer3tob",newplotsTOB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Category_rphi_layer4tob",newplotsTOB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Category_rphi_layer5tob",newplotsTOB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Category_rphi_layer6tob",newplotsTOB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Category_sas_layer1tob",newplotsTOB[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Category_sas_layer2tob",newplotsTOB[7]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   if (refplotsTOB[i]->GetEntries() == 0 || newplotsTOB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTOB[i],newplotsTOB[i]);
   refplotsTOB[i]->Draw();
   newplotsTOB[i]->Draw("sames");
   myPV->PVCompute(refplotsTOB[i] , newplotsTOB[i] , te );
 }
 
 Strip->Print("CategoryTOBCompare.eps");


 /*
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackangleProfile_rphi_layer1tob",PullTrackangleProfiletob[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackangleProfile_rphi_layer2tob",PullTrackangleProfiletob[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackangleProfile_rphi_layer3tob",PullTrackangleProfiletob[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackangleProfile_rphi_layer4tob",PullTrackangleProfiletob[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackangleProfile_rphi_layer5tob",PullTrackangleProfiletob[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackangleProfile_rphi_layer6tob",PullTrackangleProfiletob[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackangleProfile_sas_layer1tob",PullTrackangleProfiletob[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackangleProfile_sas_layer2tob",PullTrackangleProfiletob[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackangleProfile_rphi_layer1tob",newPullTrackangleProfiletob[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackangleProfile_rphi_layer2tob",newPullTrackangleProfiletob[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackangleProfile_rphi_layer3tob",newPullTrackangleProfiletob[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackangleProfile_rphi_layer4tob",newPullTrackangleProfiletob[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackangleProfile_rphi_layer5tob",newPullTrackangleProfiletob[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackangleProfile_rphi_layer6tob",newPullTrackangleProfiletob[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackangleProfile_sas_layer1tob",newPullTrackangleProfiletob[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackangleProfile_sas_layer2tob",newPullTrackangleProfiletob[7]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   if (refplotsTOB[i]->GetEntries() == 0 || newplotsTOB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   PullTrackangleProfiletob[i]->SetLineColor(2);
   newPullTrackangleProfiletob[i]->SetLineColor(4);
    newPullTrackangleProfiletob[i]->SetLineStyle(2);
    PullTrackangleProfiletob[i]->Draw();
    newPullTrackangleProfiletob[i]->Draw("sames");
    //myPV->PVCompute(PullTrackangleProfiletob[i] , newPullTrackangleProfiletob[i] , te );
 }
 
 Strip->Print("PullTrackangleProfileTOBCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_rphi_layer1tob",PullTrackwidthProfiletob[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_rphi_layer2tob",PullTrackwidthProfiletob[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_rphi_layer3tob",PullTrackwidthProfiletob[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_rphi_layer4tob",PullTrackwidthProfiletob[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_rphi_layer5tob",PullTrackwidthProfiletob[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_rphi_layer6tob",PullTrackwidthProfiletob[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_sas_layer1tob",PullTrackwidthProfiletob[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_sas_layer2tob",PullTrackwidthProfiletob[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_rphi_layer1tob",newPullTrackwidthProfiletob[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_rphi_layer2tob",newPullTrackwidthProfiletob[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_rphi_layer3tob",newPullTrackwidthProfiletob[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_rphi_layer4tob",newPullTrackwidthProfiletob[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_rphi_layer5tob",newPullTrackwidthProfiletob[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_rphi_layer6tob",newPullTrackwidthProfiletob[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_sas_layer1tob",newPullTrackwidthProfiletob[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_sas_layer2tob",newPullTrackwidthProfiletob[7]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   Strip->cd(i+1);
   PullTrackwidthProfiletob[i]->SetLineColor(2);
   newPullTrackwidthProfiletob[i]->SetLineColor(4);
    newPullTrackwidthProfiletob[i]->SetLineStyle(2);
    PullTrackwidthProfiletob[i]->Draw();
    newPullTrackwidthProfiletob[i]->Draw("sames");
    //myPV->PVCompute(PullTrackwidthProfiletob[i] , newPullTrackwidthProfiletob[i] , te );
 }
 
 Strip->Print("PullTrackwidthProfileTOBCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category1_rphi_layer1tob",PullTrackwidthProfileCategory1tob[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category1_rphi_layer2tob",PullTrackwidthProfileCategory1tob[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category1_rphi_layer3tob",PullTrackwidthProfileCategory1tob[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category1_rphi_layer4tob",PullTrackwidthProfileCategory1tob[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category1_rphi_layer5tob",PullTrackwidthProfileCategory1tob[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category1_rphi_layer6tob",PullTrackwidthProfileCategory1tob[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category1_sas_layer1tob",PullTrackwidthProfileCategory1tob[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category1_sas_layer2tob",PullTrackwidthProfileCategory1tob[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category1_rphi_layer1tob",newPullTrackwidthProfileCategory1tob[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category1_rphi_layer2tob",newPullTrackwidthProfileCategory1tob[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category1_rphi_layer3tob",newPullTrackwidthProfileCategory1tob[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category1_rphi_layer4tob",newPullTrackwidthProfileCategory1tob[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category1_rphi_layer5tob",newPullTrackwidthProfileCategory1tob[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category1_rphi_layer6tob",newPullTrackwidthProfileCategory1tob[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category1_sas_layer1tob",newPullTrackwidthProfileCategory1tob[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category1_sas_layer2tob",newPullTrackwidthProfileCategory1tob[7]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   Strip->cd(i+1);
   PullTrackwidthProfileCategory1tob[i]->SetLineColor(2);
   newPullTrackwidthProfileCategory1tob[i]->SetLineColor(4);
    newPullTrackwidthProfileCategory1tob[i]->SetLineStyle(2);
    PullTrackwidthProfileCategory1tob[i]->Draw();
    newPullTrackwidthProfileCategory1tob[i]->Draw("sames");
    //myPV->PVCompute(PullTrackwidthProfiletob[i] , newPullTrackwidthProfiletob[i] , te );
 }
 
 Strip->Print("PullTrackwidthProfileCategory1TOBCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category2_rphi_layer1tob",PullTrackwidthProfileCategory2tob[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category2_rphi_layer2tob",PullTrackwidthProfileCategory2tob[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category2_rphi_layer3tob",PullTrackwidthProfileCategory2tob[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category2_rphi_layer4tob",PullTrackwidthProfileCategory2tob[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category2_rphi_layer5tob",PullTrackwidthProfileCategory2tob[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category2_rphi_layer6tob",PullTrackwidthProfileCategory2tob[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category2_sas_layer1tob",PullTrackwidthProfileCategory2tob[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category2_sas_layer2tob",PullTrackwidthProfileCategory2tob[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category2_rphi_layer1tob",newPullTrackwidthProfileCategory2tob[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category2_rphi_layer2tob",newPullTrackwidthProfileCategory2tob[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category2_rphi_layer3tob",newPullTrackwidthProfileCategory2tob[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category2_rphi_layer4tob",newPullTrackwidthProfileCategory2tob[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category2_rphi_layer5tob",newPullTrackwidthProfileCategory2tob[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category2_rphi_layer6tob",newPullTrackwidthProfileCategory2tob[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category2_sas_layer1tob",newPullTrackwidthProfileCategory2tob[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category2_sas_layer2tob",newPullTrackwidthProfileCategory2tob[7]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   Strip->cd(i+1);
   PullTrackwidthProfileCategory2tob[i]->SetLineColor(2);
   newPullTrackwidthProfileCategory2tob[i]->SetLineColor(4);
    newPullTrackwidthProfileCategory2tob[i]->SetLineStyle(2);
    PullTrackwidthProfileCategory2tob[i]->Draw();
    newPullTrackwidthProfileCategory2tob[i]->Draw("sames");
    //myPV->PVCompute(PullTrackwidthProfiletob[i] , newPullTrackwidthProfiletob[i] , te );
 }
 
 Strip->Print("PullTrackwidthProfileCategory2TOBCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category3_rphi_layer1tob",PullTrackwidthProfileCategory3tob[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category3_rphi_layer2tob",PullTrackwidthProfileCategory3tob[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category3_rphi_layer3tob",PullTrackwidthProfileCategory3tob[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category3_rphi_layer4tob",PullTrackwidthProfileCategory3tob[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category3_rphi_layer5tob",PullTrackwidthProfileCategory3tob[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category3_rphi_layer6tob",PullTrackwidthProfileCategory3tob[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category3_sas_layer1tob",PullTrackwidthProfileCategory3tob[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category3_sas_layer2tob",PullTrackwidthProfileCategory3tob[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category3_rphi_layer1tob",newPullTrackwidthProfileCategory3tob[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category3_rphi_layer2tob",newPullTrackwidthProfileCategory3tob[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category3_rphi_layer3tob",newPullTrackwidthProfileCategory3tob[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category3_rphi_layer4tob",newPullTrackwidthProfileCategory3tob[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category3_rphi_layer5tob",newPullTrackwidthProfileCategory3tob[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category3_rphi_layer6tob",newPullTrackwidthProfileCategory3tob[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category3_sas_layer1tob",newPullTrackwidthProfileCategory3tob[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category3_sas_layer2tob",newPullTrackwidthProfileCategory3tob[7]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   Strip->cd(i+1);
   PullTrackwidthProfileCategory3tob[i]->SetLineColor(2);
   newPullTrackwidthProfileCategory3tob[i]->SetLineColor(4);
    newPullTrackwidthProfileCategory3tob[i]->SetLineStyle(2);
    PullTrackwidthProfileCategory3tob[i]->Draw();
    newPullTrackwidthProfileCategory3tob[i]->Draw("sames");
    //myPV->PVCompute(PullTrackwidthProfiletob[i] , newPullTrackwidthProfiletob[i] , te );
 }
 
 Strip->Print("PullTrackwidthProfileCategory3TOBCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category4_rphi_layer1tob",PullTrackwidthProfileCategory4tob[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category4_rphi_layer2tob",PullTrackwidthProfileCategory4tob[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category4_rphi_layer3tob",PullTrackwidthProfileCategory4tob[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category4_rphi_layer4tob",PullTrackwidthProfileCategory4tob[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category4_rphi_layer5tob",PullTrackwidthProfileCategory4tob[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category4_rphi_layer6tob",PullTrackwidthProfileCategory4tob[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category4_sas_layer1tob",PullTrackwidthProfileCategory4tob[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category4_sas_layer2tob",PullTrackwidthProfileCategory4tob[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category4_rphi_layer1tob",newPullTrackwidthProfileCategory4tob[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category4_rphi_layer2tob",newPullTrackwidthProfileCategory4tob[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category4_rphi_layer3tob",newPullTrackwidthProfileCategory4tob[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category4_rphi_layer4tob",newPullTrackwidthProfileCategory4tob[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category4_rphi_layer5tob",newPullTrackwidthProfileCategory4tob[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category4_rphi_layer6tob",newPullTrackwidthProfileCategory4tob[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category4_sas_layer1tob",newPullTrackwidthProfileCategory4tob[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/PullTrackwidthProfile_Category4_sas_layer2tob",newPullTrackwidthProfileCategory4tob[7]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   Strip->cd(i+1);
   PullTrackwidthProfileCategory4tob[i]->SetLineColor(2);
   newPullTrackwidthProfileCategory4tob[i]->SetLineColor(4);
    newPullTrackwidthProfileCategory4tob[i]->SetLineStyle(2);
    PullTrackwidthProfileCategory4tob[i]->Draw();
    newPullTrackwidthProfileCategory4tob[i]->Draw("sames");
    //myPV->PVCompute(PullTrackwidthProfiletob[i] , newPullTrackwidthProfiletob[i] , te );
 }
 
 Strip->Print("PullTrackwidthProfileCategory4TOBCompare.eps");
 */

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Nstp_rphi_layer1tob",refplotsTOB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Nstp_rphi_layer2tob",refplotsTOB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Nstp_rphi_layer3tob",refplotsTOB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Nstp_rphi_layer4tob",refplotsTOB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Nstp_rphi_layer5tob",refplotsTOB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Nstp_rphi_layer6tob",refplotsTOB[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Nstp_sas_layer1tob",refplotsTOB[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Nstp_sas_layer2tob",refplotsTOB[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Nstp_rphi_layer1tob",newplotsTOB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Nstp_rphi_layer2tob",newplotsTOB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Nstp_rphi_layer3tob",newplotsTOB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Nstp_rphi_layer4tob",newplotsTOB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Nstp_rphi_layer5tob",newplotsTOB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Nstp_rphi_layer6tob",newplotsTOB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Nstp_sas_layer1tob",newplotsTOB[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Nstp_sas_layer2tob",newplotsTOB[7]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   if (refplotsTOB[i]->GetEntries() == 0 || newplotsTOB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTOB[i],newplotsTOB[i]);
   refplotsTOB[i]->Draw();
   newplotsTOB[i]->Draw("sames");
   myPV->PVCompute(refplotsTOB[i] , newplotsTOB[i] , te );
 }
 
 Strip->Print("NstpTOBCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_rphi_layer1tob",refplotsTOB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_rphi_layer2tob",refplotsTOB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_rphi_layer3tob",refplotsTOB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_rphi_layer4tob",refplotsTOB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_rphi_layer5tob",refplotsTOB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_rphi_layer6tob",refplotsTOB[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_sas_layer1tob",refplotsTOB[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_sas_layer2tob",refplotsTOB[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_rphi_layer1tob",newplotsTOB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_rphi_layer2tob",newplotsTOB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_rphi_layer3tob",newplotsTOB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_rphi_layer4tob",newplotsTOB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_rphi_layer5tob",newplotsTOB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_rphi_layer6tob",newplotsTOB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_sas_layer1tob",newplotsTOB[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_sas_layer2tob",newplotsTOB[7]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   if (refplotsTOB[i]->GetEntries() == 0 || newplotsTOB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTOB[i],newplotsTOB[i]);
   refplotsTOB[i]->Draw();
   newplotsTOB[i]->Draw("sames");
   myPV->PVCompute(refplotsTOB[i] , newplotsTOB[i] , te );
 }
  
 Strip->Print("PosTOBCompare.eps");
  

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_LF_rphi_layer1tob",refplotsTOB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_LF_rphi_layer2tob",refplotsTOB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_LF_rphi_layer3tob",refplotsTOB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_LF_rphi_layer4tob",refplotsTOB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_LF_rphi_layer5tob",refplotsTOB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_LF_rphi_layer6tob",refplotsTOB[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_LF_sas_layer1tob",refplotsTOB[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_LF_sas_layer2tob",refplotsTOB[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_LF_rphi_layer1tob",newplotsTOB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_LF_rphi_layer2tob",newplotsTOB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_LF_rphi_layer3tob",newplotsTOB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_LF_rphi_layer4tob",newplotsTOB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_LF_rphi_layer5tob",newplotsTOB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_LF_rphi_layer6tob",newplotsTOB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_LF_sas_layer1tob",newplotsTOB[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_LF_sas_layer2tob",newplotsTOB[7]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   if (refplotsTOB[i]->GetEntries() == 0 || newplotsTOB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTOB[i],newplotsTOB[i]);
   refplotsTOB[i]->Draw();
   newplotsTOB[i]->Draw("sames");
   myPV->PVCompute(refplotsTOB[i] , newplotsTOB[i] , te );
 }
 
 Strip->Print("ErrxLFTOBCompare.eps");
  
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_MF_rphi_layer1tob",refplotsTOB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_MF_rphi_layer2tob",refplotsTOB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_MF_rphi_layer3tob",refplotsTOB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_MF_rphi_layer4tob",refplotsTOB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_MF_rphi_layer5tob",refplotsTOB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_MF_rphi_layer6tob",refplotsTOB[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_MF_sas_layer1tob",refplotsTOB[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_MF_sas_layer2tob",refplotsTOB[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_MF_rphi_layer1tob",newplotsTOB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_MF_rphi_layer2tob",newplotsTOB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_MF_rphi_layer3tob",newplotsTOB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_MF_rphi_layer4tob",newplotsTOB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_MF_rphi_layer5tob",newplotsTOB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_MF_rphi_layer6tob",newplotsTOB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_MF_sas_layer1tob",newplotsTOB[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_MF_sas_layer2tob",newplotsTOB[7]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   if (refplotsTOB[i]->GetEntries() == 0 || newplotsTOB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTOB[i],newplotsTOB[i]);
   refplotsTOB[i]->Draw();
   newplotsTOB[i]->Draw("sames");
   myPV->PVCompute(refplotsTOB[i] , newplotsTOB[i] , te );
 }
  
 Strip->Print("ErrxMFTOBCompare.eps");
  
 
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_LF_rphi_layer1tob",refplotsTOB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_LF_rphi_layer2tob",refplotsTOB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_LF_rphi_layer3tob",refplotsTOB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_LF_rphi_layer4tob",refplotsTOB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_LF_rphi_layer5tob",refplotsTOB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_LF_rphi_layer6tob",refplotsTOB[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_LF_sas_layer1tob",refplotsTOB[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_LF_sas_layer2tob",refplotsTOB[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_LF_rphi_layer1tob",newplotsTOB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_LF_rphi_layer2tob",newplotsTOB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_LF_rphi_layer3tob",newplotsTOB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_LF_rphi_layer4tob",newplotsTOB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_LF_rphi_layer5tob",newplotsTOB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_LF_rphi_layer6tob",newplotsTOB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_LF_sas_layer1tob",newplotsTOB[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_LF_sas_layer2tob",newplotsTOB[7]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   if (refplotsTOB[i]->GetEntries() == 0 || newplotsTOB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTOB[i],newplotsTOB[i]);
   refplotsTOB[i]->Draw();
   newplotsTOB[i]->Draw("sames");
   myPV->PVCompute(refplotsTOB[i] , newplotsTOB[i] , te );
 }
 
 Strip->Print("ResLFTOBCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_MF_rphi_layer1tob",refplotsTOB[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_MF_rphi_layer2tob",refplotsTOB[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_MF_rphi_layer3tob",refplotsTOB[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_MF_rphi_layer4tob",refplotsTOB[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_MF_rphi_layer5tob",refplotsTOB[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_MF_rphi_layer6tob",refplotsTOB[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_MF_sas_layer1tob",refplotsTOB[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_MF_sas_layer2tob",refplotsTOB[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_MF_rphi_layer1tob",newplotsTOB[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_MF_rphi_layer2tob",newplotsTOB[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_MF_rphi_layer3tob",newplotsTOB[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_MF_rphi_layer4tob",newplotsTOB[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_MF_rphi_layer5tob",newplotsTOB[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_MF_rphi_layer6tob",newplotsTOB[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_MF_sas_layer1tob",newplotsTOB[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Res_MF_sas_layer2tob",newplotsTOB[7]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   if (refplotsTOB[i]->GetEntries() == 0 || newplotsTOB[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTOB[i],newplotsTOB[i]);
   refplotsTOB[i]->Draw();
   newplotsTOB[i]->Draw("sames");
   myPV->PVCompute(refplotsTOB[i] , newplotsTOB[i] , te );
 }
 
 Strip->Print("ResMFTOBCompare.eps");

  /*
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Chi2_rphi_layer1tob",chi2tob[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Chi2_rphi_layer2tob",chi2tob[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Chi2_rphi_layer3tob",chi2tob[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Chi2_rphi_layer4tob",chi2tob[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Chi2_rphi_layer5tob",chi2tob[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Chi2_rphi_layer6tob",chi2tob[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Chi2_sas_layer1tob",chi2tob[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Chi2_sas_layer2tob",chi2tob[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Chi2_rphi_layer1tob",newchi2tob[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Chi2_rphi_layer2tob",newchi2tob[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Chi2_rphi_layer3tob",newchi2tob[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Chi2_rphi_layer4tob",newchi2tob[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Chi2_rphi_layer5tob",newchi2tob[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Chi2_rphi_layer6tob",newchi2tob[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Chi2_sas_layer1tob",newchi2tob[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Chi2_sas_layer2tob",newchi2tob[7]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(3,3);
  for (Int_t i=0; i<8; i++) {
    Strip->cd(i+1);
    chi2tob[i]->SetLineColor(2);
    newchi2tob[i]->SetLineColor(4);
    newchi2tob[i]->SetLineStyle(2);
    chi2tob[i]->Draw();
    newchi2tob[i]->Draw("sames");
    myPV->PVCompute(chi2tob[i] , newchi2tob[i] , te );
  }
  
  Strip->Print("Chi2TOBCompare.eps");
  */
  
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_matched_layer1tob",matchedtob[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posy_matched_layer1tob",matchedtob[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_matched_layer2tob",matchedtob[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posy_matched_layer2tob",matchedtob[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_matched_layer1tob",matchedtob[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Erry_matched_layer1tob",matchedtob[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_matched_layer2tob",matchedtob[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Erry_matched_layer2tob",matchedtob[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Resx_matched_layer1tob",matchedtob[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Resy_matched_layer1tob",matchedtob[9]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Resx_matched_layer2tob",matchedtob[10]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Resy_matched_layer2tob",matchedtob[11]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pullx_matched_layer1tob",matchedtob[12]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pully_matched_layer1tob",matchedtob[13]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pullx_matched_layer2tob",matchedtob[14]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pully_matched_layer2tob",matchedtob[15]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_matched_layer1tob",newmatchedtob[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posy_matched_layer1tob",newmatchedtob[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posx_matched_layer2tob",newmatchedtob[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Posy_matched_layer2tob",newmatchedtob[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_matched_layer1tob",newmatchedtob[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Erry_matched_layer1tob",newmatchedtob[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Errx_matched_layer2tob",newmatchedtob[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Erry_matched_layer2tob",newmatchedtob[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Resx_matched_layer1tob",newmatchedtob[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Resy_matched_layer1tob",newmatchedtob[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Resx_matched_layer2tob",newmatchedtob[10]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Resy_matched_layer2tob",newmatchedtob[11]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pullx_matched_layer1tob",newmatchedtob[12]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pully_matched_layer1tob",newmatchedtob[13]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pullx_matched_layer2tob",newmatchedtob[14]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TOB/Pully_matched_layer2tob",newmatchedtob[15]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,4);
 for (Int_t i=0; i<16; i++) {
   if (matchedtob[i]->GetEntries() == 0 || newmatchedtob[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(matchedtob[i],newmatchedtob[i]);
   matchedtob[i]->Draw();
   newmatchedtob[i]->Draw("sames");
   myPV->PVCompute(matchedtob[i] , newmatchedtob[i] , te );
 }
 
 Strip->Print("MatchedTOBCompare.eps");
 

 //=============================================================== 
// TID

 TH1F* refplotsTID[5];
 TH1F* newplotsTID[5];

 TProfile* PullTrackangleProfiletid[5];
 TProfile* PullTrackwidthProfiletid[5];
 TProfile* PullTrackwidthProfileCategory1tid[5];
 TProfile* PullTrackwidthProfileCategory2tid[5];
 TProfile* PullTrackwidthProfileCategory3tid[5];
 TProfile* PullTrackwidthProfileCategory4tid[5];
 TH1F* matchedtid[16];

 TProfile* newPullTrackangleProfiletid[5];
 TProfile* newPullTrackwidthProfiletid[5];
 TProfile* newPullTrackwidthProfileCategory1tid[5];
 TProfile* newPullTrackwidthProfileCategory2tid[5];
 TProfile* newPullTrackwidthProfileCategory3tid[5];
 TProfile* newPullTrackwidthProfileCategory4tid[5];
 TH1F* newmatchedtid[16];

 
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Adc_rphi_layer1tid",refplotsTID[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Adc_rphi_layer2tid",refplotsTID[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Adc_rphi_layer3tid",refplotsTID[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Adc_sas_layer1tid",refplotsTID[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Adc_sas_layer2tid",refplotsTID[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Adc_rphi_layer1tid",newplotsTID[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Adc_rphi_layer2tid",newplotsTID[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Adc_rphi_layer3tid",newplotsTID[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Adc_sas_layer1tid",newplotsTID[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Adc_sas_layer2tid",newplotsTID[4]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   if (refplotsTID[i]->GetEntries() == 0 || newplotsTID[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTID[i],newplotsTID[i]);
   refplotsTID[i]->Draw();
   newplotsTID[i]->Draw("sames");
   myPV->PVCompute(refplotsTID[i] , newplotsTID[i] , te );
 }
 
 Strip->Print("AdcTIDCompare.eps");

 
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_LF_rphi_layer1tid",refplotsTID[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_LF_rphi_layer2tid",refplotsTID[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_LF_rphi_layer3tid",refplotsTID[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_LF_sas_layer1tid",refplotsTID[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_LF_sas_layer2tid",refplotsTID[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_LF_rphi_layer1tid",newplotsTID[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_LF_rphi_layer2tid",newplotsTID[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_LF_rphi_layer3tid",newplotsTID[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_LF_sas_layer1tid",newplotsTID[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_LF_sas_layer2tid",newplotsTID[4]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   if (refplotsTID[i]->GetEntries() == 0 || newplotsTID[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTID[i],newplotsTID[i]);
   refplotsTID[i]->Draw();
   newplotsTID[i]->Draw("sames");
   myPV->PVCompute(refplotsTID[i] , newplotsTID[i] , te );
 }
 
 Strip->Print("PullLFTIDCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_MF_rphi_layer1tid",refplotsTID[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_MF_rphi_layer2tid",refplotsTID[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_MF_rphi_layer3tid",refplotsTID[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_MF_sas_layer1tid",refplotsTID[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_MF_sas_layer2tid",refplotsTID[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_MF_rphi_layer1tid",newplotsTID[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_MF_rphi_layer2tid",newplotsTID[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_MF_rphi_layer3tid",newplotsTID[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_MF_sas_layer1tid",newplotsTID[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pull_MF_sas_layer2tid",newplotsTID[4]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   if (refplotsTID[i]->GetEntries() == 0 || newplotsTID[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTID[i],newplotsTID[i]);
   refplotsTID[i]->Draw();
   newplotsTID[i]->Draw("sames");
   myPV->PVCompute(refplotsTID[i] , newplotsTID[i] , te );
 }
 
 Strip->Print("PullMFTIDCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackangle_rphi_layer1tid",refplotsTID[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackangle_rphi_layer2tid",refplotsTID[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackangle_rphi_layer3tid",refplotsTID[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackangle_sas_layer1tid",refplotsTID[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackangle_sas_layer2tid",refplotsTID[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackangle_rphi_layer1tid",newplotsTID[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackangle_rphi_layer2tid",newplotsTID[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackangle_rphi_layer3tid",newplotsTID[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackangle_sas_layer1tid",newplotsTID[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackangle_sas_layer2tid",newplotsTID[4]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   if (refplotsTID[i]->GetEntries() == 0 || newplotsTID[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTID[i],newplotsTID[i]);
   refplotsTID[i]->Draw();
   newplotsTID[i]->Draw("sames");
   myPV->PVCompute(refplotsTID[i] , newplotsTID[i] , te );
 }
 
 Strip->Print("TrackangleTIDCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackwidth_rphi_layer1tid",refplotsTID[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackwidth_rphi_layer2tid",refplotsTID[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackwidth_rphi_layer3tid",refplotsTID[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackwidth_sas_layer1tid",refplotsTID[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackwidth_sas_layer2tid",refplotsTID[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackwidth_rphi_layer1tid",newplotsTID[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackwidth_rphi_layer2tid",newplotsTID[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackwidth_rphi_layer3tid",newplotsTID[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackwidth_sas_layer1tid",newplotsTID[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Trackwidth_sas_layer2tid",newplotsTID[4]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   if (refplotsTID[i]->GetEntries() == 0 || newplotsTID[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTID[i],newplotsTID[i]);
   refplotsTID[i]->Draw();
   newplotsTID[i]->Draw("sames");
   myPV->PVCompute(refplotsTID[i] , newplotsTID[i] , te );
 }
 
 Strip->Print("TrackwidthTIDCompare.eps");


 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Expectedwidth_rphi_layer1tid",refplotsTID[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Expectedwidth_rphi_layer2tid",refplotsTID[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Expectedwidth_rphi_layer3tid",refplotsTID[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Expectedwidth_sas_layer1tid",refplotsTID[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Expectedwidth_sas_layer2tid",refplotsTID[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Expectedwidth_rphi_layer1tid",newplotsTID[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Expectedwidth_rphi_layer2tid",newplotsTID[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Expectedwidth_rphi_layer3tid",newplotsTID[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Expectedwidth_sas_layer1tid",newplotsTID[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Expectedwidth_sas_layer2tid",newplotsTID[4]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   if (refplotsTID[i]->GetEntries() == 0 || newplotsTID[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTID[i],newplotsTID[i]);
   refplotsTID[i]->SetLineColor(2);
   refplotsTID[i]->Add( refplotsTID[i],refplotsTID[i], 1/(refplotsTID[i]->GetEntries()),0.);
   newplotsTID[i]->SetLineColor(4);
   newplotsTID[i]->Add( newplotsTID[i],newplotsTID[i], 1/(newplotsTID[i]->GetEntries()),0.);
    newplotsTID[i]->SetLineStyle(2);
    refplotsTID[i]->Draw();
    newplotsTID[i]->Draw("sames");
    myPV->PVCompute(refplotsTID[i] , newplotsTID[i] , te );
 }
 
 Strip->Print("ExpectedwidthTIDCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Category_rphi_layer1tid",refplotsTID[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Category_rphi_layer2tid",refplotsTID[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Category_rphi_layer3tid",refplotsTID[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Category_sas_layer1tid",refplotsTID[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Category_sas_layer2tid",refplotsTID[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Category_rphi_layer1tid",newplotsTID[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Category_rphi_layer2tid",newplotsTID[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Category_rphi_layer3tid",newplotsTID[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Category_sas_layer1tid",newplotsTID[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Category_sas_layer2tid",newplotsTID[4]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   if (refplotsTID[i]->GetEntries() == 0 || newplotsTID[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTID[i],newplotsTID[i]);
   refplotsTID[i]->SetLineColor(2);
   refplotsTID[i]->Add( refplotsTID[i],refplotsTID[i], 1/(refplotsTID[i]->GetEntries()),0.);
   newplotsTID[i]->SetLineColor(4);
   newplotsTID[i]->Add( newplotsTID[i],newplotsTID[i], 1/(newplotsTID[i]->GetEntries()),0.);
    newplotsTID[i]->SetLineStyle(2);
    refplotsTID[i]->Draw();
    newplotsTID[i]->Draw("sames");
    myPV->PVCompute(refplotsTID[i] , newplotsTID[i] , te );
 }
 
 Strip->Print("CategoryTIDCompare.eps");
 /*
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackangleProfile_rphi_layer1tid",PullTrackangleProfiletid[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackangleProfile_rphi_layer2tid",PullTrackangleProfiletid[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackangleProfile_rphi_layer3tid",PullTrackangleProfiletid[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackangleProfile_sas_layer1tid",PullTrackangleProfiletid[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackangleProfile_sas_layer2tid",PullTrackangleProfiletid[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackangleProfile_rphi_layer1tid",newPullTrackangleProfiletid[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackangleProfile_rphi_layer2tid",newPullTrackangleProfiletid[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackangleProfile_rphi_layer3tid",newPullTrackangleProfiletid[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackangleProfile_sas_layer1tid",newPullTrackangleProfiletid[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackangleProfile_sas_layer2tid",newPullTrackangleProfiletid[4]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   Strip->cd(i+1);
   PullTrackangleProfiletid[i]->SetLineColor(2);
   newPullTrackangleProfiletid[i]->SetLineColor(4);
    newPullTrackangleProfiletid[i]->SetLineStyle(2);
    PullTrackangleProfiletid[i]->Draw();
    newPullTrackangleProfiletid[i]->Draw("sames");
    //myPV->PVCompute(PullTrackangleProfiletid[i] , newPullTrackangleProfiletid[i] , te );
 }
 
 Strip->Print("PullTrackangleProfileTIDCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_rphi_layer1tid",PullTrackwidthProfiletid[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_rphi_layer2tid",PullTrackwidthProfiletid[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_rphi_layer3tid",PullTrackwidthProfiletid[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_sas_layer1tid",PullTrackwidthProfiletid[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_sas_layer2tid",PullTrackwidthProfiletid[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_rphi_layer1tid",newPullTrackwidthProfiletid[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_rphi_layer2tid",newPullTrackwidthProfiletid[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_rphi_layer3tid",newPullTrackwidthProfiletid[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_sas_layer1tid",newPullTrackwidthProfiletid[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_sas_layer2tid",newPullTrackwidthProfiletid[4]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   Strip->cd(i+1);
   PullTrackwidthProfiletid[i]->SetLineColor(2);
   newPullTrackwidthProfiletid[i]->SetLineColor(4);
    newPullTrackwidthProfiletid[i]->SetLineStyle(2);
    PullTrackwidthProfiletid[i]->Draw();
    newPullTrackwidthProfiletid[i]->Draw("sames");
    //myPV->PVCompute(PullTrackwidthProfiletid[i] , newPullTrackwidthProfiletid[i] , te );
 }
 
 Strip->Print("PullTrackwidthProfileTIDCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category1_rphi_layer1tid",PullTrackwidthProfileCategory1tid[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category1_rphi_layer2tid",PullTrackwidthProfileCategory1tid[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category1_rphi_layer3tid",PullTrackwidthProfileCategory1tid[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category1_sas_layer1tid",PullTrackwidthProfileCategory1tid[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category1_sas_layer2tid",PullTrackwidthProfileCategory1tid[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category1_rphi_layer1tid",newPullTrackwidthProfileCategory1tid[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category1_rphi_layer2tid",newPullTrackwidthProfileCategory1tid[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category1_rphi_layer3tid",newPullTrackwidthProfileCategory1tid[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category1_sas_layer1tid",newPullTrackwidthProfileCategory1tid[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category1_sas_layer2tid",newPullTrackwidthProfileCategory1tid[4]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   Strip->cd(i+1);
   PullTrackwidthProfileCategory1tid[i]->SetLineColor(2);
   newPullTrackwidthProfileCategory1tid[i]->SetLineColor(4);
    newPullTrackwidthProfileCategory1tid[i]->SetLineStyle(2);
    PullTrackwidthProfileCategory1tid[i]->Draw();
    newPullTrackwidthProfileCategory1tid[i]->Draw("sames");
    //myPV->PVCompute(PullTrackwidthProfiletid[i] , newPullTrackwidthProfiletid[i] , te );
 }
 
 Strip->Print("PullTrackwidthProfileCategory1TIDCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category2_rphi_layer1tid",PullTrackwidthProfileCategory2tid[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category2_rphi_layer2tid",PullTrackwidthProfileCategory2tid[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category2_rphi_layer3tid",PullTrackwidthProfileCategory2tid[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category2_sas_layer1tid",PullTrackwidthProfileCategory2tid[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category2_sas_layer2tid",PullTrackwidthProfileCategory2tid[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category2_rphi_layer1tid",newPullTrackwidthProfileCategory2tid[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category2_rphi_layer2tid",newPullTrackwidthProfileCategory2tid[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category2_rphi_layer3tid",newPullTrackwidthProfileCategory2tid[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category2_sas_layer1tid",newPullTrackwidthProfileCategory2tid[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category2_sas_layer2tid",newPullTrackwidthProfileCategory2tid[4]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   Strip->cd(i+1);
   PullTrackwidthProfileCategory2tid[i]->SetLineColor(2);
   newPullTrackwidthProfileCategory2tid[i]->SetLineColor(4);
    newPullTrackwidthProfileCategory2tid[i]->SetLineStyle(2);
    PullTrackwidthProfileCategory2tid[i]->Draw();
    newPullTrackwidthProfileCategory2tid[i]->Draw("sames");
    //myPV->PVCompute(PullTrackwidthProfiletid[i] , newPullTrackwidthProfiletid[i] , te );
 }
 
 Strip->Print("PullTrackwidthProfileCategory2TIDCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category3_rphi_layer1tid",PullTrackwidthProfileCategory3tid[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category3_rphi_layer2tid",PullTrackwidthProfileCategory3tid[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category3_rphi_layer3tid",PullTrackwidthProfileCategory3tid[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category3_sas_layer1tid",PullTrackwidthProfileCategory3tid[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category3_sas_layer2tid",PullTrackwidthProfileCategory3tid[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category3_rphi_layer1tid",newPullTrackwidthProfileCategory3tid[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category3_rphi_layer2tid",newPullTrackwidthProfileCategory3tid[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category3_rphi_layer3tid",newPullTrackwidthProfileCategory3tid[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category3_sas_layer1tid",newPullTrackwidthProfileCategory3tid[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category3_sas_layer2tid",newPullTrackwidthProfileCategory3tid[4]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   Strip->cd(i+1);
   PullTrackwidthProfileCategory3tid[i]->SetLineColor(2);
   newPullTrackwidthProfileCategory3tid[i]->SetLineColor(4);
    newPullTrackwidthProfileCategory3tid[i]->SetLineStyle(2);
    PullTrackwidthProfileCategory3tid[i]->Draw();
    newPullTrackwidthProfileCategory3tid[i]->Draw("sames");
    //myPV->PVCompute(PullTrackwidthProfiletid[i] , newPullTrackwidthProfiletid[i] , te );
 }
 
 Strip->Print("PullTrackwidthProfileCategory3TIDCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category4_rphi_layer1tid",PullTrackwidthProfileCategory4tid[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category4_rphi_layer2tid",PullTrackwidthProfileCategory4tid[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category4_rphi_layer3tid",PullTrackwidthProfileCategory4tid[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category4_sas_layer1tid",PullTrackwidthProfileCategory4tid[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category4_sas_layer2tid",PullTrackwidthProfileCategory4tid[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category4_rphi_layer1tid",newPullTrackwidthProfileCategory4tid[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category4_rphi_layer2tid",newPullTrackwidthProfileCategory4tid[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category4_rphi_layer3tid",newPullTrackwidthProfileCategory4tid[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category4_sas_layer1tid",newPullTrackwidthProfileCategory4tid[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/PullTrackwidthProfile_Category4_sas_layer2tid",newPullTrackwidthProfileCategory4tid[4]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   Strip->cd(i+1);
   PullTrackwidthProfileCategory4tid[i]->SetLineColor(2);
   newPullTrackwidthProfileCategory4tid[i]->SetLineColor(4);
    newPullTrackwidthProfileCategory4tid[i]->SetLineStyle(2);
    PullTrackwidthProfileCategory4tid[i]->Draw();
    newPullTrackwidthProfileCategory4tid[i]->Draw("sames");
    //myPV->PVCompute(PullTrackwidthProfiletid[i] , newPullTrackwidthProfiletid[i] , te );
 }
 
 Strip->Print("PullTrackwidthProfileCategory4TIDCompare.eps");
 */
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Nstp_rphi_layer1tid",refplotsTID[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Nstp_rphi_layer2tid",refplotsTID[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Nstp_rphi_layer3tid",refplotsTID[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Nstp_sas_layer1tid",refplotsTID[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Nstp_sas_layer2tid",refplotsTID[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Nstp_rphi_layer1tid",newplotsTID[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Nstp_rphi_layer2tid",newplotsTID[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Nstp_rphi_layer3tid",newplotsTID[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Nstp_sas_layer1tid",newplotsTID[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Nstp_sas_layer2tid",newplotsTID[4]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   if (refplotsTID[i]->GetEntries() == 0 || newplotsTID[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTID[i],newplotsTID[i]);
   refplotsTID[i]->SetLineColor(2);
   refplotsTID[i]->Add( refplotsTID[i],refplotsTID[i], 1/(refplotsTID[i]->GetEntries()),0.);
   newplotsTID[i]->SetLineColor(4);
   newplotsTID[i]->Add( newplotsTID[i],newplotsTID[i], 1/(newplotsTID[i]->GetEntries()),0.);
   newplotsTID[i]->SetLineStyle(2);
   refplotsTID[i]->Draw();
   newplotsTID[i]->Draw("sames");
   myPV->PVCompute(refplotsTID[i] , newplotsTID[i] , te );
 }
 
 Strip->Print("NstpTIDCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Posx_rphi_layer1tid",refplotsTID[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Posx_rphi_layer2tid",refplotsTID[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Posx_rphi_layer3tid",refplotsTID[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Posx_sas_layer1tid",refplotsTID[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Posx_sas_layer2tid",refplotsTID[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Posx_rphi_layer1tid",newplotsTID[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Posx_rphi_layer2tid",newplotsTID[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Posx_rphi_layer3tid",newplotsTID[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Posx_sas_layer1tid",newplotsTID[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Posx_sas_layer2tid",newplotsTID[4]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   if (refplotsTID[i]->GetEntries() == 0 || newplotsTID[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(refplotsTID[i],newplotsTID[i]);
   refplotsTID[i]->SetLineColor(2);
   refplotsTID[i]->Add( refplotsTID[i],refplotsTID[i], 1/(refplotsTID[i]->GetEntries()),0.);
   newplotsTID[i]->SetLineColor(4);
   newplotsTID[i]->Add( newplotsTID[i],newplotsTID[i], 1/(newplotsTID[i]->GetEntries()),0.);
   newplotsTID[i]->SetLineStyle(2);
   refplotsTID[i]->Draw();
   newplotsTID[i]->Draw("sames");
   myPV->PVCompute(refplotsTID[i] , newplotsTID[i] , te );
 }
  
 Strip->Print("PosTIDCompare.eps");
  

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_LF_rphi_layer1tid",refplotsTID[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_LF_rphi_layer2tid",refplotsTID[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_LF_rphi_layer3tid",refplotsTID[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_LF_sas_layer1tid",refplotsTID[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_LF_sas_layer2tid",refplotsTID[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_LF_rphi_layer1tid",newplotsTID[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_LF_rphi_layer2tid",newplotsTID[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_LF_rphi_layer3tid",newplotsTID[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_LF_sas_layer1tid",newplotsTID[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_LF_sas_layer2tid",newplotsTID[4]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<5; i++) {
   if (refplotsTID[i]->GetEntries() == 0 || newplotsTID[i]->GetEntries() == 0) continue;
    Strip->cd(i+1);
   SetUpHistograms(refplotsTID[i],newplotsTID[i]);
    refplotsTID[i]->SetLineColor(2);
   refplotsTID[i]->Add( refplotsTID[i],refplotsTID[i], 1/(refplotsTID[i]->GetEntries()),0.);
    newplotsTID[i]->SetLineColor(4);
   newplotsTID[i]->Add( newplotsTID[i],newplotsTID[i], 1/(newplotsTID[i]->GetEntries()),0.);
    newplotsTID[i]->SetLineStyle(2);
    refplotsTID[i]->Draw();
    newplotsTID[i]->Draw("sames");
    myPV->PVCompute(refplotsTID[i] , newplotsTID[i] , te );
  }
  
  Strip->Print("ErrxLFTIDCompare.eps");
  
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_MF_rphi_layer1tid",refplotsTID[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_MF_rphi_layer2tid",refplotsTID[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_MF_rphi_layer3tid",refplotsTID[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_MF_sas_layer1tid",refplotsTID[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_MF_sas_layer2tid",refplotsTID[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_MF_rphi_layer1tid",newplotsTID[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_MF_rphi_layer2tid",newplotsTID[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_MF_rphi_layer3tid",newplotsTID[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_MF_sas_layer1tid",newplotsTID[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_MF_sas_layer2tid",newplotsTID[4]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<5; i++) {
   if (refplotsTID[i]->GetEntries() == 0 || newplotsTID[i]->GetEntries() == 0) continue;
    Strip->cd(i+1);
   SetUpHistograms(refplotsTID[i],newplotsTID[i]);
    refplotsTID[i]->SetLineColor(2);
   refplotsTID[i]->Add( refplotsTID[i],refplotsTID[i], 1/(refplotsTID[i]->GetEntries()),0.);
    newplotsTID[i]->SetLineColor(4);
   newplotsTID[i]->Add( newplotsTID[i],newplotsTID[i], 1/(newplotsTID[i]->GetEntries()),0.);
    newplotsTID[i]->SetLineStyle(2);
    refplotsTID[i]->Draw();
    newplotsTID[i]->Draw("sames");
    myPV->PVCompute(refplotsTID[i] , newplotsTID[i] , te );
  }
  
  Strip->Print("ErrxMFTIDCompare.eps");
  
 
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_LF_rphi_layer1tid",refplotsTID[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_LF_rphi_layer2tid",refplotsTID[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_LF_rphi_layer3tid",refplotsTID[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_LF_sas_layer1tid",refplotsTID[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_LF_sas_layer2tid",refplotsTID[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_LF_rphi_layer1tid",newplotsTID[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_LF_rphi_layer2tid",newplotsTID[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_LF_rphi_layer3tid",newplotsTID[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_LF_sas_layer1tid",newplotsTID[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_LF_sas_layer2tid",newplotsTID[4]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<5; i++) {
   if (refplotsTID[i]->GetEntries() == 0 || newplotsTID[i]->GetEntries() == 0) continue;
    Strip->cd(i+1);
   SetUpHistograms(refplotsTID[i],newplotsTID[i]);
    refplotsTID[i]->SetLineColor(2);
   refplotsTID[i]->Add( refplotsTID[i],refplotsTID[i], 1/(refplotsTID[i]->GetEntries()),0.);
    newplotsTID[i]->SetLineColor(4);
   newplotsTID[i]->Add( newplotsTID[i],newplotsTID[i], 1/(newplotsTID[i]->GetEntries()),0.);
    newplotsTID[i]->SetLineStyle(2);
    refplotsTID[i]->Draw();
    newplotsTID[i]->Draw("sames");
    myPV->PVCompute(refplotsTID[i] , newplotsTID[i] , te );
  }
  
  Strip->Print("ResLFTIDCompare.eps");


 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_MF_rphi_layer1tid",refplotsTID[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_MF_rphi_layer2tid",refplotsTID[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_MF_rphi_layer3tid",refplotsTID[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_MF_sas_layer1tid",refplotsTID[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_MF_sas_layer2tid",refplotsTID[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_MF_rphi_layer1tid",newplotsTID[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_MF_rphi_layer2tid",newplotsTID[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_MF_rphi_layer3tid",newplotsTID[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_MF_sas_layer1tid",newplotsTID[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Res_MF_sas_layer2tid",newplotsTID[4]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<5; i++) {
   if (refplotsTID[i]->GetEntries() == 0 || newplotsTID[i]->GetEntries() == 0) continue;
    Strip->cd(i+1);
   SetUpHistograms(refplotsTID[i],newplotsTID[i]);
    refplotsTID[i]->SetLineColor(2);
   refplotsTID[i]->Add( refplotsTID[i],refplotsTID[i], 1/(refplotsTID[i]->GetEntries()),0.);
    newplotsTID[i]->SetLineColor(4);
   newplotsTID[i]->Add( newplotsTID[i],newplotsTID[i], 1/(newplotsTID[i]->GetEntries()),0.);
    newplotsTID[i]->SetLineStyle(2);
    refplotsTID[i]->Draw();
    newplotsTID[i]->Draw("sames");
    myPV->PVCompute(refplotsTID[i] , newplotsTID[i] , te );
  }
  
  Strip->Print("ResMFTIDCompare.eps");
  /*
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Chi2_rphi_layer1tid",chi2tid[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Chi2_rphi_layer2tid",chi2tid[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Chi2_rphi_layer3tid",chi2tid[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Chi2_sas_layer1tid",chi2tid[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Chi2_sas_layer2tid",chi2tid[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Chi2_rphi_layer1tid",newchi2tid[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Chi2_rphi_layer2tid",newchi2tid[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Chi2_rphi_layer3tid",newchi2tid[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Chi2_sas_layer1tid",newchi2tid[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Chi2_sas_layer2tid",newchi2tid[4]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<5; i++) {
   if (refplotsTID[i]->GetEntries() == 0 || newplotsTID[i]->GetEntries() == 0) continue;
    Strip->cd(i+1);
    chi2tid[i]->SetLineColor(2);
   refplotsTID[i]->Add( refplotsTID[i],refplotsTID[i], 1/(refplotsTID[i]->GetEntries()),0.);
    newchi2tid[i]->SetLineColor(4);
   newplotsTID[i]->Add( newplotsTID[i],newplotsTID[i], 1/(newplotsTID[i]->GetEntries()),0.);
    newchi2tid[i]->SetLineStyle(2);
    chi2tid[i]->Draw();
    newchi2tid[i]->Draw("sames");
    myPV->PVCompute(chi2tid[i] , newchi2tid[i] , te );
  }
  
  Strip->Print("Chi2TIDCompare.eps");
  */
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Posx_matched_layer1tid",matchedtid[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Posy_matched_layer1tid",matchedtid[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Posx_matched_layer2tid",matchedtid[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Posy_matched_layer2tid",matchedtid[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_matched_layer1tid",matchedtid[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Erry_matched_layer1tid",matchedtid[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_matched_layer2tid",matchedtid[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Erry_matched_layer2tid",matchedtid[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Resx_matched_layer1tid",matchedtid[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Resy_matched_layer1tid",matchedtid[9]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Resx_matched_layer2tid",matchedtid[10]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Resy_matched_layer2tid",matchedtid[11]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pullx_matched_layer1tid",matchedtid[12]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pully_matched_layer1tid",matchedtid[13]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pullx_matched_layer2tid",matchedtid[14]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pully_matched_layer2tid",matchedtid[15]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Posx_matched_layer1tid",newmatchedtid[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Posy_matched_layer1tid",newmatchedtid[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Posx_matched_layer2tid",newmatchedtid[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Posy_matched_layer2tid",newmatchedtid[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_matched_layer1tid",newmatchedtid[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Erry_matched_layer1tid",newmatchedtid[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Errx_matched_layer2tid",newmatchedtid[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Erry_matched_layer2tid",newmatchedtid[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Resx_matched_layer1tid",newmatchedtid[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Resy_matched_layer1tid",newmatchedtid[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Resx_matched_layer2tid",newmatchedtid[10]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Resy_matched_layer2tid",newmatchedtid[11]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pullx_matched_layer1tid",newmatchedtid[12]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pully_matched_layer1tid",newmatchedtid[13]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pullx_matched_layer2tid",newmatchedtid[14]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TID/Pully_matched_layer2tid",newmatchedtid[15]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,4);
 for (Int_t i=0; i<16; i++) {
   if (matchedtid[i]->GetEntries() == 0 || newmatchedtid[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(matchedtid[i],newmatchedtid[i]);
   matchedtid[i]->SetLineColor(2);
   matchedtid[i]->Add( matchedtid[i],matchedtid[i], 1/(matchedtid[i]->GetEntries()),0.);
   newmatchedtid[i]->SetLineColor(4);
   newmatchedtid[i]->Add( newmatchedtid[i],newmatchedtid[i], 1/(newmatchedtid[i]->GetEntries()),0.);
   newmatchedtid[i]->SetLineStyle(2);
   matchedtid[i]->Draw();
   newmatchedtid[i]->Draw("sames");
   myPV->PVCompute(matchedtid[i] , newmatchedtid[i] , te );
 }
 
 Strip->Print("MatchedTIDCompare.eps");


 //======================================================================================================
// TEC

 TH1F* refplotsTEC[10];
 TH1F* newplotsTEC[10];

 TProfile* PullTrackangleProfiletec[10];
 TProfile* PullTrackwidthProfiletec[10];
 TProfile* PullTrackwidthProfileCategory1tec[10];
 TProfile* PullTrackwidthProfileCategory2tec[10];
 TProfile* PullTrackwidthProfileCategory3tec[10];
 TProfile* PullTrackwidthProfileCategory4tec[10];
 TH1F* matchedtec1[12];
 TH1F* matchedtec2[12];

 TProfile* newPullTrackangleProfiletec[10];
 TProfile* newPullTrackwidthProfiletec[10];
 TProfile* newPullTrackwidthProfileCategory1tec[10];
 TProfile* newPullTrackwidthProfileCategory2tec[10];
 TProfile* newPullTrackwidthProfileCategory3tec[10];
 TProfile* newPullTrackwidthProfileCategory4tec[10];
 TH1F* newmatchedtec1[12];
 TH1F* newmatchedtec2[12];

 
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_rphi_layer1tec",refplotsTEC[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_rphi_layer2tec",refplotsTEC[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_rphi_layer3tec",refplotsTEC[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_rphi_layer4tec",refplotsTEC[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_rphi_layer5tec",refplotsTEC[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_rphi_layer6tec",refplotsTEC[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_rphi_layer7tec",refplotsTEC[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_sas_layer1tec",refplotsTEC[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_sas_layer2tec",refplotsTEC[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_sas_layer5tec",refplotsTEC[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_rphi_layer1tec",newplotsTEC[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_rphi_layer2tec",newplotsTEC[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_rphi_layer3tec",newplotsTEC[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_rphi_layer4tec",newplotsTEC[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_rphi_layer5tec",newplotsTEC[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_rphi_layer6tec",newplotsTEC[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_rphi_layer7tec",newplotsTEC[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_sas_layer1tec",newplotsTEC[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_sas_layer2tec",newplotsTEC[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Adc_sas_layer5tec",newplotsTEC[9]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip->cd(i+1);
   SetUpHistograms(refplotsTEC[i],newplotsTEC[i]);
   refplotsTEC[i]->SetLineColor(2);
   refplotsTEC[i]->Add( refplotsTEC[i],refplotsTEC[i], 1/(refplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineColor(4);
   newplotsTEC[i]->Add( newplotsTEC[i],newplotsTEC[i], 1/(newplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineStyle(2);
   refplotsTEC[i]->Draw();
   newplotsTEC[i]->Draw("sames");
   myPV->PVCompute(refplotsTEC[i] , newplotsTEC[i] , te );
 }
 
 Strip->Print("AdcTECCompare.eps");
 
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_rphi_layer1tec",refplotsTEC[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_rphi_layer2tec",refplotsTEC[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_rphi_layer3tec",refplotsTEC[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_rphi_layer4tec",refplotsTEC[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_rphi_layer5tec",refplotsTEC[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_rphi_layer6tec",refplotsTEC[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_rphi_layer7tec",refplotsTEC[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_sas_layer1tec",refplotsTEC[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_sas_layer2tec",refplotsTEC[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_sas_layer5tec",refplotsTEC[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_rphi_layer1tec",newplotsTEC[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_rphi_layer2tec",newplotsTEC[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_rphi_layer3tec",newplotsTEC[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_rphi_layer4tec",newplotsTEC[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_rphi_layer5tec",newplotsTEC[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_rphi_layer6tec",newplotsTEC[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_rphi_layer7tec",newplotsTEC[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_sas_layer1tec",newplotsTEC[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_sas_layer2tec",newplotsTEC[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_LF_sas_layer5tec",newplotsTEC[9]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip->cd(i+1);
   SetUpHistograms(refplotsTEC[i],newplotsTEC[i]);
   refplotsTEC[i]->SetLineColor(2);
   refplotsTEC[i]->Add( refplotsTEC[i],refplotsTEC[i], 1/(refplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineColor(4);
   newplotsTEC[i]->Add( newplotsTEC[i],newplotsTEC[i], 1/(newplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineStyle(2);
   refplotsTEC[i]->Draw();
   newplotsTEC[i]->Draw("sames");
   myPV->PVCompute(refplotsTEC[i] , newplotsTEC[i] , te );
 }
 
 Strip->Print("PullLFTECCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_rphi_layer1tec",refplotsTEC[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_rphi_layer2tec",refplotsTEC[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_rphi_layer3tec",refplotsTEC[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_rphi_layer4tec",refplotsTEC[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_rphi_layer5tec",refplotsTEC[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_rphi_layer6tec",refplotsTEC[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_rphi_layer7tec",refplotsTEC[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_sas_layer1tec",refplotsTEC[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_sas_layer2tec",refplotsTEC[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_sas_layer5tec",refplotsTEC[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_rphi_layer1tec",newplotsTEC[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_rphi_layer2tec",newplotsTEC[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_rphi_layer3tec",newplotsTEC[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_rphi_layer4tec",newplotsTEC[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_rphi_layer5tec",newplotsTEC[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_rphi_layer6tec",newplotsTEC[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_rphi_layer7tec",newplotsTEC[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_sas_layer1tec",newplotsTEC[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_sas_layer2tec",newplotsTEC[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pull_MF_sas_layer5tec",newplotsTEC[9]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip->cd(i+1);
   SetUpHistograms(refplotsTEC[i],newplotsTEC[i]);
   refplotsTEC[i]->SetLineColor(2);
   refplotsTEC[i]->Add( refplotsTEC[i],refplotsTEC[i], 1/(refplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineColor(4);
   newplotsTEC[i]->Add( newplotsTEC[i],newplotsTEC[i], 1/(newplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineStyle(2);
   refplotsTEC[i]->Draw();
   newplotsTEC[i]->Draw("sames");
   myPV->PVCompute(refplotsTEC[i] , newplotsTEC[i] , te );
 }
 
 Strip->Print("PullMFTECCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_rphi_layer1tec",refplotsTEC[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_rphi_layer2tec",refplotsTEC[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_rphi_layer3tec",refplotsTEC[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_rphi_layer4tec",refplotsTEC[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_rphi_layer5tec",refplotsTEC[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_rphi_layer6tec",refplotsTEC[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_rphi_layer7tec",refplotsTEC[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_sas_layer1tec",refplotsTEC[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_sas_layer2tec",refplotsTEC[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_sas_layer5tec",refplotsTEC[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_rphi_layer1tec",newplotsTEC[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_rphi_layer2tec",newplotsTEC[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_rphi_layer3tec",newplotsTEC[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_rphi_layer4tec",newplotsTEC[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_rphi_layer5tec",newplotsTEC[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_rphi_layer6tec",newplotsTEC[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_rphi_layer7tec",newplotsTEC[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_sas_layer1tec",newplotsTEC[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_sas_layer2tec",newplotsTEC[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackangle_sas_layer5tec",newplotsTEC[9]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip->cd(i+1);
   SetUpHistograms(refplotsTEC[i],newplotsTEC[i]);
   refplotsTEC[i]->SetLineColor(2);
   refplotsTEC[i]->Add( refplotsTEC[i],refplotsTEC[i], 1/(refplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineColor(4);
   newplotsTEC[i]->Add( newplotsTEC[i],newplotsTEC[i], 1/(newplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineStyle(2);
   refplotsTEC[i]->Draw();
   newplotsTEC[i]->Draw("sames");
   myPV->PVCompute(refplotsTEC[i] , newplotsTEC[i] , te );
 }
 
 Strip->Print("TrackangleTECCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_rphi_layer1tec",refplotsTEC[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_rphi_layer2tec",refplotsTEC[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_rphi_layer3tec",refplotsTEC[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_rphi_layer4tec",refplotsTEC[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_rphi_layer5tec",refplotsTEC[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_rphi_layer6tec",refplotsTEC[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_rphi_layer7tec",refplotsTEC[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_sas_layer1tec",refplotsTEC[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_sas_layer2tec",refplotsTEC[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_sas_layer5tec",refplotsTEC[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_rphi_layer1tec",newplotsTEC[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_rphi_layer2tec",newplotsTEC[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_rphi_layer3tec",newplotsTEC[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_rphi_layer4tec",newplotsTEC[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_rphi_layer5tec",newplotsTEC[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_rphi_layer6tec",newplotsTEC[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_rphi_layer7tec",newplotsTEC[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_sas_layer1tec",newplotsTEC[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_sas_layer2tec",newplotsTEC[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Trackwidth_sas_layer5tec",newplotsTEC[9]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip->cd(i+1);
   SetUpHistograms(refplotsTEC[i],newplotsTEC[i]);
   refplotsTEC[i]->SetLineColor(2);
   refplotsTEC[i]->Add( refplotsTEC[i],refplotsTEC[i], 1/(refplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineColor(4);
   newplotsTEC[i]->Add( newplotsTEC[i],newplotsTEC[i], 1/(newplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineStyle(2);
   refplotsTEC[i]->Draw();
   newplotsTEC[i]->Draw("sames");
   myPV->PVCompute(refplotsTEC[i] , newplotsTEC[i] , te );
 }
 
 Strip->Print("TrackwidthTECCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_rphi_layer1tec",refplotsTEC[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_rphi_layer2tec",refplotsTEC[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_rphi_layer3tec",refplotsTEC[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_rphi_layer4tec",refplotsTEC[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_rphi_layer5tec",refplotsTEC[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_rphi_layer6tec",refplotsTEC[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_rphi_layer7tec",refplotsTEC[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_sas_layer1tec",refplotsTEC[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_sas_layer2tec",refplotsTEC[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_sas_layer5tec",refplotsTEC[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_rphi_layer1tec",newplotsTEC[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_rphi_layer2tec",newplotsTEC[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_rphi_layer3tec",newplotsTEC[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_rphi_layer4tec",newplotsTEC[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_rphi_layer5tec",newplotsTEC[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_rphi_layer6tec",newplotsTEC[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_rphi_layer7tec",newplotsTEC[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_sas_layer1tec",newplotsTEC[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_sas_layer2tec",newplotsTEC[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Expectedwidth_sas_layer5tec",newplotsTEC[9]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip->cd(i+1);
   SetUpHistograms(refplotsTEC[i],newplotsTEC[i]);
   refplotsTEC[i]->SetLineColor(2);
   refplotsTEC[i]->Add( refplotsTEC[i],refplotsTEC[i], 1/(refplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineColor(4);
   newplotsTEC[i]->Add( newplotsTEC[i],newplotsTEC[i], 1/(newplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineStyle(2);
   refplotsTEC[i]->Draw();
   newplotsTEC[i]->Draw("sames");
   myPV->PVCompute(refplotsTEC[i] , newplotsTEC[i] , te );
 }
 
 Strip->Print("ExpectedwidthTECCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_rphi_layer1tec",refplotsTEC[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_rphi_layer2tec",refplotsTEC[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_rphi_layer3tec",refplotsTEC[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_rphi_layer4tec",refplotsTEC[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_rphi_layer5tec",refplotsTEC[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_rphi_layer6tec",refplotsTEC[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_rphi_layer7tec",refplotsTEC[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_sas_layer1tec",refplotsTEC[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_sas_layer2tec",refplotsTEC[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_sas_layer5tec",refplotsTEC[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_rphi_layer1tec",newplotsTEC[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_rphi_layer2tec",newplotsTEC[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_rphi_layer3tec",newplotsTEC[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_rphi_layer4tec",newplotsTEC[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_rphi_layer5tec",newplotsTEC[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_rphi_layer6tec",newplotsTEC[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_rphi_layer7tec",newplotsTEC[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_sas_layer1tec",newplotsTEC[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_sas_layer2tec",newplotsTEC[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Category_sas_layer5tec",newplotsTEC[9]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip->cd(i+1);
   SetUpHistograms(refplotsTEC[i],newplotsTEC[i]);
   refplotsTEC[i]->SetLineColor(2);
   refplotsTEC[i]->Add( refplotsTEC[i],refplotsTEC[i], 1/(refplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineColor(4);
   newplotsTEC[i]->Add( newplotsTEC[i],newplotsTEC[i], 1/(newplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineStyle(2);
   refplotsTEC[i]->Draw();
   newplotsTEC[i]->Draw("sames");
   myPV->PVCompute(refplotsTEC[i] , newplotsTEC[i] , te );
 }
 
 Strip->Print("CategoryTECCompare.eps");

 /*
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_rphi_layer1tec",PullTrackangleProfiletec[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_rphi_layer2tec",PullTrackangleProfiletec[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_rphi_layer3tec",PullTrackangleProfiletec[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_rphi_layer4tec",PullTrackangleProfiletec[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_rphi_layer5tec",PullTrackangleProfiletec[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_rphi_layer6tec",PullTrackangleProfiletec[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_rphi_layer7tec",PullTrackangleProfiletec[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_sas_layer1tec",PullTrackangleProfiletec[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_sas_layer2tec",PullTrackangleProfiletec[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_sas_layer5tec",PullTrackangleProfiletec[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_rphi_layer1tec",newPullTrackangleProfiletec[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_rphi_layer2tec",newPullTrackangleProfiletec[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_rphi_layer3tec",newPullTrackangleProfiletec[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_rphi_layer4tec",newPullTrackangleProfiletec[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_rphi_layer5tec",newPullTrackangleProfiletec[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_rphi_layer6tec",newPullTrackangleProfiletec[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_rphi_layer7tec",newPullTrackangleProfiletec[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_sas_layer1tec",newPullTrackangleProfiletec[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_sas_layer2tec",newPullTrackangleProfiletec[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackangleProfile_sas_layer5tec",newPullTrackangleProfiletec[9]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip->cd(i+1);
   PullTrackangleProfiletec[i]->SetLineColor(2);
   newPullTrackangleProfiletec[i]->SetLineColor(4);
    newPullTrackangleProfiletec[i]->SetLineStyle(2);
    PullTrackangleProfiletec[i]->Draw();
    newPullTrackangleProfiletec[i]->Draw("sames");
    //myPV->PVCompute(PullTrackangleProfiletec[i] , newPullTrackangleProfiletec[i] , te );
 }
 
 Strip->Print("PullTrackangleProfileTECCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_rphi_layer1tec",PullTrackwidthProfiletec[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_rphi_layer2tec",PullTrackwidthProfiletec[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_rphi_layer3tec",PullTrackwidthProfiletec[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_rphi_layer4tec",PullTrackwidthProfiletec[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_rphi_layer5tec",PullTrackwidthProfiletec[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_rphi_layer6tec",PullTrackwidthProfiletec[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_rphi_layer7tec",PullTrackwidthProfiletec[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_sas_layer1tec",PullTrackwidthProfiletec[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_sas_layer2tec",PullTrackwidthProfiletec[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_sas_layer5tec",PullTrackwidthProfiletec[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_rphi_layer1tec",newPullTrackwidthProfiletec[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_rphi_layer2tec",newPullTrackwidthProfiletec[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_rphi_layer3tec",newPullTrackwidthProfiletec[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_rphi_layer4tec",newPullTrackwidthProfiletec[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_rphi_layer5tec",newPullTrackwidthProfiletec[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_rphi_layer6tec",newPullTrackwidthProfiletec[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_rphi_layer7tec",newPullTrackwidthProfiletec[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_sas_layer1tec",newPullTrackwidthProfiletec[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_sas_layer2tec",newPullTrackwidthProfiletec[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/PullTrackwidthProfile_sas_layer5tec",newPullTrackwidthProfiletec[9]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip->cd(i+1);
   PullTrackwidthProfiletec[i]->SetLineColor(2);
   newPullTrackwidthProfiletec[i]->SetLineColor(4);
    newPullTrackwidthProfiletec[i]->SetLineStyle(2);
    PullTrackwidthProfiletec[i]->Draw();
    newPullTrackwidthProfiletec[i]->Draw("sames");
    //myPV->PVCompute(PullTrackwidthProfiletec[i] , newPullTrackwidthProfiletec[i] , te );
 }
 
 Strip->Print("PullTrackwidthProfileTECCompare.eps");
 */
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_rphi_layer1tec",refplotsTEC[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_rphi_layer2tec",refplotsTEC[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_rphi_layer3tec",refplotsTEC[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_rphi_layer4tec",refplotsTEC[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_rphi_layer5tec",refplotsTEC[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_rphi_layer6tec",refplotsTEC[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_rphi_layer7tec",refplotsTEC[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_sas_layer1tec",refplotsTEC[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_sas_layer2tec",refplotsTEC[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_sas_layer5tec",refplotsTEC[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_rphi_layer1tec",newplotsTEC[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_rphi_layer2tec",newplotsTEC[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_rphi_layer3tec",newplotsTEC[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_rphi_layer4tec",newplotsTEC[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_rphi_layer5tec",newplotsTEC[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_rphi_layer6tec",newplotsTEC[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_rphi_layer7tec",newplotsTEC[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_sas_layer1tec",newplotsTEC[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_sas_layer2tec",newplotsTEC[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Nstp_sas_layer5tec",newplotsTEC[9]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip->cd(i+1);
   SetUpHistograms(refplotsTEC[i],newplotsTEC[i]);
   refplotsTEC[i]->SetLineColor(2);
   refplotsTEC[i]->Add( refplotsTEC[i],refplotsTEC[i], 1/(refplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineColor(4);
   newplotsTEC[i]->Add( newplotsTEC[i],newplotsTEC[i], 1/(newplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineStyle(2);
   refplotsTEC[i]->Draw();
   newplotsTEC[i]->Draw("sames");
   myPV->PVCompute(refplotsTEC[i] , newplotsTEC[i] , te );
 }
 
 Strip->Print("NstpTECCompare.eps");

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_rphi_layer1tec",refplotsTEC[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_rphi_layer2tec",refplotsTEC[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_rphi_layer3tec",refplotsTEC[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_rphi_layer4tec",refplotsTEC[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_rphi_layer5tec",refplotsTEC[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_rphi_layer6tec",refplotsTEC[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_rphi_layer7tec",refplotsTEC[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_sas_layer1tec",refplotsTEC[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_sas_layer2tec",refplotsTEC[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_sas_layer5tec",refplotsTEC[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_rphi_layer1tec",newplotsTEC[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_rphi_layer2tec",newplotsTEC[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_rphi_layer3tec",newplotsTEC[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_rphi_layer4tec",newplotsTEC[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_rphi_layer5tec",newplotsTEC[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_rphi_layer6tec",newplotsTEC[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_rphi_layer7tec",newplotsTEC[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_sas_layer1tec",newplotsTEC[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_sas_layer2tec",newplotsTEC[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_sas_layer5tec",newplotsTEC[9]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip->cd(i+1);
   SetUpHistograms(refplotsTEC[i],newplotsTEC[i]);
   refplotsTEC[i]->SetLineColor(2);
   refplotsTEC[i]->Add( refplotsTEC[i],refplotsTEC[i], 1/(refplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineColor(4);
   newplotsTEC[i]->Add( newplotsTEC[i],newplotsTEC[i], 1/(newplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineStyle(2);
   refplotsTEC[i]->Draw();
   newplotsTEC[i]->Draw("sames");
   myPV->PVCompute(refplotsTEC[i] , newplotsTEC[i] , te );
 }
  
 Strip->Print("PosTECCompare.eps");
  

 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_rphi_layer1tec",refplotsTEC[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_rphi_layer2tec",refplotsTEC[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_rphi_layer3tec",refplotsTEC[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_rphi_layer4tec",refplotsTEC[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_rphi_layer5tec",refplotsTEC[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_rphi_layer6tec",refplotsTEC[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_rphi_layer7tec",refplotsTEC[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_sas_layer1tec",refplotsTEC[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_sas_layer2tec",refplotsTEC[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_sas_layer5tec",refplotsTEC[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_rphi_layer1tec",newplotsTEC[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_rphi_layer2tec",newplotsTEC[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_rphi_layer3tec",newplotsTEC[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_rphi_layer4tec",newplotsTEC[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_rphi_layer5tec",newplotsTEC[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_rphi_layer6tec",newplotsTEC[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_rphi_layer7tec",newplotsTEC[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_sas_layer1tec",newplotsTEC[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_sas_layer2tec",newplotsTEC[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_LF_sas_layer5tec",newplotsTEC[9]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip->cd(i+1);
   SetUpHistograms(refplotsTEC[i],newplotsTEC[i]);
   refplotsTEC[i]->SetLineColor(2);
   refplotsTEC[i]->Add( refplotsTEC[i],refplotsTEC[i], 1/(refplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineColor(4);
   newplotsTEC[i]->Add( newplotsTEC[i],newplotsTEC[i], 1/(newplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineStyle(2);
   refplotsTEC[i]->Draw();
   newplotsTEC[i]->Draw("sames");
   myPV->PVCompute(refplotsTEC[i] , newplotsTEC[i] , te );
 }
  
 Strip->Print("ErrxLFTECCompare.eps");
  
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_rphi_layer1tec",refplotsTEC[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_rphi_layer2tec",refplotsTEC[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_rphi_layer3tec",refplotsTEC[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_rphi_layer4tec",refplotsTEC[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_rphi_layer5tec",refplotsTEC[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_rphi_layer6tec",refplotsTEC[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_rphi_layer7tec",refplotsTEC[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_sas_layer1tec",refplotsTEC[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_sas_layer2tec",refplotsTEC[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_sas_layer5tec",refplotsTEC[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_rphi_layer1tec",newplotsTEC[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_rphi_layer2tec",newplotsTEC[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_rphi_layer3tec",newplotsTEC[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_rphi_layer4tec",newplotsTEC[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_rphi_layer5tec",newplotsTEC[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_rphi_layer6tec",newplotsTEC[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_rphi_layer7tec",newplotsTEC[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_sas_layer1tec",newplotsTEC[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_sas_layer2tec",newplotsTEC[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_MF_sas_layer5tec",newplotsTEC[9]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip->cd(i+1);
   SetUpHistograms(refplotsTEC[i],newplotsTEC[i]);
   refplotsTEC[i]->SetLineColor(2);
   refplotsTEC[i]->Add( refplotsTEC[i],refplotsTEC[i], 1/(refplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineColor(4);
   newplotsTEC[i]->Add( newplotsTEC[i],newplotsTEC[i], 1/(newplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineStyle(2);
   refplotsTEC[i]->Draw();
   newplotsTEC[i]->Draw("sames");
   myPV->PVCompute(refplotsTEC[i] , newplotsTEC[i] , te );
 }
  
 Strip->Print("ErrxMFTECCompare.eps");
  
 
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_rphi_layer1tec",refplotsTEC[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_rphi_layer2tec",refplotsTEC[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_rphi_layer3tec",refplotsTEC[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_rphi_layer4tec",refplotsTEC[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_rphi_layer5tec",refplotsTEC[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_rphi_layer6tec",refplotsTEC[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_rphi_layer7tec",refplotsTEC[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_sas_layer1tec",refplotsTEC[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_sas_layer2tec",refplotsTEC[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_sas_layer5tec",refplotsTEC[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_rphi_layer1tec",newplotsTEC[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_rphi_layer2tec",newplotsTEC[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_rphi_layer3tec",newplotsTEC[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_rphi_layer4tec",newplotsTEC[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_rphi_layer5tec",newplotsTEC[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_rphi_layer6tec",newplotsTEC[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_rphi_layer7tec",newplotsTEC[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_sas_layer1tec",newplotsTEC[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_sas_layer2tec",newplotsTEC[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_LF_sas_layer5tec",newplotsTEC[9]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip->cd(i+1);
   SetUpHistograms(refplotsTEC[i],newplotsTEC[i]);
   refplotsTEC[i]->SetLineColor(2);
   refplotsTEC[i]->Add( refplotsTEC[i],refplotsTEC[i], 1/(refplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineColor(4);
   newplotsTEC[i]->Add( newplotsTEC[i],newplotsTEC[i], 1/(newplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineStyle(2);
   refplotsTEC[i]->Draw();
   newplotsTEC[i]->Draw("sames");
   myPV->PVCompute(refplotsTEC[i] , newplotsTEC[i] , te );
 }
  
 Strip->Print("ResLFTECCompare.eps");


 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_rphi_layer1tec",refplotsTEC[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_rphi_layer2tec",refplotsTEC[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_rphi_layer3tec",refplotsTEC[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_rphi_layer4tec",refplotsTEC[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_rphi_layer5tec",refplotsTEC[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_rphi_layer6tec",refplotsTEC[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_rphi_layer7tec",refplotsTEC[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_sas_layer1tec",refplotsTEC[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_sas_layer2tec",refplotsTEC[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_sas_layer5tec",refplotsTEC[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_rphi_layer1tec",newplotsTEC[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_rphi_layer2tec",newplotsTEC[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_rphi_layer3tec",newplotsTEC[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_rphi_layer4tec",newplotsTEC[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_rphi_layer5tec",newplotsTEC[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_rphi_layer6tec",newplotsTEC[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_rphi_layer7tec",newplotsTEC[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_sas_layer1tec",newplotsTEC[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_sas_layer2tec",newplotsTEC[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Res_MF_sas_layer5tec",newplotsTEC[9]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip->cd(i+1);
   SetUpHistograms(refplotsTEC[i],newplotsTEC[i]);
   refplotsTEC[i]->SetLineColor(2);
   refplotsTEC[i]->Add( refplotsTEC[i],refplotsTEC[i], 1/(refplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineColor(4);
   newplotsTEC[i]->Add( newplotsTEC[i],newplotsTEC[i], 1/(newplotsTEC[i]->GetEntries()),0.);
   newplotsTEC[i]->SetLineStyle(2);
   refplotsTEC[i]->Draw();
   newplotsTEC[i]->Draw("sames");
   myPV->PVCompute(refplotsTEC[i] , newplotsTEC[i] , te );
 }
  
 Strip->Print("ResMFTECCompare.eps");



  /*
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_rphi_layer1tec",chi2tec[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_rphi_layer2tec",chi2tec[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_rphi_layer3tec",chi2tec[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_rphi_layer4tec",chi2tec[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_rphi_layer5tec",chi2tec[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_rphi_layer6tec",chi2tec[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_rphi_layer7tec",chi2tec[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_sas_layer1tec",chi2tec[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_sas_layer2tec",chi2tec[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_sas_layer5tec",chi2tec[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_rphi_layer1tec",newchi2tec[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_rphi_layer2tec",newchi2tec[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_rphi_layer3tec",newchi2tec[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_rphi_layer4tec",newchi2tec[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_rphi_layer5tec",newchi2tec[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_rphi_layer6tec",newchi2tec[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_rphi_layer7tec",newchi2tec[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_sas_layer1tec",newchi2tec[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_sas_layer2tec",newchi2tec[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_sas_layer5tec",newchi2tec[9]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(4,3);
  for (Int_t i=0; i<10; i++) {
    Strip->cd(i+1);
    chi2tec[i]->SetLineColor(2);
   refplotsTEC[i]->Add( refplotsTEC[i],refplotsTEC[i], 1/(refplotsTEC[i]->GetEntries()),0.);
    newchi2tec[i]->SetLineColor(4);
    newchi2tec[i]->SetLineStyle(2);
    chi2tec[i]->Draw();
    newchi2tec[i]->Draw("sames");
    myPV->PVCompute(chi2tec[i] , newchi2tec[i] , te );
  }
  
  Strip->Print("Chi2TECCompare.eps");
  */
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_matched_layer1tec",matchedtec1[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posy_matched_layer1tec",matchedtec1[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_matched_layer2tec",matchedtec1[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posy_matched_layer2tec",matchedtec1[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_matched_layer5tec",matchedtec1[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posy_matched_layer5tec",matchedtec1[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_matched_layer1tec",matchedtec1[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Erry_matched_layer1tec",matchedtec1[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_matched_layer2tec",matchedtec1[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Erry_matched_layer2tec",matchedtec1[9]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_matched_layer5tec",matchedtec1[10]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Erry_matched_layer5tec",matchedtec1[11]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_matched_layer1tec",newmatchedtec1[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posy_matched_layer1tec",newmatchedtec1[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_matched_layer2tec",newmatchedtec1[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posy_matched_layer2tec",newmatchedtec1[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posx_matched_layer5tec",newmatchedtec1[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Posy_matched_layer5tec",newmatchedtec1[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_matched_layer1tec",newmatchedtec1[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Erry_matched_layer1tec",newmatchedtec1[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_matched_layer2tec",newmatchedtec1[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Erry_matched_layer2tec",newmatchedtec1[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Errx_matched_layer5tec",newmatchedtec1[10]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Erry_matched_layer5tec",newmatchedtec1[11]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<12; i++) {
   if (matchedtec1[i]->GetEntries() == 0 || newmatchedtec1[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(matchedtec1[i],newmatchedtec1[i]);
   matchedtec1[i]->Draw();
   newmatchedtec1[i]->Draw("sames");
   myPV->PVCompute(matchedtec1[i] , newmatchedtec1[i] , te );
 }
 
 Strip->Print("MatchedTECCompare_1.eps");
 
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Resx_matched_layer1tec",matchedtec2[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Resy_matched_layer1tec",matchedtec2[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Resx_matched_layer2tec",matchedtec2[2]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Resy_matched_layer2tec",matchedtec2[3]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Resx_matched_layer5tec",matchedtec2[4]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Resy_matched_layer5tec",matchedtec2[5]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pullx_matched_layer1tec",matchedtec2[6]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pully_matched_layer1tec",matchedtec2[7]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pullx_matched_layer2tec",matchedtec2[8]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pully_matched_layer2tec",matchedtec2[9]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pullx_matched_layer5tec",matchedtec2[10]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pully_matched_layer5tec",matchedtec2[11]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Resx_matched_layer1tec",newmatchedtec2[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Resy_matched_layer1tec",newmatchedtec2[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Resx_matched_layer2tec",newmatchedtec2[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Resy_matched_layer2tec",newmatchedtec2[3]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Resx_matched_layer5tec",newmatchedtec2[4]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Resy_matched_layer5tec",newmatchedtec2[5]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pullx_matched_layer1tec",newmatchedtec2[6]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pully_matched_layer1tec",newmatchedtec2[7]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pullx_matched_layer2tec",newmatchedtec2[8]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pully_matched_layer2tec",newmatchedtec2[9]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pullx_matched_layer5tec",newmatchedtec2[10]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Pully_matched_layer5tec",newmatchedtec2[11]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<12; i++) {
   if (matchedtec2[i]->GetEntries() == 0 || newmatchedtec2[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(matchedtec2[i],newmatchedtec2[i]);
   matchedtec2[i]->Draw();
   newmatchedtec2[i]->Draw("sames");
   myPV->PVCompute(matchedtec2[i] , newmatchedtec2[i] , te );
 }
 
 Strip->Print("MatchedTECCompare_2.eps");


 /*
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_matched_layer1tec",matchedchi2tec[0]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_matched_layer2tec",matchedchi2tec[1]);
 rfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_matched_layer5tec",matchedchi2tec[2]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_matched_layer1tec",newmatchedchi2tec[0]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_matched_layer2tec",newmatchedchi2tec[1]);
 sfile->GetObject("DQMData/TrackingRecHits/Strip/TEC/Chi2_matched_layer5tec",newmatchedchi2tec[2]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   Strip->cd(i+1);
   matchedchi2tec[i]->SetLineColor(2);
   newmatchedchi2tec[i]->SetLineColor(4);
   newmatchedchi2tec[i]->SetLineStyle(2);
   matchedchi2tec[i]->Draw();
   newmatchedchi2tec[i]->Draw("sames");
   myPV->PVCompute(matchedchi2tec[i] , newmatchedchi2tec[i] , te );
 }
 
 Strip->Print("MatchedChi2TECCompare.eps");
 */
}


void SiStripDigiCompare()
{

 gROOT ->Reset();
 char*  rfilename = "stripdigihisto.root";
 char*  sfilename = "stripdigihisto.root";

 delete gROOT->GetListOfFiles()->FindObject(rfilename);
 delete gROOT->GetListOfFiles()->FindObject(sfilename); 

 TText* te = new TText();
 TFile * rfile = new TFile(rfilename);
 TFile * sfile = new TFile(sfilename);

 rfile->cd("DQMData");
 sfile->cd("DQMData");

 gDirectory->ls();

 Char_t histo[200];

 gROOT->ProcessLine(".x HistoCompare.C");
 HistoCompare * myPV = new HistoCompare();


// TIB
 
 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(2,2);

   TH1* meNdigiTIB_[4];
   TH1* newmeNdigiTIB_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/ndigi_tib_layer_%d_zm;1",i+1);
      rfile->GetObject(histo ,meNdigiTIB_[i]);
      sfile->GetObject(histo ,newmeNdigiTIB_[i]);
      meNdigiTIB_[i];
      newmeNdigiTIB_[i];
      Strip->cd(i+1);
      meNdigiTIB_[i]->SetLineColor(2);
      newmeNdigiTIB_[i]->SetLineColor(4);
      newmeNdigiTIB_[i]->SetLineStyle(2);
      meNdigiTIB_[i]->Draw();
      newmeNdigiTIB_[i]->Draw("sames");
      myPV->PVCompute(meNdigiTIB_[i] , newmeNdigiTIB_[i] , te );
   }

   Strip->Print("NdigiTIBCompare_ZM.eps");
 }

 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(2,2);

   TH1* meNdigiTIB_[4];
   TH1* newmeNdigiTIB_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/ndigi_tib_layer_%d_zp;1",i+1);
      rfile->GetObject(histo ,meNdigiTIB_[i]);
      sfile->GetObject(histo ,newmeNdigiTIB_[i]);
      meNdigiTIB_[i];
      newmeNdigiTIB_[i];
      Strip->cd(i+1);
      meNdigiTIB_[i]->SetLineColor(2);
      newmeNdigiTIB_[i]->SetLineColor(4);
      newmeNdigiTIB_[i]->SetLineStyle(2);
      meNdigiTIB_[i]->Draw();
      newmeNdigiTIB_[i]->Draw("sames");
      myPV->PVCompute(meNdigiTIB_[i] , newmeNdigiTIB_[i] , te );
   }

   Strip->Print("NdigiTIBCompare_ZP.eps");
 }


 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(2,2);

   TH1* meAdcTIB_[4];
   TH1* newmeAdcTIB_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/adc_tib_%d_zm;1",i+1);
      rfile->GetObject(histo ,meAdcTIB_[i]);
      sfile->GetObject(histo ,newmeAdcTIB_[i]);
      meAdcTIB_[i];
      newmeAdcTIB_[i];
      Strip->cd(i+1);
      gPad->SetLogy();
      meAdcTIB_[i]->SetLineColor(2);
      newmeAdcTIB_[i]->SetLineColor(4);
      newmeAdcTIB_[i]->SetLineStyle(2);
      meAdcTIB_[i]->Draw();
      newmeAdcTIB_[i]->Draw("sames");
      myPV->PVCompute(meAdcTIB_[i] , newmeAdcTIB_[i] , te );
   }

   Strip->Print("AdcOfStripTIBCompare_ZM.eps"); 
 }

 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(1,3);

   TH1* meStripTIB_[3];
   TH1* newmeStripTIB_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/strip_tib_%d_zm;1",i+1);
      rfile->GetObject(histo ,meStripTIB_[i]);
      sfile->GetObject(histo ,newmeStripTIB_[i]);
      meStripTIB_[i];
      newmeStripTIB_[i];
      Strip->cd(i+1);
      meStripTIB_[i]->SetLineColor(2);
      newmeStripTIB_[i]->SetLineColor(4);
      newmeStripTIB_[i]->SetLineStyle(2);
      meStripTIB_[i]->Draw();
      newmeStripTIB_[i]->Draw("sames");
      myPV->PVCompute(meStripTIB_[i], newmeStripTIB_[i] , te );
   }

   Strip->Print("StripNumOfStripTIBCompare_ZM.eps");
 }

if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(2,2);

   TH1* meAdcTIB_[4];
   TH1* newmeAdcTIB_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/adc_tib_%d_zp;1",i+1);
      rfile->GetObject(histo ,meAdcTIB_[i]);
      sfile->GetObject(histo ,newmeAdcTIB_[i]);
      meAdcTIB_[i];
      newmeAdcTIB_[i];
      Strip->cd(i+1);
      gPad->SetLogy();
      meAdcTIB_[i]->SetLineColor(2);
      newmeAdcTIB_[i]->SetLineColor(4);
      newmeAdcTIB_[i]->SetLineStyle(2);
      meAdcTIB_[i]->Draw();
      newmeAdcTIB_[i]->Draw("sames");
      myPV->PVCompute(meAdcTIB_[i] , newmeAdcTIB_[i] , te );
   }

   Strip->Print("AdcOfStripTIBCompare_ZP.eps");
 }

 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(1,3);

   TH1* meStripTIB_[3];
   TH1* newmeStripTIB_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/strip_tib_%d_zp;1",i+1);
      rfile->GetObject(histo ,meStripTIB_[i]);
      sfile->GetObject(histo ,newmeStripTIB_[i]);
      meStripTIB_[i];
      newmeStripTIB_[i];
      Strip->cd(i+1);
      meStripTIB_[i]->SetLineColor(2);
      newmeStripTIB_[i]->SetLineColor(4);
      newmeStripTIB_[i]->SetLineStyle(2);
      meStripTIB_[i]->Draw();
      newmeStripTIB_[i]->Draw("sames");
      myPV->PVCompute(meStripTIB_[i], newmeStripTIB_[i] , te );
   }

   Strip->Print("StripNumOfStripTIBCompare_ZP.eps");
 }

// TOB

 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(2,3);

   TH1* meNdigiTOB_[6];
   TH1* newmeNdigiTOB_[6];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"DQMData/ndigi_tob_layer_%d_zm;1",i+1);
      rfile->GetObject(histo ,meNdigiTOB_[i]);
      sfile->GetObject(histo ,newmeNdigiTOB_[i]);
      meNdigiTOB_[i];
      newmeNdigiTOB_[i];
      Strip->cd(i+1);
      meNdigiTOB_[i]->SetLineColor(2);
      newmeNdigiTOB_[i]->SetLineColor(4);
      newmeNdigiTOB_[i]->SetLineStyle(2);
      meNdigiTOB_[i]->Draw();
      newmeNdigiTOB_[i]->Draw("sames");
      myPV->PVCompute(meNdigiTOB_[i] , newmeNdigiTOB_[i] , te );
   }

   Strip->Print("NdigiTOBCompare_ZM.eps");
 }

 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(2,3);

   TH1* meNdigiTOB_[6];
   TH1* newmeNdigiTOB_[6];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"DQMData/ndigi_tob_layer_%d_zp;1",i+1);
      rfile->GetObject(histo ,meNdigiTOB_[i]);
      sfile->GetObject(histo ,newmeNdigiTOB_[i]);
      meNdigiTOB_[i];
      newmeNdigiTOB_[i];
      Strip->cd(i+1);
      meNdigiTOB_[i]->SetLineColor(2);
      newmeNdigiTOB_[i]->SetLineColor(4);
      newmeNdigiTOB_[i]->SetLineStyle(2);
      meNdigiTOB_[i]->Draw();
      newmeNdigiTOB_[i]->Draw("sames");
      myPV->PVCompute(meNdigiTOB_[i] , newmeNdigiTOB_[i] , te );
   }

   Strip->Print("NdigiTOBCompare_ZP.eps");
 }



 if(1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);

   TH1* meAdcTOB_[6];
   TH1* newmeAdcTOB_[6];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"DQMData/adc_tob_%d_zm;1",i+1);
      rfile->GetObject(histo ,meAdcTOB_[i]);
      sfile->GetObject(histo ,newmeAdcTOB_[i]);
      meAdcTOB_[i];
      newmeAdcTOB_[i];
      Strip->cd(i+1);
      gPad->SetLogy();
      meAdcTOB_[i]->SetLineColor(2);
      newmeAdcTOB_[i]->SetLineColor(4);
      newmeAdcTOB_[i]->SetLineStyle(2);
      meAdcTOB_[i]->Draw();
      newmeAdcTOB_[i]->Draw("sames");
      myPV->PVCompute( meAdcTOB_[i],newmeAdcTOB_[i],te);
   }
   Strip->Print("AdcOfStripTOBCompare_ZM.eps");
 }

 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meStripTOB_[6];
   TH1* newmeStripTOB_[6];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"DQMData/strip_tob_%d_zm;1",i+1);
      rfile->GetObject(histo ,meStripTOB_[i]);
      sfile->GetObject(histo ,newmeStripTOB_[i]);
      meStripTOB_[i];
      newmeStripTOB_[i];
      Strip->cd(i+1);
      meStripTOB_[i]->SetLineColor(2);
      newmeStripTOB_[i]->SetLineColor(4);
      newmeStripTOB_[i]->SetLineStyle(2);
      meStripTOB_[i]->Draw();
      newmeStripTOB_[i]->Draw("sames");
      myPV->PVCompute(meStripTOB_[i],newmeStripTOB_[i],te);
   }

   Strip->Print("StripNumOfStripTOBCompare_ZM.eps");
 }

 if(1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);

   TH1* meAdcTOB_[6];
   TH1* newmeAdcTOB_[6];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"DQMData/adc_tob_%d_zp;1",i+1);
      rfile->GetObject(histo ,meAdcTOB_[i]);
      sfile->GetObject(histo ,newmeAdcTOB_[i]);
      meAdcTOB_[i];
      newmeAdcTOB_[i];
      Strip->cd(i+1);
      gPad->SetLogy();
      meAdcTOB_[i]->SetLineColor(2);
      newmeAdcTOB_[i]->SetLineColor(4);
      newmeAdcTOB_[i]->SetLineStyle(2);
      meAdcTOB_[i]->Draw();
      newmeAdcTOB_[i]->Draw("sames");
      myPV->PVCompute( meAdcTOB_[i],newmeAdcTOB_[i],te);
   }
   Strip->Print("AdcOfStripTOBCompare_ZP.eps");
 }

 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meStripTOB_[6];
   TH1* newmeStripTOB_[6];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"DQMData/strip_tob_%d_zp;1",i+1);
      rfile->GetObject(histo ,meStripTOB_[i]);
      sfile->GetObject(histo ,newmeStripTOB_[i]);
      meStripTOB_[i];
      newmeStripTOB_[i];
      Strip->cd(i+1);
      meStripTOB_[i]->SetLineColor(2);
      newmeStripTOB_[i]->SetLineColor(4);
      newmeStripTOB_[i]->SetLineStyle(2);
      meStripTOB_[i]->Draw();
      newmeStripTOB_[i]->Draw("sames");
      myPV->PVCompute(meStripTOB_[i],newmeStripTOB_[i],te);
   }

   Strip->Print("StripNumOfStripTOBCompare_ZP.eps");
 }

//TID

 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(1,3);

   TH1* meNdigiTID_[3];
   TH1* newmeNdigiTID_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/ndigi_tid_wheel_%d_zm;1",i+1);
      rfile->GetObject(histo ,meNdigiTID_[i]);
      sfile->GetObject(histo ,newmeNdigiTID_[i]);
      meNdigiTID_[i];
      newmeNdigiTID_[i];
      Strip->cd(i+1);
      meNdigiTID_[i]->SetLineColor(2);
      newmeNdigiTID_[i]->SetLineColor(4);
      newmeNdigiTID_[i]->SetLineStyle(2);
      meNdigiTID_[i]->Draw();
      newmeNdigiTID_[i]->Draw("sames");
      myPV->PVCompute(meNdigiTID_[i] , newmeNdigiTID_[i] , te );
   }

   Strip->Print("NdigiTIDCompare_ZM.eps");
 }

if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(1,3);

   TH1* meNdigiTID_[3];
   TH1* newmeNdigiTID_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/ndigi_tid_wheel_%d_zp;1",i+1);
      rfile->GetObject(histo ,meNdigiTID_[i]);
      sfile->GetObject(histo ,newmeNdigiTID_[i]);
      meNdigiTID_[i];
      newmeNdigiTID_[i];
      Strip->cd(i+1);
      meNdigiTID_[i]->SetLineColor(2);
      newmeNdigiTID_[i]->SetLineColor(4);
      newmeNdigiTID_[i]->SetLineStyle(2);
      meNdigiTID_[i]->Draw();
      newmeNdigiTID_[i]->Draw("sames");
      myPV->PVCompute(meNdigiTID_[i] , newmeNdigiTID_[i] , te );
   }

   Strip->Print("NdigiTIDCompare_ZP.eps");
 }

   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(1,3);
   TH1* meAdcTID_[3];
   TH1* newmeAdcTID_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/adc_tid_%d_zm;1",i+1);
      rfile->GetObject(histo ,meAdcTID_[i]);
      sfile->GetObject(histo ,newmeAdcTID_[i]);
      meAdcTID_[i];
      newmeAdcTID_[i];

      Strip->cd(i+1);
      gPad->SetLogy();
      meAdcTID_[i]->SetLineColor(2);
      newmeAdcTID_[i]->SetLineColor(4);
      newmeAdcTID_[i]->SetLineStyle(2);
      meAdcTID_[i]->Draw();
      newmeAdcTID_[i]->Draw("sames");
      myPV->PVCompute(meAdcTID_[i],newmeAdcTID_[i],te);
   }

   Strip->Print("AdcOfStripTIDCompare_ZM.eps");
   }

   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(1,3);
   TH1* meStripTID_[3];
   TH1* newmeStripTID_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/strip_tid_%d_zm;1",i+1);
      rfile->GetObject(histo ,meStripTID_[i]);
      sfile->GetObject(histo ,newmeStripTID_[i]);
      meStripTID_[i];
      newmeStripTID_[i];

      Strip->cd(i+1);
      meStripTID_[i]->SetLineColor(2);
      newmeStripTID_[i]->SetLineColor(4);
      newmeStripTID_[i]->SetLineStyle(2);
      meStripTID_[i]->Draw();
      newmeStripTID_[i]->Draw("sames");
      myPV->PVCompute(meStripTID_[i],newmeStripTID_[i],te);
   }

   Strip->Print("StripNumOfStripTIDCompare_ZM.eps");
   }
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(1,3);
   TH1* meAdcTID_[3];
   TH1* newmeAdcTID_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/adc_tid_%d_zp;1",i+1);
      rfile->GetObject(histo ,meAdcTID_[i]);
      sfile->GetObject(histo ,newmeAdcTID_[i]);
      meAdcTID_[i];
      newmeAdcTID_[i];

      Strip->cd(i+1);
      gPad->SetLogy();
      meAdcTID_[i]->SetLineColor(2);
      newmeAdcTID_[i]->SetLineColor(4);
      newmeAdcTID_[i]->SetLineStyle(2);
      meAdcTID_[i]->Draw();
      newmeAdcTID_[i]->Draw("sames");
      myPV->PVCompute(meAdcTID_[i],newmeAdcTID_[i],te);
   }

   Strip->Print("AdcOfStripTIDCompare_ZP.eps");
   }

   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(1,3);
   TH1* meStripTID_[3];
   TH1* newmeStripTID_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/strip_tid_%d_zp;1",i+1);
      rfile->GetObject(histo ,meStripTID_[i]);
      sfile->GetObject(histo ,newmeStripTID_[i]);
      meStripTID_[i];
      newmeStripTID_[i];

      Strip->cd(i+1);
      meStripTID_[i]->SetLineColor(2);
      newmeStripTID_[i]->SetLineColor(4);
      newmeStripTID_[i]->SetLineStyle(2);
      meStripTID_[i]->Draw();
      newmeStripTID_[i]->Draw("sames");
      myPV->PVCompute(meStripTID_[i],newmeStripTID_[i],te);
   }

   Strip->Print("StripNumOfStripTIDCompare_ZP.eps");
   }

//IEC
 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,3);

   TH1* meNdigiTEC_[9];
   TH1* newmeNdigiTEC_[9];

   for (Int_t i=0; i<9; i++){
      sprintf(histo,"DQMData/ndigi_tec_wheel_%d_zm;1",i+1);
      rfile->GetObject(histo ,meNdigiTEC_[i]);
      sfile->GetObject(histo ,newmeNdigiTEC_[i]);
      meNdigiTEC_[i];
      newmeNdigiTEC_[i];
      Strip->cd(i+1);
      meNdigiTEC_[i]->SetLineColor(2);
      newmeNdigiTEC_[i]->SetLineColor(4);
      newmeNdigiTEC_[i]->SetLineStyle(2);
      meNdigiTEC_[i]->Draw();
      newmeNdigiTEC_[i]->Draw("sames");
      myPV->PVCompute(meNdigiTEC_[i] , newmeNdigiTEC_[i] , te );
   }

   Strip->Print("NdigiTECCompare_ZM.eps");
 }
if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,3);

   TH1* meNdigiTEC_[9];
   TH1* newmeNdigiTEC_[9];

   for (Int_t i=0; i<9; i++){
      sprintf(histo,"DQMData/ndigi_tec_wheel_%d_zp;1",i+1);
      rfile->GetObject(histo ,meNdigiTEC_[i]);
      sfile->GetObject(histo ,newmeNdigiTEC_[i]);
      meNdigiTEC_[i];
      newmeNdigiTEC_[i];
      Strip->cd(i+1);
      meNdigiTEC_[i]->SetLineColor(2);
      newmeNdigiTEC_[i]->SetLineColor(4);
      newmeNdigiTEC_[i]->SetLineStyle(2);
      meNdigiTEC_[i]->Draw();
      newmeNdigiTEC_[i]->Draw("sames");
      myPV->PVCompute(meNdigiTEC_[i] , newmeNdigiTEC_[i] , te );
   }

   Strip->Print("NdigiTECCompare_ZP.eps");
 }

   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,3);
   TH1* meAdcTEC_[9];
   TH1* newmeAdcTEC_[9];

   for (Int_t i=0; i<9; i++){
      sprintf(histo,"DQMData/adc_tec_%d_zm;1",i+1);
      rfile->GetObject(histo ,meAdcTEC_[i]);
      sfile->GetObject(histo ,newmeAdcTEC_[i]);
      meAdcTEC_[i];
      newmeAdcTEC_[i];
      Strip->cd(i+1);
      gPad->SetLogy();
      meAdcTEC_[i]->SetLineColor(2);
      newmeAdcTEC_[i]->SetLineColor(4);
      newmeAdcTEC_[i]->SetLineStyle(2);
      meAdcTEC_[i]->Draw();
      newmeAdcTEC_[i]->Draw("sames");
      myPV->PVCompute(meAdcTEC_[i],newmeAdcTEC_[i],te);
   }

   Strip->Print("AdcOfStripTECCompare_ZM.eps");
   } 

   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,3);
   TH1* meStripTEC_[9];
   TH1* newmeStripTEC_[9];
 
   for (Int_t i=0; i<9; i++){
      sprintf(histo,"DQMData/strip_tec_%d_zm;1",i+1);
      rfile->GetObject(histo ,meStripTEC_[i]);
      sfile->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1); 
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
   }

   Strip->Print("StripNumOfStripTECCompare_ZM.eps");
   }
  if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,3);
   TH1* meAdcTEC_[9];
   TH1* newmeAdcTEC_[9];

   for (Int_t i=0; i<9; i++){
      sprintf(histo,"DQMData/adc_tec_%d_zp;1",i+1);
      rfile->GetObject(histo ,meAdcTEC_[i]);
      sfile->GetObject(histo ,newmeAdcTEC_[i]);
      meAdcTEC_[i];
      newmeAdcTEC_[i];
      Strip->cd(i+1);
      gPad->SetLogy();
      meAdcTEC_[i]->SetLineColor(2);
      newmeAdcTEC_[i]->SetLineColor(4);
      newmeAdcTEC_[i]->SetLineStyle(2);
      meAdcTEC_[i]->Draw();
      newmeAdcTEC_[i]->Draw("sames");
      myPV->PVCompute(meAdcTEC_[i],newmeAdcTEC_[i],te);
   }

   Strip->Print("AdcOfStripTECCompare_ZP.eps");
   }

   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,3);
   TH1* meStripTEC_[9];
   TH1* newmeStripTEC_[9];

   for (Int_t i=0; i<9; i++){
      sprintf(histo,"DQMData/strip_tec_%d_zp;1",i+1);
      rfile->GetObject(histo ,meStripTEC_[i]);
      sfile->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1);
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
   }

   Strip->Print("StripNumOfStripTECCompare_ZP.eps");
   }

}


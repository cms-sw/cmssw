void SiPixelDigiCompare()
{

 gROOT ->Reset();
 char*  rfilename = "pixeldigihisto.root";
 char*  sfilename = "pixeldigihisto.root"; 

 delete gROOT->GetListOfFiles()->FindObject(rfilename);
 delete gROOT->GetListOfFiles()->FindObject(sfilename);

 TText* te = new TText();
 TFile * rfile = new TFile(rfilename);
 TFile * sfile = new TFile(rfilename);
 Char_t histo[200];

 rfile->cd("DQMData");
 gDirectory->ls();

 sfile->cd("DQMData");
 gDirectory->ls();
 gROOT->ProcessLine(".x HistoCompare.C");
 HistoCompare * myPV = new HistoCompare();

////////////////////////////////////
//            Barrel Pixel        //
////////////////////////////////////

///1st Layer
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"DQMData/adc_layer1ladder%d;1",i+1);
      rfile->GetObject(histo ,meAdcLadder_[i]);
      sfile->GetObject(histo ,newmeAdcLadder_[i]); 
      meAdcLadder_[i];
      newmeAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->SetLineColor(2);
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->SetLineColor(4);
      newmeAdcLadder_[i]->Draw("Sames"); 
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );
    }
   
   Pixel->Print("AdcOfPXBLayer1_compare.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"DQMData/row_layer1ladder%d;1",i+1);
      rfile->GetObject(histo ,meAdcLadder_[i]);
      sfile->GetObject(histo ,newmeAdcLadder_[i]);
      meAdcLadder_[i];
      newmeAdcLadder_[i];

      Pixel->cd(i+1);
      meAdcLadder_[i]->Draw();
      meAdcLadder_[i]->SetLineColor(2);
      newmeAdcLadder_[i]->SetLineColor(4);
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );
    }

   Pixel->Print("RowOfPXBLayer1_compare.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"DQMData/col_layer1ladder%d;1",i+1);
      rfile->GetObject(histo ,meAdcLadder_[i]);
      sfile->GetObject(histo ,newmeAdcLadder_[i]); 
      meAdcLadder_[i];
      newmeAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->SetLineColor(2);  
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );

    }

   Pixel->Print("ColOfPXBLayer1_compare.eps");
}

///2nd Layer

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];
   for (Int_t i=0; i<8; i++){
      sprintf(histo,"DQMData/adc_layer2ladder%d;1",i+1);
      rfile->GetObject(histo ,meAdcLadder_[i]);
      sfile->GetObject(histo ,newmeAdcLadder_[i]);
      meAdcLadder_[i];
      newmeAdcLadder_[i]; 
      Pixel->cd(i+1);
      meAdcLadder_[i]->SetLineColor(2);
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );
  
    }

   Pixel->Print("AdcOfPXBLayer2_compare.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];
   for (Int_t i=0; i<8; i++){
      sprintf(histo,"DQMData/row_layer2ladder%d;1",i+1);
      rfile->GetObject(histo ,meAdcLadder_[i]);
      sfile->GetObject(histo ,newmeAdcLadder_[i]);
      meAdcLadder_[i];
      newmeAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->SetLineColor(2);
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );

    }

   Pixel->Print("RowOfPXBLayer2_compare.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];
   for (Int_t i=0; i<8; i++){
      sprintf(histo,"DQMData/col_layer2ladder%d;1",i+1);
      rfile->GetObject(histo ,meAdcLadder_[i]);
      sfile->GetObject(histo ,newmeAdcLadder_[i]);
      meAdcLadder_[i];
      newmeAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->SetLineColor(2);  
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );
    }
  
   Pixel->Print("ColOfPXBLayer2_compare.eps");
}

///3rd Layer
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"DQMData/adc_layer3ladder%d;1",i+1);
      rfile->GetObject(histo ,meAdcLadder_[i]);
      sfile->GetObject(histo ,newmeAdcLadder_[i]);
      meAdcLadder_[i];
      newmeAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->SetLineColor(2);
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );
 
    }

   Pixel->Print("AdcOfPXBLayer3_compare.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"DQMData/row_layer3ladder%d;1",i+1);
      rfile->GetObject(histo ,meAdcLadder_[i]);
      sfile->GetObject(histo ,newmeAdcLadder_[i]);
      meAdcLadder_[i];
      newmeAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->SetLineColor(2);
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );

    }

   Pixel->Print("RowOfPXBLayer3_compare.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"DQMData/col_layer3ladder%d;1",i+1);
      rfile->GetObject(histo ,meAdcLadder_[i]);
      sfile->GetObject(histo ,newmeAdcLadder_[i]);
      meAdcLadder_[i];
      newmeAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->SetLineColor(2);
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );

    }

   Pixel->Print("ColOfPXBLayer3_compare.eps");
}

/* digi multiplicity */
if (1) {
   TH2* meMultiLadder_[3]; 
   TH2* newmeMultiLadder_[3];
   TProfile* meMultiLayer_[3];
   TProfile* newmeMultiLayer_[3];

   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/digi_multi_layer%d;1",i+1);
      rfile->GetObject(histo ,meMultiLadder_[i]);
      sfile->GetObject(histo ,newmeMultiLadder_[i]);
      meMultiLadder_[i];
      newmeMultiLadder_[i];
      Pixel->cd(i+1);
      meMultiLayer_[i] = meMultiLadder_[i]->ProfileX();
      newmeMultiLayer_[i] = newmeMultiLadder_[i]->ProfileX();
      meMultiLayer_[i]->SetLineColor(2);
      meMultiLayer_[i]->Draw();
      newmeMultiLayer_[i]->SetLineColor(4);
      newmeMultiLayer_[i]->Draw("Sames");
      myPV->PVCompute(meMultiLayer_[i] , newmeMultiLayer_[i] , te );
    }

   Pixel->Print("DigiNumOfBarrel_compare.eps");
}

///////////////////////////////////////////////////
//        Forward Pixel  Plots                  //
//////////////////////////////////////////////////
/*  Z Minus Side Disk 1 */
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];
   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/adc_zm_disk1_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);  
    }

   Pixel->Print("AdcZmDisk1Panel1_compare.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];
   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/row_zm_disk1_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
    }

   Pixel->Print("RowZmDisk1Panel1_compare.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/col_zm_disk1_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);

    }

   Pixel->Print("ColZmDisk1Panel1_compare.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);
   
   TH1* meAdc_[3];
   TH1* newmeAdc_[4];
   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/adc_zm_disk1_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
       sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
    }

   Pixel->Print("AdcZmDisk1Panel2_compare.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/row_zm_disk1_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
    }

   Pixel->Print("RowZmDisk1Panel2_compare.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/col_zm_disk1_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
    }

   Pixel->Print("ColZmDisk1Panel2_compare.eps");
}

/* Z Minus Side Disk 2 */
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];
   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/adc_zm_disk2_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
    }

   Pixel->Print("AdcZmDisk2Panel1_compare.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];
   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/row_zm_disk2_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
    }

   Pixel->Print("RowZmDisk2Panel1_compare.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/col_zm_disk2_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i]; 
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
    }

   Pixel->Print("ColZmDisk2Panel1_compare.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/adc_zm_disk2_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
    }

   Pixel->Print("AdcZmDisk2Panel2_compare.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/row_zm_disk2_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
    }

   Pixel->Print("RowZmDisk2Panel2_compare.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/col_zm_disk2_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
    }

   Pixel->Print("ColZmDisk2Panel2_compare.eps");
}

/*  Z Plus  Side  Disk 1 */
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/adc_zp_disk1_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
    }

   Pixel->Print("AdcZpDisk1Panel1_compare.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/row_zp_disk1_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);

    }

   Pixel->Print("RowZpDisk1Panel1_compare.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/col_zp_disk1_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);  
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
    }

   Pixel->Print("ColZpDisk1Panel1_compare.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/adc_zp_disk1_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
    }

   Pixel->Print("AdcZpDisk1Panel2_compare.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/row_zp_disk1_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);

    }

   Pixel->Print("RowZpDisk1Panel2_compare.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/col_zp_disk1_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
    }

   Pixel->Print("ColZpDisk1Panel2_compare.eps");
}

/* Z Plus Side  Disk2 */
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/adc_zp_disk2_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
    }

   Pixel->Print("AdcZpDisk2Panel1_compare.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];
   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/row_zp_disk2_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);

    }

   Pixel->Print("RowZpDisk2Panel1_compare.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/col_zp_disk2_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
    }

   Pixel->Print("ColZpDisk2Panel1_compare.eps");
}


if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/adc_zp_disk2_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
    }

   Pixel->Print("AdcZpDisk2Panel2_compare.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/row_zp_disk2_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
    }

   Pixel->Print("RowZpDisk2Panel2_compare.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/col_zp_disk2_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      sfile->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);

    }

   Pixel->Print("ColZpDisk2Panel2_compare.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH2* meMulti_[8];
   TH2* newmeMulti_[8];
   TProfile*  pro_[8];
   TProfile*  newpro_[8];

      rfile->GetObject("DQMData/digi_zp_disk1_panel1;1" ,meMulti_[0]);
      meMulti_[0];
      rfile->GetObject("DQMData/digi_zp_disk1_panel2;1" ,meMulti_[1]);
      meMulti_[1];
      rfile->GetObject("DQMData/digi_zp_disk2_panel1;1" ,meMulti_[2]);
      meMulti_[2];
      rfile->GetObject("DQMData/digi_zp_disk2_panel2;1" ,meMulti_[3]);
      meMulti_[3];
      rfile->GetObject("DQMData/digi_zm_disk1_panel1;1" ,meMulti_[4]);
      meMulti_[4];
      rfile->GetObject("DQMData/digi_zm_disk1_panel2;1" ,meMulti_[5]);
      meMulti_[5];
      rfile->GetObject("DQMData/digi_zm_disk2_panel1;1" ,meMulti_[6]);
      meMulti_[6];
      rfile->GetObject("DQMData/digi_zm_disk2_panel2;1" ,meMulti_[7]);
      meMulti_[7];

      sfile->GetObject("DQMData/digi_zp_disk1_panel1;1" ,newmeMulti_[0]);
      newmeMulti_[0];
      sfile->GetObject("DQMData/digi_zp_disk1_panel2;1" ,newmeMulti_[1]);
      newmeMulti_[1];
      sfile->GetObject("DQMData/digi_zp_disk2_panel1;1" ,newmeMulti_[2]);
      newmeMulti_[2];
      sfile->GetObject("DQMData/digi_zp_disk2_panel2;1" ,newmeMulti_[3]);
      newmeMulti_[3];
      sfile->GetObject("DQMData/digi_zm_disk1_panel1;1" ,newmeMulti_[4]);
      newmeMulti_[4];
      sfile->GetObject("DQMData/digi_zm_disk1_panel2;1" ,newmeMulti_[5]);
      newmeMulti_[5];
      sfile->GetObject("DQMData/digi_zm_disk2_panel1;1" ,newmeMulti_[6]);
      newmeMulti_[6];
      sfile->GetObject("DQMData/digi_zm_disk2_panel2;1" ,newmeMulti_[7]);
      newmeMulti_[7];

   for(int i = 0; i< 8; i ++) {
      Pixel->cd(i+1);
      pro_[i]=meMulti_[i]->ProfileX();
      newpro_[i]=newmeMulti_[i]->ProfileX();
      pro_[i]->SetLineColor(2);
      newpro_[i]->SetLineColor(4);
      pro_[i]->Draw();
      newpro_[i]->Draw("sames");
      myPV->PVCompute(pro_[i],newpro_[i],te);
   }
 
   Pixel->Print("DigiNumOfEndcap_compare.eps");
}


}

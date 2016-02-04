void SiPixelDigiPlots()
{

 gROOT ->Reset();
 char*  rfilename = "pixeldigihisto.root";

 delete gROOT->GetListOfFiles()->FindObject(rfilename);

 TText* te = new TText();
 TFile * rfile = new TFile(rfilename);
 Char_t histo[200];

 rfile->cd("DQMData/TrackerDigis/Pixel");
 gDirectory->ls();

////////////////////////////////////
//            Barrel Pixel        //
////////////////////////////////////

///1st Layer
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/adc_layer1ring%d;1",i+1);
      rfile->GetObject(histo ,meAdcLadder_[i]);
      meAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->Draw();
    }

   Pixel->Print("AdcOfPXBLayer1.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/row_layer1ring%d;1",i+1);
      rfile->GetObject(histo ,meAdcLadder_[i]);
      meAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->Draw();
    }

   Pixel->Print("RowOfPXBLayer1.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/col_layer1ring%d;1",i+1);
      rfile->GetObject(histo ,meAdcLadder_[i]);
      meAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->Draw();
    }

   Pixel->Print("ColOfPXBLayer1.eps");
}

///2nd Layer

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/adc_layer2ring%d;1",i+1);
      rfile->GetObject(histo ,meAdcLadder_[i]);
      meAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->Draw();
    }

   Pixel->Print("AdcOfPXBLayer2.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/row_layer2ring%d;1",i+1);
      rfile->GetObject(histo ,meAdcLadder_[i]);
      meAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->Draw();
    }

   Pixel->Print("RowOfPXBLayer2.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/col_layer2ring%d;1",i+1);
      rfile->GetObject(histo ,meAdcLadder_[i]);
      meAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->Draw();
    }
  
   Pixel->Print("ColOfPXBLayer2.eps");
}

///3rd Layer
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/adc_layer3ring%d;1",i+1);
      rfile->GetObject(histo ,meAdcLadder_[i]);
      meAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->Draw();
    }

   Pixel->Print("AdcOfPXBLayer3.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/row_layer3ring%d;1",i+1);
      rfile->GetObject(histo ,meAdcLadder_[i]);
      meAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->Draw();
    }

   Pixel->Print("RowOfPXBLayer3.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/col_layer3ring%d;1",i+1);
      rfile->GetObject(histo ,meAdcLadder_[i]);
      meAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->Draw();
    }

   Pixel->Print("ColOfPXBLayer3.eps");
}

/* digi multiplicity */
if (1) {
   TH2* meMultiLadder_[3]; 
   TProfile* meMultiLayer_[3];

   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/digi_multi_layer%d;1",i+1);
      rfile->GetObject(histo ,meMultiLadder_[i]);
      meMultiLadder_[i];
      Pixel->cd(i+1);
      meMultiLayer_[i] = meMultiLadder_[i]->ProfileX();
      meMultiLayer_[i]->Draw();
    }

   Pixel->Print("DigiNumOfBarrel.eps");
}

///////////////////////////////////////////////////
//        Forward Pixel  Plots                  //
//////////////////////////////////////////////////
/*  Z Minus Side Disk 1 */
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/adc_zm_disk1_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("AdcZmDisk1Panel1.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/row_zm_disk1_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("RowZmDisk1Panel1.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/col_zm_disk1_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("ColZmDisk1Panel1.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/adc_zm_disk1_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("AdcZmDisk1Panel2.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/row_zm_disk1_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("RowZmDisk1Panel2.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/col_zm_disk1_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("ColZmDisk1Panel2.eps");
}

/* Z Minus Side Disk 2 */
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/adc_zm_disk2_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("AdcZmDisk2Panel1.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/row_zm_disk2_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("RowZmDisk2Panel1.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/col_zm_disk2_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("ColZmDisk2Panel1.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/adc_zm_disk2_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("AdcZmDisk2Panel2.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/row_zm_disk2_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("RowZmDisk2Panel2.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/col_zm_disk2_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("ColZmDisk2Panel2.eps");
}

/*  Z Plus  Side  Disk 1 */
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/adc_zp_disk1_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("AdcZpDisk1Panel1.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/row_zp_disk1_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("RowZpDisk1Panel1.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/col_zp_disk1_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("ColZpDisk1Panel1.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/adc_zp_disk1_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("AdcZpDisk1Panel2.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/row_zp_disk1_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("RowZpDisk1Panel2.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/col_zp_disk1_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("ColZpDisk1Panel2.eps");
}

/* Z Plus Side  Disk2 */
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/adc_zp_disk2_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("AdcZpDisk2Panel1.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/row_zp_disk2_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("RowZpDisk2Panel1.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/col_zp_disk2_panel1_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("ColZpDisk2Panel1.eps");
}


if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/adc_zp_disk2_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("AdcZpDisk2Panel2.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/row_zp_disk2_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("RowZpDisk2Panel2.eps");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"DQMData/TrackerDigis/Pixel/col_zp_disk2_panel2_plaq%d;1",i+1);
      rfile->GetObject(histo ,meAdc_[i]);
      meAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->Draw();
    }

   Pixel->Print("ColZpDisk2Panel2.eps");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH2* meMulti_[8];
   TProfile*  pro_[8];

      rfile->GetObject("DQMData/TrackerDigis/Pixel/digi_zp_disk1_panel1;1" ,meMulti_[0]);
      meMulti_[0];
      rfile->GetObject("DQMData/TrackerDigis/Pixel/digi_zp_disk1_panel2;1" ,meMulti_[1]);
      meMulti_[1];
      rfile->GetObject("DQMData/TrackerDigis/Pixel/digi_zp_disk2_panel1;1" ,meMulti_[2]);
      meMulti_[2];
      rfile->GetObject("DQMData/TrackerDigis/Pixel/digi_zp_disk2_panel2;1" ,meMulti_[3]);
      meMulti_[3];
      rfile->GetObject("DQMData/TrackerDigis/Pixel/digi_zm_disk1_panel1;1" ,meMulti_[4]);
      meMulti_[4];
      rfile->GetObject("DQMData/TrackerDigis/Pixel/digi_zm_disk1_panel2;1" ,meMulti_[5]);
      meMulti_[5];
      rfile->GetObject("DQMData/TrackerDigis/Pixel/digi_zm_disk2_panel1;1" ,meMulti_[6]);
      meMulti_[6];
      rfile->GetObject("DQMData/TrackerDigis/Pixel/digi_zm_disk2_panel2;1" ,meMulti_[7]);
      meMulti_[7];

   for(int i = 0; i< 8; i ++) {
      Pixel->cd(i+1);
      pro_[i]=meMulti_[i]->ProfileX();
      pro_[i]->Draw();
   }
 
   Pixel->Print("DigiNumOfEndcap.eps");
}


}

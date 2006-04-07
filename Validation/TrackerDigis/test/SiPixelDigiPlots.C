void SiPixelDigiPlots()
{

 gROOT ->Reset();
 char*  rfilename = "pixeldigihisto.root";

 delete gROOT->GetListOfFiles()->FindObject(rfilename);

 TText* te = new TText();
 TFile * rfile = new TFile(rfilename);
 Char_t histo[200];

 rfile->cd("DQMData");
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
      sprintf(histo,"DQMData/adc_layer1ladder%d;1",i+1);
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
      sprintf(histo,"DQMData/row_layer1ladder%d;1",i+1);
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
      sprintf(histo,"DQMData/col_layer1ladder%d;1",i+1);
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
      sprintf(histo,"DQMData/adc_layer2ladder%d;1",i+1);
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
      sprintf(histo,"DQMData/row_layer2ladder%d;1",i+1);
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
      sprintf(histo,"DQMData/col_layer2ladder%d;1",i+1);
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
      sprintf(histo,"DQMData/adc_layer3ladder%d;1",i+1);
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
      sprintf(histo,"DQMData/row_layer3ladder%d;1",i+1);
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
      sprintf(histo,"DQMData/col_layer3ladder%d;1",i+1);
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
      sprintf(histo,"DQMData/digi_multi_layer%d;1",i+1);
      rfile->GetObject(histo ,meMultiLadder_[i]);
      meMultiLadder_[i];
      Pixel->cd(i+1);
      meMultiLayer_[i] = meMultiLadder_[i]->ProfileX();
      meMultiLayer_[i]->Draw();
    }

   Pixel->Print("DigiNumOfBarrel.eps");
}


}

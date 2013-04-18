void SiPixelDigiCompare()
{

 gROOT ->Reset();
 char*  sfilename = "pixeldigihisto.root";
 char*  rfilename = "../pixeldigihisto.root"; 

 delete gROOT->GetListOfFiles()->FindObject(rfilename);
 delete gROOT->GetListOfFiles()->FindObject(sfilename);

 TText* te = new TText();
 TFile * rfile = new TFile(rfilename);
 TDirectory * rdir=gDirectory; 
 TFile * sfile = new TFile(sfilename);
 TDirectory * sdir=gDirectory; 
 Char_t histo[200];


 TLegend leg(0.3, 0.83, 0.55, 0.90);
 //Get list of Keys from the Reference file.
  TList* ref_list = rfile->GetListOfKeys() ;
  if (!ref_list) {
      std::cout<<"=========>> AutoComaprison:: There is no Keys available in the Reference file."<<std::endl;
      exit(1) ;
   }

  //Get list of Keys from the New file.
  TList* new_list = sfile->GetListOfKeys() ;
  if (!new_list) {
      std::cout<<"=========>> AutoComaprison:: There is no Keys available in New file."<<std::endl;
      exit(1) ;
   }


  //Iterate on the List of Keys of the  Reference file.
  TIter     refkey_iter( ref_list) ;
  TKey*     ref_key ;
  TObject*  ref_obj ;

  char rver[50];
  char cver[50];
  while ( ref_key = (TKey*) refkey_iter() ) {
      ref_obj = ref_key->ReadObj() ;
      if (strcmp(ref_obj->IsA()->GetName(),"TObjString")==0) {

         TObjString * rversion = dynamic_cast< TObjString*> (ref_obj);
         sprintf(rver, "%s", rversion->GetName());
         std::cout<<" Ref. version =" << rver<<std::endl;
         break;

      }
  }

  //Iterate on the List of Keys of the  Reference file.
  TIter     newkey_iter( new_list) ;
  TKey*     new_key ;
  TObject*  new_obj ;
  while ( new_key = (TKey*) newkey_iter() ) {
      new_obj = new_key->ReadObj() ;
      if (strcmp(new_obj->IsA()->GetName(),"TObjString")==0) {

         TObjString * cversion = dynamic_cast< TObjString*> (new_obj);
         sprintf(cver, "%s", cversion->GetName());
         std::cout<<" Cur version =" << cver<<std::endl;
         break;

      }
  }

 if(rfile->cd("DQMData/Run 1/TrackerDigisV"))rfile->cd("DQMData/Run 1/TrackerDigisV/Run summary/TrackerDigis/Pixel");
 else rfile->cd("DQMData/TrackerDigisV/TrackerDigis/Pixel");
 rdir=gDirectory;

 if(sfile->cd("DQMData/Run 1/TrackerDigisV"))sfile->cd("DQMData/Run 1/TrackerDigisV/Run summary/TrackerDigis/Pixel");
 else sfile->cd("DQMData/TrackerDigisV/TrackerDigis/Pixel");
 sdir=gDirectory; 

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
      sprintf(histo,"adc_layer1ring%d;1",i+1);
      rdir->GetObject(histo ,meAdcLadder_[i]);
      sdir->GetObject(histo ,newmeAdcLadder_[i]); 
      meAdcLadder_[i];
      newmeAdcLadder_[i];
      Pixel->cd(i+1);
      gPad->SetLogy();
      meAdcLadder_[i]->SetLineColor(2);
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->SetLineColor(4);
      newmeAdcLadder_[i]->SetLineStyle(2);
      newmeAdcLadder_[i]->Draw("Sames"); 
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );
      leg.Clear();
      leg.AddEntry(meAdcLadder_[i],rver , "l");
      leg.AddEntry(newmeAdcLadder_[i],cver , "l");
      leg.Draw();


    }
   
   Pixel->Print("AdcOfPXBLayer1_compare.eps");
   Pixel->Print("AdcOfPXBLayer1_compare.gif");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"row_layer1ring%d;1",i+1);
      rdir->GetObject(histo ,meAdcLadder_[i]);
      sdir->GetObject(histo ,newmeAdcLadder_[i]);
      meAdcLadder_[i];
      newmeAdcLadder_[i];

      Pixel->cd(i+1);
      meAdcLadder_[i]->Draw();
      meAdcLadder_[i]->SetLineColor(2);
      newmeAdcLadder_[i]->SetLineColor(4);
      newmeAdcLadder_[i]->SetLineStyle(2); 
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );
      leg.Clear();
      leg.AddEntry(meAdcLadder_[i],rver , "l");
      leg.AddEntry(newmeAdcLadder_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("RowOfPXBLayer1_compare.eps");
   Pixel->Print("RowOfPXBLayer1_compare.gif");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"col_layer1ring%d;1",i+1);
      rdir->GetObject(histo ,meAdcLadder_[i]);
      sdir->GetObject(histo ,newmeAdcLadder_[i]); 
      meAdcLadder_[i];
      newmeAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->SetLineColor(2);  
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->SetLineColor(4);
      newmeAdcLadder_[i]->SetLineStyle(2);   
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );
      leg.Clear();
      leg.AddEntry(meAdcLadder_[i],rver , "l");
      leg.AddEntry(newmeAdcLadder_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("ColOfPXBLayer1_compare.eps");
   Pixel->Print("ColOfPXBLayer1_compare.gif");
}

if (1) {
  TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"digimulti_layer1ring%d;1",i+1);
      rdir->GetObject(histo ,meAdcLadder_[i]);
      sdir->GetObject(histo ,newmeAdcLadder_[i]);
      meAdcLadder_[i];
      newmeAdcLadder_[i];
      Pixel->cd(i+1);
      gPad->SetLogy();
      meAdcLadder_[i]->SetLineColor(2);
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->SetLineColor(4);
      newmeAdcLadder_[i]->SetLineStyle(2);
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );
      leg.Clear();
      leg.AddEntry(meAdcLadder_[i],rver , "l");
      leg.AddEntry(newmeAdcLadder_[i],cver , "l");
      leg.Draw();

    }


   Pixel->Print("DigiMultiPXBLayer1_compare.eps");
   Pixel->Print("DigiMultiPXBLayer1_compare.gif");
}

///2nd Layer

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];
   for (Int_t i=0; i<8; i++){
      sprintf(histo,"adc_layer2ring%d;1",i+1);
      rdir->GetObject(histo ,meAdcLadder_[i]);
      sdir->GetObject(histo ,newmeAdcLadder_[i]);
      meAdcLadder_[i];
      newmeAdcLadder_[i]; 
      Pixel->cd(i+1);
      gPad->SetLogy();
      meAdcLadder_[i]->SetLineColor(2);
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->SetLineColor(4);
      newmeAdcLadder_[i]->SetLineStyle(2);
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );
      leg.Clear();
      leg.AddEntry(meAdcLadder_[i],rver , "l");
      leg.AddEntry(newmeAdcLadder_[i],cver , "l");
      leg.Draw();
  
    }

   Pixel->Print("AdcOfPXBLayer2_compare.eps");
   Pixel->Print("AdcOfPXBLayer2_compare.gif");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];
   for (Int_t i=0; i<8; i++){
      sprintf(histo,"row_layer2ring%d;1",i+1);
      rdir->GetObject(histo ,meAdcLadder_[i]);
      sdir->GetObject(histo ,newmeAdcLadder_[i]);
      meAdcLadder_[i];
      newmeAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->SetLineColor(2);
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->SetLineColor(4);
      newmeAdcLadder_[i]->SetLineStyle(2);
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );
      leg.Clear();
      leg.AddEntry(meAdcLadder_[i],rver , "l");
      leg.AddEntry(newmeAdcLadder_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("RowOfPXBLayer2_compare.eps");
   Pixel->Print("RowOfPXBLayer2_compare.gif");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];
   for (Int_t i=0; i<8; i++){
      sprintf(histo,"col_layer2ring%d;1",i+1);
      rdir->GetObject(histo ,meAdcLadder_[i]);
      sdir->GetObject(histo ,newmeAdcLadder_[i]);
      meAdcLadder_[i];
      newmeAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->SetLineColor(2);  
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->SetLineColor(4);
      newmeAdcLadder_[i]->SetLineStyle(2);
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );
      leg.Clear();
      leg.AddEntry(meAdcLadder_[i],rver , "l");
      leg.AddEntry(newmeAdcLadder_[i],cver , "l");
      leg.Draw();

    }
  
   Pixel->Print("ColOfPXBLayer2_compare.eps");
   Pixel->Print("ColOfPXBLayer2_compare.gif");
}
if (1) {
  TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"digimulti_layer2ring%d;1",i+1);
      rdir->GetObject(histo ,meAdcLadder_[i]);
      sdir->GetObject(histo ,newmeAdcLadder_[i]);
      meAdcLadder_[i];
      newmeAdcLadder_[i];
      Pixel->cd(i+1);
      gPad->SetLogy();
      meAdcLadder_[i]->SetLineColor(2);
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->SetLineColor(4);
      newmeAdcLadder_[i]->SetLineStyle(2);
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );
      leg.Clear();
      leg.AddEntry(meAdcLadder_[i],rver , "l");
      leg.AddEntry(newmeAdcLadder_[i],cver , "l");
      leg.Draw();

    }


   Pixel->Print("DigiMultiPXBLayer2_compare.eps");
   Pixel->Print("DigiMultiPXBLayer2_compare.gif");
}

///3rd Layer
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"adc_layer3ring%d;1",i+1);
      rdir->GetObject(histo ,meAdcLadder_[i]);
      sdir->GetObject(histo ,newmeAdcLadder_[i]);
      meAdcLadder_[i];
      newmeAdcLadder_[i];
      Pixel->cd(i+1);
      gPad->SetLogy();
      meAdcLadder_[i]->SetLineColor(2);
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->SetLineColor(4);
      newmeAdcLadder_[i]->SetLineStyle(2);
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );
      leg.Clear();
      leg.AddEntry(meAdcLadder_[i],rver , "l");
      leg.AddEntry(newmeAdcLadder_[i],cver , "l");
      leg.Draw();
 
    }

   Pixel->Print("AdcOfPXBLayer3_compare.eps");
   Pixel->Print("AdcOfPXBLayer3_compare.gif");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"row_layer3ring%d;1",i+1);
      rdir->GetObject(histo ,meAdcLadder_[i]);
      sdir->GetObject(histo ,newmeAdcLadder_[i]);
      meAdcLadder_[i];
      newmeAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->SetLineColor(2);
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->SetLineColor(4);
      newmeAdcLadder_[i]->SetLineStyle(2);
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );
      leg.Clear();
      leg.AddEntry(meAdcLadder_[i],rver , "l");
      leg.AddEntry(newmeAdcLadder_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("RowOfPXBLayer3_compare.eps");
   Pixel->Print("RowOfPXBLayer3_compare.gif");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"col_layer3ring%d;1",i+1);
      rdir->GetObject(histo ,meAdcLadder_[i]);
      sdir->GetObject(histo ,newmeAdcLadder_[i]);
      meAdcLadder_[i];
      newmeAdcLadder_[i];
      Pixel->cd(i+1);
      meAdcLadder_[i]->SetLineColor(2);
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->SetLineColor(4);
      newmeAdcLadder_[i]->SetLineStyle(2);
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );
      leg.Clear();
      leg.AddEntry(meAdcLadder_[i],rver , "l");
      leg.AddEntry(newmeAdcLadder_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("ColOfPXBLayer3_compare.eps");
   Pixel->Print("ColOfPXBLayer3_compare.gif");
}

/* digi multiplicity per ring */
if (1) {
  TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meAdcLadder_[8];
   TH1* newmeAdcLadder_[8];

   for (Int_t i=0; i<8; i++){
      sprintf(histo,"digimulti_layer3ring%d;1",i+1);
      rdir->GetObject(histo ,meAdcLadder_[i]);
      sdir->GetObject(histo ,newmeAdcLadder_[i]);
      meAdcLadder_[i];
      newmeAdcLadder_[i];
      Pixel->cd(i+1);
      gPad->SetLogy();
      meAdcLadder_[i]->SetLineColor(2);
      meAdcLadder_[i]->Draw();
      newmeAdcLadder_[i]->SetLineColor(4);
      newmeAdcLadder_[i]->SetLineStyle(2);
      newmeAdcLadder_[i]->Draw("Sames");
      myPV->PVCompute(meAdcLadder_[i] , newmeAdcLadder_[i] , te );
      leg.Clear();
      leg.AddEntry(meAdcLadder_[i],rver , "l");
      leg.AddEntry(newmeAdcLadder_[i],cver , "l");
      leg.Draw();

    }


   Pixel->Print("DigiMultiPXBLayer3_compare.eps");
   Pixel->Print("DigiMultiPXBLayer3_compare.gif");
}

/* Digi Number versus Ladder Num. */
if (1) {
  TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(1,3);
   TProfile* meLadder_[3];
   TProfile* newmeLadder_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"digi_layer%d_ladders;1",i+1);
      rdir->GetObject(histo ,meLadder_[i]);
      sdir->GetObject(histo ,newmeLadder_[i]);
      meLadder_[i];
      newmeLadder_[i];
      Pixel->cd(i+1);
      //gPad->SetLogy();
      meLadder_[i]->SetLineColor(2);
      meLadder_[i]->Draw();
      newmeAdcLadder_[i]->SetLineColor(4);
      newmeLadder_[i]->SetLineStyle(2);
      newmeLadder_[i]->Draw("Sames");
      myPV->PVCompute(meLadder_[i] , newmeLadder_[i] , te );
      leg.Clear();
      leg.AddEntry(meAdcLadder_[i],rver , "l");
      leg.AddEntry(newmeAdcLadder_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("DigiMultiPXBLadders_compare.eps");
   Pixel->Print("DigiMultiPXBLadders_compare.gif");
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
      sprintf(histo,"adc_zm_disk1_panel1_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      gPad->SetLogy();
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);  
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("AdcZmDisk1Panel1_compare.eps");
   Pixel->Print("AdcZmDisk1Panel1_compare.gif");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];
   for (Int_t i=0; i<4; i++){
      sprintf(histo,"row_zm_disk1_panel1_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("RowZmDisk1Panel1_compare.eps");
   Pixel->Print("RowZmDisk1Panel1_compare.gif");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"col_zm_disk1_panel1_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("ColZmDisk1Panel1_compare.eps");
   Pixel->Print("ColZmDisk1Panel1_compare.gif");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);
   
   TH1* meAdc_[3];
   TH1* newmeAdc_[4];
   for (Int_t i=0; i<3; i++){
      sprintf(histo,"adc_zm_disk1_panel2_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
       sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      gPad->SetLogy();
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("AdcZmDisk1Panel2_compare.eps");
   Pixel->Print("AdcZmDisk1Panel2_compare.gif");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"row_zm_disk1_panel2_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("RowZmDisk1Panel2_compare.eps");
   Pixel->Print("RowZmDisk1Panel2_compare.gif");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"col_zm_disk1_panel2_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("ColZmDisk1Panel2_compare.eps");
   Pixel->Print("ColZmDisk1Panel2_compare.gif");
}

/* Z Minus Side Disk 2 */
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];
   for (Int_t i=0; i<4; i++){
      sprintf(histo,"adc_zm_disk2_panel1_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      gPad->SetLogy();
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("AdcZmDisk2Panel1_compare.eps");
   Pixel->Print("AdcZmDisk2Panel1_compare.gif");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];
   for (Int_t i=0; i<4; i++){
      sprintf(histo,"row_zm_disk2_panel1_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("RowZmDisk2Panel1_compare.eps");
   Pixel->Print("RowZmDisk2Panel1_compare.gif");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"col_zm_disk2_panel1_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i]; 
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("ColZmDisk2Panel1_compare.eps");
   Pixel->Print("ColZmDisk2Panel1_compare.gif");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"adc_zm_disk2_panel2_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      gPad->SetLogy();
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("AdcZmDisk2Panel2_compare.eps");
   Pixel->Print("AdcZmDisk2Panel2_compare.gif");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"row_zm_disk2_panel2_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("RowZmDisk2Panel2_compare.eps");
   Pixel->Print("RowZmDisk2Panel2_compare.gif");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"col_zm_disk2_panel2_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("ColZmDisk2Panel2_compare.eps");
   Pixel->Print("ColZmDisk2Panel2_compare.gif");
}

/*  Z Plus  Side  Disk 1 */
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"adc_zp_disk1_panel1_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      gPad->SetLogy();
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("AdcZpDisk1Panel1_compare.eps");
   Pixel->Print("AdcZpDisk1Panel1_compare.gif");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"row_zp_disk1_panel1_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("RowZpDisk1Panel1_compare.eps");
   Pixel->Print("RowZpDisk1Panel1_compare.gif");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"col_zp_disk1_panel1_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);  
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("ColZpDisk1Panel1_compare.eps");
   Pixel->Print("ColZpDisk1Panel1_compare.gif");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"adc_zp_disk1_panel2_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      gPad->SetLogy();
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("AdcZpDisk1Panel2_compare.eps");
   Pixel->Print("AdcZpDisk1Panel2_compare.gif");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"row_zp_disk1_panel2_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("RowZpDisk1Panel2_compare.eps");
   Pixel->Print("RowZpDisk1Panel2_compare.gif");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"col_zp_disk1_panel2_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("ColZpDisk1Panel2_compare.eps");
   Pixel->Print("ColZpDisk1Panel2_compare.gif");
}

/* Z Plus Side  Disk2 */
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"adc_zp_disk2_panel1_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      gPad->SetLogy();
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2); 
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("AdcZpDisk2Panel1_compare.eps");
   Pixel->Print("AdcZpDisk2Panel1_compare.gif");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];
   for (Int_t i=0; i<4; i++){
      sprintf(histo,"row_zp_disk2_panel1_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("RowZpDisk2Panel1_compare.eps");
   Pixel->Print("RowZpDisk2Panel1_compare.gif");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",1000,1000);
   Pixel->Divide(2,2);

   TH1* meAdc_[4];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"col_zp_disk2_panel1_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("ColZpDisk2Panel1_compare.eps");
   Pixel->Print("ColZpDisk2Panel1_compare.gif");
}


if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"adc_zp_disk2_panel2_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      gPad->SetLogy();
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("AdcZpDisk2Panel2_compare.eps");
   Pixel->Print("AdcZpDisk2Panel2_compare.gif");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"row_zp_disk2_panel2_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("RowZpDisk2Panel2_compare.eps");
   Pixel->Print("RowZpDisk2Panel2_compare.gif");
}
if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",600,1000);
   Pixel->Divide(1,3);

   TH1* meAdc_[3];
   TH1* newmeAdc_[4];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"col_zp_disk2_panel2_plaq%d;1",i+1);
      rdir->GetObject(histo ,meAdc_[i]);
      sdir->GetObject(histo ,newmeAdc_[i]);
      meAdc_[i];
      newmeAdc_[i];
      Pixel->cd(i+1);
      meAdc_[i]->SetLineColor(2);
      newmeAdc_[i]->SetLineColor(4);
      newmeAdc_[i]->SetLineStyle(2);
      meAdc_[i]->Draw();
      newmeAdc_[i]->Draw("sames");
      myPV->PVCompute(meAdc_[i],newmeAdc_[i],te);
      leg.Clear();
      leg.AddEntry(meAdc_[i],rver , "l");
      leg.AddEntry(newmeAdc_[i],cver , "l");
      leg.Draw();

    }

   Pixel->Print("ColZpDisk2Panel2_compare.eps");
   Pixel->Print("ColZpDisk2Panel2_compare.gif");
}

if (1) {
   TCanvas * Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   TH1* meMulti_[8];
   TH1* newmeMulti_[8];

      rdir->GetObject("digi_zp_disk1_panel1;1" ,meMulti_[0]);
      meMulti_[0];
      rdir->GetObject("digi_zp_disk1_panel2;1" ,meMulti_[1]);
      meMulti_[1];
      rdir->GetObject("digi_zp_disk2_panel1;1" ,meMulti_[2]);
      meMulti_[2];
      rdir->GetObject("digi_zp_disk2_panel2;1" ,meMulti_[3]);
      meMulti_[3];
      rdir->GetObject("digi_zm_disk1_panel1;1" ,meMulti_[4]);
      meMulti_[4];
      rdir->GetObject("digi_zm_disk1_panel2;1" ,meMulti_[5]);
      meMulti_[5];
      rdir->GetObject("digi_zm_disk2_panel1;1" ,meMulti_[6]);
      meMulti_[6];
      rdir->GetObject("digi_zm_disk2_panel2;1" ,meMulti_[7]);
      meMulti_[7];

      sdir->GetObject("digi_zp_disk1_panel1;1" ,newmeMulti_[0]);
      newmeMulti_[0];
      sdir->GetObject("digi_zp_disk1_panel2;1" ,newmeMulti_[1]);
      newmeMulti_[1];
      sdir->GetObject("digi_zp_disk2_panel1;1" ,newmeMulti_[2]);
      newmeMulti_[2];
      sdir->GetObject("digi_zp_disk2_panel2;1" ,newmeMulti_[3]);
      newmeMulti_[3];
      sdir->GetObject("digi_zm_disk1_panel1;1" ,newmeMulti_[4]);
      newmeMulti_[4];
      sdir->GetObject("digi_zm_disk1_panel2;1" ,newmeMulti_[5]);
      newmeMulti_[5];
      sdir->GetObject("digi_zm_disk2_panel1;1" ,newmeMulti_[6]);
      newmeMulti_[6];
      sdir->GetObject("digi_zm_disk2_panel2;1" ,newmeMulti_[7]);
      newmeMulti_[7];

   for(int i = 0; i< 8; i ++) {
      Pixel->cd(i+1);
      gPad->SetLogy();
      meMulti_[i]->SetLineColor(2);
      newmeMulti_[i]->SetLineColor(4);
      newmeMulti_[i]->SetLineStyle(2);
      meMulti_[i]->Draw();
      newmeMulti_[i]->Draw("sames");
      myPV->PVCompute(meMulti_[i],newmeMulti_[i],te);
      leg.Clear();
      leg.AddEntry(meMulti_[i],rver , "l");
      leg.AddEntry(newmeMulti_[i],cver , "l");
      leg.Draw();

   }
 
   Pixel->Print("DigiMultiOfEndcap_compare.eps");
   Pixel->Print("DigiMultiOfEndcap_compare.gif");
}


}

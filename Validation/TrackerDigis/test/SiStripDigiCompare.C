void SiStripDigiCompare()
{

 gROOT ->Reset();
 char*  sfilename = "stripdigihisto.root";
 char*  rfilename = "../stripdigihisto.root";

 delete gROOT->GetListOfFiles()->FindObject(rfilename);
 delete gROOT->GetListOfFiles()->FindObject(sfilename); 

 TText* te = new TText();
 TFile * rfile = new TFile(rfilename);
 TDirectory * rdir=gDirectory; 
 TFile * sfile = new TFile(sfilename);
 TDirectory * sdir=gDirectory; 

 if(rfile->cd("DQMData/Run 1/TrackerDigisV"))rfile->cd("DQMData/Run 1/TrackerDigisV/Run summary/TrackerDigis/Strip");
 else rfile->cd("DQMData/TrackerDigisV/TrackerDigis/Strip");
 rdir=gDirectory;

 if(sfile->cd("DQMData/Run 1/TrackerDigisV"))sfile->cd("DQMData/Run 1/TrackerDigisV/Run summary/TrackerDigis/Strip");
 else sfile->cd("DQMData/TrackerDigisV/TrackerDigis/Strip");
 sdir=gDirectory; 

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

 //gDirectory->ls();

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
      sprintf(histo,"ndigi_tib_layer_%d_zm;1",i+1);
      rdir->GetObject(histo ,meNdigiTIB_[i]);
      sdir->GetObject(histo ,newmeNdigiTIB_[i]);
      meNdigiTIB_[i];
      newmeNdigiTIB_[i];
      Strip->cd(i+1);
      meNdigiTIB_[i]->SetLineColor(2);
      newmeNdigiTIB_[i]->SetLineColor(4);
      newmeNdigiTIB_[i]->SetLineStyle(2);
      meNdigiTIB_[i]->Draw();
      newmeNdigiTIB_[i]->Draw("sames");
      myPV->PVCompute(meNdigiTIB_[i] , newmeNdigiTIB_[i] , te );
      leg.Clear();
      leg.AddEntry(meNdigiTIB_[i],rver , "l");
      leg.AddEntry(newmeNdigiTIB_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("NdigiTIBCompare_ZM.eps");
   Strip->Print("NdigiTIBCompare_ZM.gif");
 }

 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(2,2);

   TH1* meNdigiTIB_[4];
   TH1* newmeNdigiTIB_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"ndigi_tib_layer_%d_zp;1",i+1);
      rdir->GetObject(histo ,meNdigiTIB_[i]);
      sdir->GetObject(histo ,newmeNdigiTIB_[i]);
      meNdigiTIB_[i];
      newmeNdigiTIB_[i];
      Strip->cd(i+1);
      meNdigiTIB_[i]->SetLineColor(2);
      newmeNdigiTIB_[i]->SetLineColor(4);
      newmeNdigiTIB_[i]->SetLineStyle(2);
      meNdigiTIB_[i]->Draw();
      newmeNdigiTIB_[i]->Draw("sames");
      myPV->PVCompute(meNdigiTIB_[i] , newmeNdigiTIB_[i] , te );
      leg.Clear();
      leg.AddEntry(meNdigiTIB_[i],rver , "l");
      leg.AddEntry(newmeNdigiTIB_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("NdigiTIBCompare_ZP.eps");
   Strip->Print("NdigiTIBCompare_ZP.gif");
 }

//TIB  1st Lyaer ADC of both sides
 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTIB_[12];
   TH1* newmeAdcTIB_[12];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"adc_tib_layer1_extmodule%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

  
  for (Int_t i=3; i<6; i++){
      sprintf(histo,"adc_tib_layer1_intmodule%d_zp;1",i-2);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<9; i++){
      sprintf(histo,"adc_tib_layer1_extmodule%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

  for (Int_t i=9; i<12; i++){
      sprintf(histo,"adc_tib_layer1_intmodule%d_zm;1",i-8);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

 Strip->Print("AdcOfTIBLayer1Compare.eps"); 
 Strip->Print("AdcOfTIBLayer1Compare.gif"); 
 }
//TIB  1st Lyaer Strip of both sides
 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTIB_[12];
   TH1* newmeAdcTIB_[12];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"strip_tib_layer1_extmodule%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

  for (Int_t i=3; i<6; i++){
      sprintf(histo,"strip_tib_layer1_intmodule%d_zp;1",i-2);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<9; i++){
      sprintf(histo,"strip_tib_layer1_extmodule%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

  for (Int_t i=9; i<12; i++){
      sprintf(histo,"strip_tib_layer1_intmodule%d_zm;1",i-8);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

 Strip->Print("StripNumOfTIBLayer1Compare.eps");
 Strip->Print("StripNumOfTIBLayer1Compare.gif");

 }
//ITB  2nd Lyaer
 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTIB_[12];
   TH1* newmeAdcTIB_[12];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"adc_tib_layer2_extmodule%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

  for (Int_t i=3; i<6; i++){
      sprintf(histo,"adc_tib_layer2_intmodule%d_zp;1",i-2);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<9; i++){
      sprintf(histo,"adc_tib_layer2_extmodule%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

  for (Int_t i=9; i<12; i++){
      sprintf(histo,"adc_tib_layer2_intmodule%d_zm;1",i-8);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

 Strip->Print("AdcOfTIBLayer2Compare.eps");
 Strip->Print("AdcOfTIBLayer2Compare.gif");
 }

 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTIB_[12];
   TH1* newmeAdcTIB_[12];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"strip_tib_layer2_extmodule%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

  for (Int_t i=3; i<6; i++){
      sprintf(histo,"strip_tib_layer2_intmodule%d_zp;1",i-2);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<9; i++){
      sprintf(histo,"strip_tib_layer2_extmodule%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

  for (Int_t i=9; i<12; i++){
      sprintf(histo,"strip_tib_layer2_intmodule%d_zm;1",i-8);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

 Strip->Print("StripNumOfTIBLayer2Compare.eps");
 Strip->Print("StripNumOfTIBLayer2Compare.gif");

 }
// TIB  3rd Layer
 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTIB_[12];
   TH1* newmeAdcTIB_[12];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"adc_tib_layer3_extmodule%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

  for (Int_t i=3; i<6; i++){
      sprintf(histo,"adc_tib_layer3_intmodule%d_zp;1",i-2);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<9; i++){
      sprintf(histo,"adc_tib_layer3_extmodule%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

  for (Int_t i=9; i<12; i++){
      sprintf(histo,"adc_tib_layer3_intmodule%d_zm;1",i-8);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

 Strip->Print("AdcOfTIBLayer3Compare.eps");
 Strip->Print("AdcOfTIBLayer3Compare.gif");
 }

 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTIB_[12];
   TH1* newmeAdcTIB_[12];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"strip_tib_layer3_extmodule%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

  for (Int_t i=3; i<6; i++){
      sprintf(histo,"strip_tib_layer3_intmodule%d_zp;1",i-2);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<9; i++){
      sprintf(histo,"strip_tib_layer3_extmodule%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

  for (Int_t i=9; i<12; i++){
      sprintf(histo,"strip_tib_layer3_intmodule%d_zm;1",i-8);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

 Strip->Print("StripNumOfTIBLayer3Compare.eps");
 Strip->Print("StripNumOfTIBLayer3Compare.gif");
 }


// TIB  4th Layer
 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTIB_[12];
   TH1* newmeAdcTIB_[12];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"adc_tib_layer4_extmodule%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();
 
  }

  for (Int_t i=3; i<6; i++){
      sprintf(histo,"adc_tib_layer4_intmodule%d_zp;1",i-2);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();
   }
   for (Int_t i=6; i<9; i++){
      sprintf(histo,"adc_tib_layer4_extmodule%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

  for (Int_t i=9; i<12; i++){
      sprintf(histo,"adc_tib_layer4_intmodule%d_zm;1",i-8);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

 Strip->Print("AdcOfTIBLayer4Compare.eps");
 Strip->Print("AdcOfTIBLayer4Compare.gif");
 }

 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTIB_[12];
   TH1* newmeAdcTIB_[12];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"strip_tib_layer4_extmodule%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

  for (Int_t i=3; i<6; i++){
      sprintf(histo,"strip_tib_layer4_intmodule%d_zp;1",i-2);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<9; i++){
      sprintf(histo,"strip_tib_layer4_extmodule%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

  for (Int_t i=9; i<12; i++){
      sprintf(histo,"strip_tib_layer4_intmodule%d_zm;1",i-8);
      rdir->GetObject(histo ,meAdcTIB_[i]);
      sdir->GetObject(histo ,newmeAdcTIB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTIB_[i],rver , "l");
      leg.AddEntry(newmeAdcTIB_[i],cver , "l");
      leg.Draw();

   }

 Strip->Print("StripNumOfTIBLayer4Compare.eps");
 Strip->Print("StripNumOfTIBLayer4Compare.gif");

 }



//////////////////////////////////////
// TOB

 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(2,3);

   TH1* meNdigiTOB_[6];
   TH1* newmeNdigiTOB_[6];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"ndigi_tob_layer_%d_zm;1",i+1);
      rdir->GetObject(histo ,meNdigiTOB_[i]);
      sdir->GetObject(histo ,newmeNdigiTOB_[i]);
      meNdigiTOB_[i];
      newmeNdigiTOB_[i];
      Strip->cd(i+1);
      meNdigiTOB_[i]->SetLineColor(2);
      newmeNdigiTOB_[i]->SetLineColor(4);
      newmeNdigiTOB_[i]->SetLineStyle(2);
      meNdigiTOB_[i]->Draw();
      newmeNdigiTOB_[i]->Draw("sames");
      myPV->PVCompute(meNdigiTOB_[i] , newmeNdigiTOB_[i] , te );
      leg.Clear();
      leg.AddEntry(meNdigiTOB_[i],rver , "l");
      leg.AddEntry(newmeNdigiTOB_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("NdigiTOBCompare_ZM.eps");
   Strip->Print("NdigiTOBCompare_ZM.gif");
 }

 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(2,3);

   TH1* meNdigiTOB_[6];
   TH1* newmeNdigiTOB_[6];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"ndigi_tob_layer_%d_zp;1",i+1);
      rdir->GetObject(histo ,meNdigiTOB_[i]);
      sdir->GetObject(histo ,newmeNdigiTOB_[i]);
      meNdigiTOB_[i];
      newmeNdigiTOB_[i];
      Strip->cd(i+1);
      gPad->SetLogy();
      meNdigiTOB_[i]->SetLineColor(2);
      newmeNdigiTOB_[i]->SetLineColor(4);
      newmeNdigiTOB_[i]->SetLineStyle(2);
      meNdigiTOB_[i]->Draw();
      newmeNdigiTOB_[i]->Draw("sames");
      myPV->PVCompute(meNdigiTOB_[i] , newmeNdigiTOB_[i] , te );
      leg.Clear();
      leg.AddEntry(meNdigiTOB_[i],rver , "l");
      leg.AddEntry(newmeNdigiTOB_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("NdigiTOBCompare_ZP.eps");
   Strip->Print("NdigiTOBCompare_ZP.gif");
 }

//TOB 1st Layer

 if(1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTOB_[12];
   TH1* newmeAdcTOB_[12];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"adc_tob_layer1_module%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<12; i++){
      sprintf(histo,"adc_tob_layer1_module%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTOBLayer1Compare.eps");
   Strip->Print("AdcOfTOBLayer1Compare.gif");
 }

 if(1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTOB_[12];
   TH1* newmeAdcTOB_[12];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"strip_tob_layer1_module%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<12; i++){
      sprintf(histo,"strip_tob_layer1_module%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTOBLayer1Compare.eps");
   Strip->Print("StripNumOfTOBLayer1Compare.gif");
 }
//TOB  2nd Layer
 if(1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTOB_[12];
   TH1* newmeAdcTOB_[12];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"adc_tob_layer2_module%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<12; i++){
      sprintf(histo,"adc_tob_layer2_module%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTOBLayer2Compare.eps");
   Strip->Print("AdcOfTOBLayer2Compare.gif");
 }

 if(1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTOB_[12];
   TH1* newmeAdcTOB_[12];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"strip_tob_layer2_module%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<12; i++){
      sprintf(histo,"strip_tob_layer2_module%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTOBLayer2Compare.eps");
   Strip->Print("StripNumOfTOBLayer2Compare.gif");
 }
//TOB  3rd Layer

 if(1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTOB_[12];
   TH1* newmeAdcTOB_[12];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"adc_tob_layer3_module%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<12; i++){
      sprintf(histo,"adc_tob_layer3_module%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTOBLayer3Compare.eps");
   Strip->Print("AdcOfTOBLayer3Compare.gif");
 }

 if(1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTOB_[12];
   TH1* newmeAdcTOB_[12];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"strip_tob_layer3_module%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<12; i++){
      sprintf(histo,"strip_tob_layer3_module%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTOBLayer3Compare.eps");
   Strip->Print("StripNumOfTOBLayer3Compare.gif");
 }
//TOB  4th Layer

 if(1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTOB_[12];
   TH1* newmeAdcTOB_[12];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"adc_tob_layer4_module%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<12; i++){
      sprintf(histo,"adc_tob_layer4_module%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTOBLayer4Compare.eps");
   Strip->Print("AdcOfTOBLayer4Compare.gif");
 }

 if(1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTOB_[12];
   TH1* newmeAdcTOB_[12];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"strip_tob_layer4_module%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<12; i++){
      sprintf(histo,"strip_tob_layer4_module%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTOBLayer4Compare.eps");
   Strip->Print("StripNumOfTOBLayer4Compare.gif");
 }
//TOB  5th Layer
 if(1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTOB_[12];
   TH1* newmeAdcTOB_[12];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"adc_tob_layer5_module%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<12; i++){
      sprintf(histo,"adc_tob_layer5_module%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTOBLayer5Compare.eps");
   Strip->Print("AdcOfTOBLayer5Compare.gif");
 }

 if(1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTOB_[12];
   TH1* newmeAdcTOB_[12];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"strip_tob_layer5_module%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<12; i++){
      sprintf(histo,"strip_tob_layer5_module%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTOBLayer5Compare.eps");
   Strip->Print("StripNumOfTOBLayer5Compare.gif");
 }
//TOB  6th Layer

 if(1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTOB_[12];
   TH1* newmeAdcTOB_[12];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"adc_tob_layer6_module%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<12; i++){
      sprintf(histo,"adc_tob_layer6_module%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTOBLayer6Compare.eps");
   Strip->Print("AdcOfTOBLayer6Compare.gif");
 }

 if(1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,4);

   TH1* meAdcTOB_[12];
   TH1* newmeAdcTOB_[12];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"strip_tob_layer6_module%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }
   for (Int_t i=6; i<12; i++){
      sprintf(histo,"strip_tob_layer6_module%d_zm;1",i-5);
      rdir->GetObject(histo ,meAdcTOB_[i]);
      sdir->GetObject(histo ,newmeAdcTOB_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTOB_[i],rver , "l");
      leg.AddEntry(newmeAdcTOB_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTOBLayer6Compare.eps");
   Strip->Print("StripNumOfTOBLayer6Compare.gif");
 }
//TID

 if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(1,3);

   TH1* meNdigiTID_[3];
   TH1* newmeNdigiTID_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"ndigi_tid_wheel_%d_zm;1",i+1);
      rdir->GetObject(histo ,meNdigiTID_[i]);
      sdir->GetObject(histo ,newmeNdigiTID_[i]);
      meNdigiTID_[i];
      newmeNdigiTID_[i];
      Strip->cd(i+1);
      gPad->SetLogy();
      meNdigiTID_[i]->SetLineColor(2);
      newmeNdigiTID_[i]->SetLineColor(4);
      newmeNdigiTID_[i]->SetLineStyle(2);
      meNdigiTID_[i]->Draw();
      newmeNdigiTID_[i]->Draw("sames");
      myPV->PVCompute(meNdigiTID_[i] , newmeNdigiTID_[i] , te );
      leg.Clear();
      leg.AddEntry(meNdigiTID_[i],rver , "l");
      leg.AddEntry(newmeNdigiTID_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("NdigiTIDCompare_ZM.eps");
   Strip->Print("NdigiTIDCompare_ZM.gif");
 }

if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(1,3);

   TH1* meNdigiTID_[3];
   TH1* newmeNdigiTID_[3];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"ndigi_tid_wheel_%d_zp;1",i+1);
      rdir->GetObject(histo ,meNdigiTID_[i]);
      sdir->GetObject(histo ,newmeNdigiTID_[i]);
      meNdigiTID_[i];
      newmeNdigiTID_[i];
      Strip->cd(i+1);
      gPad->SetLogy();
      meNdigiTID_[i]->SetLineColor(2);
      newmeNdigiTID_[i]->SetLineColor(4);
      newmeNdigiTID_[i]->SetLineStyle(2);
      meNdigiTID_[i]->Draw();
      newmeNdigiTID_[i]->Draw("sames");
      myPV->PVCompute(meNdigiTID_[i] , newmeNdigiTID_[i] , te );
      leg.Clear();
      leg.AddEntry(meNdigiTID_[i],rver , "l");
      leg.AddEntry(newmeNdigiTID_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("NdigiTIDCompare_ZP.eps");
   Strip->Print("NdigiTIDCompare_ZP.gif");
 }

// TID  1 st Wheel
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,800);
   Strip->Divide(3,2);
   TH1* meAdcTID_[6];
   TH1* newmeAdcTID_[6];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"adc_tid_wheel1_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTID_[i]);
      sdir->GetObject(histo ,newmeAdcTID_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTID_[i],rver , "l");
      leg.AddEntry(newmeAdcTID_[i],cver , "l");
      leg.Draw();

   }

   for (Int_t i=3; i<6; i++){
      sprintf(histo,"adc_tid_wheel1_ring%d_zm;1",i-2);
      rdir->GetObject(histo ,meAdcTID_[i]);
      sdir->GetObject(histo ,newmeAdcTID_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTID_[i],rver , "l");
      leg.AddEntry(newmeAdcTID_[i],cver , "l");
      leg.Draw();

   }
   Strip->Print("AdcOfTIDWheel1Compare.eps");
   Strip->Print("AdcOfTIDWheel1Compare.gif");
}

if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,800);
   Strip->Divide(3,2);
   TH1* meAdcTID_[6];
   TH1* newmeAdcTID_[6];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"strip_tid_wheel1_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTID_[i]);
      sdir->GetObject(histo ,newmeAdcTID_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTID_[i],rver , "l");
      leg.AddEntry(newmeAdcTID_[i],cver , "l");
      leg.Draw();

   }

   for (Int_t i=3; i<6; i++){
      sprintf(histo,"strip_tid_wheel1_ring%d_zm;1",i-2);
      rdir->GetObject(histo ,meAdcTID_[i]);
      sdir->GetObject(histo ,newmeAdcTID_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTID_[i],rver , "l");
      leg.AddEntry(newmeAdcTID_[i],cver , "l");
      leg.Draw();

   }
   Strip->Print("StripNumOfTIDWheel1Compare.eps");
   Strip->Print("StripNumOfTIDWheel1Compare.gif");
}
//TID 2nd  Wheel
if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,800);
   Strip->Divide(3,2);
   TH1* meAdcTID_[6];
   TH1* newmeAdcTID_[6];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"adc_tid_wheel2_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTID_[i]);
      sdir->GetObject(histo ,newmeAdcTID_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTID_[i],rver , "l");
      leg.AddEntry(newmeAdcTID_[i],cver , "l");
      leg.Draw();

   }

   for (Int_t i=3; i<6; i++){
      sprintf(histo,"adc_tid_wheel2_ring%d_zm;1",i-2);
      rdir->GetObject(histo ,meAdcTID_[i]);
      sdir->GetObject(histo ,newmeAdcTID_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTID_[i],rver , "l");
      leg.AddEntry(newmeAdcTID_[i],cver , "l");
      leg.Draw();

   }
   Strip->Print("AdcOfTIDWheel2Compare.eps");
   Strip->Print("AdcOfTIDWheel2Compare.gif");
}

if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,800);
   Strip->Divide(3,2);
   TH1* meAdcTID_[6];
   TH1* newmeAdcTID_[6];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"strip_tid_wheel2_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTID_[i]);
      sdir->GetObject(histo ,newmeAdcTID_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTID_[i],rver , "l");
      leg.AddEntry(newmeAdcTID_[i],cver , "l");
      leg.Draw();

   }

   for (Int_t i=3; i<6; i++){
      sprintf(histo,"strip_tid_wheel2_ring%d_zm;1",i-2);
      rdir->GetObject(histo ,meAdcTID_[i]);
      sdir->GetObject(histo ,newmeAdcTID_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTID_[i],rver , "l");
      leg.AddEntry(newmeAdcTID_[i],cver , "l");
      leg.Draw();

   }
   Strip->Print("StripNumOfTIDWheel2Compare.eps");
   Strip->Print("StripNumOfTIDWheel2Compare.gif");
}
//TID 3rd  Wheel
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,800);
   Strip->Divide(3,2);
   TH1* meAdcTID_[6];
   TH1* newmeAdcTID_[6];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"adc_tid_wheel3_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTID_[i]);
      sdir->GetObject(histo ,newmeAdcTID_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTID_[i],rver , "l");
      leg.AddEntry(newmeAdcTID_[i],cver , "l");
      leg.Draw();
   }

   for (Int_t i=3; i<6; i++){
      sprintf(histo,"adc_tid_wheel3_ring%d_zm;1",i-2);
      rdir->GetObject(histo ,meAdcTID_[i]);
      sdir->GetObject(histo ,newmeAdcTID_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTID_[i],rver , "l");
      leg.AddEntry(newmeAdcTID_[i],cver , "l");
      leg.Draw();
   }
   Strip->Print("AdcOfTIDWheel3Compare.eps");
   Strip->Print("AdcOfTIDWheel3Compare.gif");
}

if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",1000,800);
   Strip->Divide(3,2);
   TH1* meAdcTID_[6];
   TH1* newmeAdcTID_[6];

   for (Int_t i=0; i<3; i++){
      sprintf(histo,"strip_tid_wheel3_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTID_[i]);
      sdir->GetObject(histo ,newmeAdcTID_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTID_[i],rver , "l");
      leg.AddEntry(newmeAdcTID_[i],cver , "l");
      leg.Draw();
   }

   for (Int_t i=3; i<6; i++){
      sprintf(histo,"strip_tid_wheel3_ring%d_zm;1",i-2);
      rdir->GetObject(histo ,meAdcTID_[i]);
      sdir->GetObject(histo ,newmeAdcTID_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTID_[i],rver , "l");
      leg.AddEntry(newmeAdcTID_[i],cver , "l");
      leg.Draw();
   }
   Strip->Print("StripNumOfTIDWheel3Compare.eps");
   Strip->Print("StripNumOfTIDWheel3Compare.gif");
}
//IEC
if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,3);

   TH1* meNdigiTEC_[9];
   TH1* newmeNdigiTEC_[9];

   for (Int_t i=0; i<9; i++){
      sprintf(histo,"ndigi_tec_wheel_%d_zm;1",i+1);
      rdir->GetObject(histo ,meNdigiTEC_[i]);
      sdir->GetObject(histo ,newmeNdigiTEC_[i]);
      meNdigiTEC_[i];
      newmeNdigiTEC_[i];
      Strip->cd(i+1);
      gPad->SetLogy();
      meNdigiTEC_[i]->SetLineColor(2);
      newmeNdigiTEC_[i]->SetLineColor(4);
      newmeNdigiTEC_[i]->SetLineStyle(2);
      meNdigiTEC_[i]->Draw();
      newmeNdigiTEC_[i]->Draw("sames");
      myPV->PVCompute(meNdigiTEC_[i] , newmeNdigiTEC_[i] , te );
      leg.Clear();
      leg.AddEntry(meNdigiTEC_[i],rver , "l");
      leg.AddEntry(newmeNdigiTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("NdigiTECCompare_ZM.eps");
   Strip->Print("NdigiTECCompare_ZM.gif");
 }
if (1) {
  TCanvas * Strip = new TCanvas("Strip","Strip",1000,1000);
   Strip->Divide(3,3);

   TH1* meNdigiTEC_[9];
   TH1* newmeNdigiTEC_[9];

   for (Int_t i=0; i<9; i++){
      sprintf(histo,"ndigi_tec_wheel_%d_zp;1",i+1);
      rdir->GetObject(histo ,meNdigiTEC_[i]);
      sdir->GetObject(histo ,newmeNdigiTEC_[i]);
      meNdigiTEC_[i];
      newmeNdigiTEC_[i];
      Strip->cd(i+1);
      gPad->SetLogy();
      meNdigiTEC_[i]->SetLineColor(2);
      newmeNdigiTEC_[i]->SetLineColor(4);
      newmeNdigiTEC_[i]->SetLineStyle(2);
      meNdigiTEC_[i]->Draw();
      newmeNdigiTEC_[i]->Draw("sames");
      myPV->PVCompute(meNdigiTEC_[i] , newmeNdigiTEC_[i] , te );
      leg.Clear();
      leg.AddEntry(meNdigiTEC_[i],rver , "l");
      leg.AddEntry(newmeNdigiTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("NdigiTECCompare_ZP.eps");
   Strip->Print("NdigiTECCompare_ZP.gif");
 }

//TEC 1st Wheel in ZMinus Side
if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,4);
   TH1* meAdcTEC_[7];
   TH1* newmeAdcTEC_[7];

   for (Int_t i=0; i<7; i++){
      sprintf(histo,"adc_tec_wheel1_ring%d_zm;1",i+1);
      rdir->GetObject(histo ,meAdcTEC_[i]);
      sdir->GetObject(histo ,newmeAdcTEC_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTEC_[i],rver , "l");
      leg.AddEntry(newmeAdcTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTECWheel1Compare_ZM.eps");
   Strip->Print("AdcOfTECWheel1Compare_ZM.gif");
} 
if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,4);
   TH1* meStripTEC_[7];
   TH1* newmeStripTEC_[7];
 
   for (Int_t i=0; i<7; i++){
      sprintf(histo,"strip_tec_wheel1_ring%d_zm;1",i+1);
      rdir->GetObject(histo ,meStripTEC_[i]);
      sdir->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1);
      gPad->SetLogy(); 
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
      leg.Clear();
      leg.AddEntry(meStripTEC_[i],rver , "l");
      leg.AddEntry(newmeStripTEC_[i],cver , "l");
      leg.Draw();
   }

   Strip->Print("StripNumOfTECWheel1Compare_ZM.eps");
   Strip->Print("StripNumOfTECWheel1Compare_ZM.gif");
}
//TEC 2nd Wheel in ZMinus Side
if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,4);
   TH1* meAdcTEC_[7];
   TH1* newmeAdcTEC_[7];

   for (Int_t i=0; i<7; i++){
      sprintf(histo,"adc_tec_wheel2_ring%d_zm;1",i+1);
      rdir->GetObject(histo ,meAdcTEC_[i]);
      sdir->GetObject(histo ,newmeAdcTEC_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTEC_[i],rver , "l");
      leg.AddEntry(newmeAdcTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTECWheel2Compare_ZM.eps");
   Strip->Print("AdcOfTECWheel2Compare_ZM.gif");
} 
if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,4);
   TH1* meStripTEC_[7];
   TH1* newmeStripTEC_[7];
 
   for (Int_t i=0; i<7; i++){
      sprintf(histo,"strip_tec_wheel2_ring%d_zm;1",i+1);
      rdir->GetObject(histo ,meStripTEC_[i]);
      sdir->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1); 
      gPad->SetLogy();
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
      leg.Clear();
      leg.AddEntry(meStripTEC_[i],rver , "l");
      leg.AddEntry(newmeStripTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTECWheel2Compare_ZM.eps");
   Strip->Print("StripNumOfTECWheel2Compare_ZM.gif");
}

//TEC 3rd Wheel in ZMinus Side
if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,4);
   TH1* meAdcTEC_[7];
   TH1* newmeAdcTEC_[7];

   for (Int_t i=0; i<7; i++){
      sprintf(histo,"adc_tec_wheel3_ring%d_zm;1",i+1);
      rdir->GetObject(histo ,meAdcTEC_[i]);
      sdir->GetObject(histo ,newmeAdcTEC_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTEC_[i],rver , "l");
      leg.AddEntry(newmeAdcTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTECWheel3Compare_ZM.eps");
   Strip->Print("AdcOfTECWheel3Compare_ZM.gif");
} 
if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,4);
   TH1* meStripTEC_[7];
   TH1* newmeStripTEC_[7];
 
   for (Int_t i=0; i<7; i++){
      sprintf(histo,"strip_tec_wheel3_ring%d_zm;1",i+1);
      rdir->GetObject(histo ,meStripTEC_[i]);
      sdir->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1); 
      gPad->SetLogy();
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
      leg.Clear();
      leg.AddEntry(meStripTEC_[i],rver , "l");
      leg.AddEntry(newmeStripTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTECWheel3Compare_ZM.eps");
   Strip->Print("StripNumOfTECWheel3Compare_ZM.gif");
}

//TEC 4th Wheel in ZMinus Side
if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meAdcTEC_[6];
   TH1* newmeAdcTEC_[6];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"adc_tec_wheel4_ring%d_zm;1",i+1);
      rdir->GetObject(histo ,meAdcTEC_[i]);
      sdir->GetObject(histo ,newmeAdcTEC_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTEC_[i],rver , "l");
      leg.AddEntry(newmeAdcTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTECWheel4Compare_ZM.eps");
   Strip->Print("AdcOfTECWheel4Compare_ZM.gif");
} 
if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meStripTEC_[6];
   TH1* newmeStripTEC_[6];
 
   for (Int_t i=0; i<6; i++){
      sprintf(histo,"strip_tec_wheel4_ring%d_zm;1",i+1);
      rdir->GetObject(histo ,meStripTEC_[i]);
      sdir->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1); 
      gPad->SetLogy();
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
      leg.Clear();
      leg.AddEntry(meStripTEC_[i],rver , "l");
      leg.AddEntry(newmeStripTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTECWheel4Compare_ZM.eps");
   Strip->Print("StripNumOfTECWheel4Compare_ZM.gif");
}

//TEC 5th Wheel in ZMinus Side
if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meAdcTEC_[6];
   TH1* newmeAdcTEC_[6];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"adc_tec_wheel5_ring%d_zm;1",i+1);
      rdir->GetObject(histo ,meAdcTEC_[i]);
      sdir->GetObject(histo ,newmeAdcTEC_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTEC_[i],rver , "l");
      leg.AddEntry(newmeAdcTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTECWheel5Compare_ZM.eps");
   Strip->Print("AdcOfTECWheel5Compare_ZM.gif");
} 
if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meStripTEC_[6];
   TH1* newmeStripTEC_[6];
 
   for (Int_t i=0; i<6; i++){
      sprintf(histo,"strip_tec_wheel5_ring%d_zm;1",i+1);
      rdir->GetObject(histo ,meStripTEC_[i]);
      sdir->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1); 
      gPad->SetLogy();
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
      leg.Clear();
      leg.AddEntry(meStripTEC_[i],rver , "l");
      leg.AddEntry(newmeStripTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTECWheel5Compare_ZM.eps");
   Strip->Print("StripNumOfTECWheel5Compare_ZM.gif");
}

//TEC 6th Wheel in ZMinus Side
if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meAdcTEC_[6];
   TH1* newmeAdcTEC_[6];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"adc_tec_wheel6_ring%d_zm;1",i+1);
      rdir->GetObject(histo ,meAdcTEC_[i]);
      sdir->GetObject(histo ,newmeAdcTEC_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTEC_[i],rver , "l");
      leg.AddEntry(newmeAdcTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTECWheel6Compare_ZM.eps");
   Strip->Print("AdcOfTECWheel6Compare_ZM.gif");
} 
if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meStripTEC_[6];
   TH1* newmeStripTEC_[6];
 
   for (Int_t i=0; i<6; i++){
      sprintf(histo,"strip_tec_wheel6_ring%d_zm;1",i+1);
      rdir->GetObject(histo ,meStripTEC_[i]);
      sdir->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1); 
      gPad->SetLogy();
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
      leg.Clear();
      leg.AddEntry(meStripTEC_[i],rver , "l");
      leg.AddEntry(newmeStripTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTECWheel6Compare_ZM.eps");
   Strip->Print("StripNumOfTECWheel6Compare_ZM.gif");
}

//TEC 7th Wheel in ZMinus Side
if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meAdcTEC_[5];
   TH1* newmeAdcTEC_[5];

   for (Int_t i=0; i<5; i++){
      sprintf(histo,"adc_tec_wheel7_ring%d_zm;1",i+1);
      rdir->GetObject(histo ,meAdcTEC_[i]);
      sdir->GetObject(histo ,newmeAdcTEC_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTEC_[i],rver , "l");
      leg.AddEntry(newmeAdcTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTECWheel7Compare_ZM.eps");
   Strip->Print("AdcOfTECWheel7Compare_ZM.gif");
} 
if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meStripTEC_[5];
   TH1* newmeStripTEC_[5];
 
   for (Int_t i=0; i<5; i++){
      sprintf(histo,"strip_tec_wheel7_ring%d_zm;1",i+1);
      rdir->GetObject(histo ,meStripTEC_[i]);
      sdir->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1); 
      gPad->SetLogy();
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
      leg.Clear();
      leg.AddEntry(meStripTEC_[i],rver , "l");
      leg.AddEntry(newmeStripTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTECWheel7Compare_ZM.eps");
   Strip->Print("StripNumOfTECWheel7Compare_ZM.gif");
}

//TEC 8th Wheel in ZMinus Side
if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meAdcTEC_[5];
   TH1* newmeAdcTEC_[5];

   for (Int_t i=0; i<5; i++){
      sprintf(histo,"adc_tec_wheel8_ring%d_zm;1",i+1);
      rdir->GetObject(histo ,meAdcTEC_[i]);
      sdir->GetObject(histo ,newmeAdcTEC_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTEC_[i],rver , "l");
      leg.AddEntry(newmeAdcTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTECWheel8Compare_ZM.eps");
   Strip->Print("AdcOfTECWheel8Compare_ZM.gif");
} 
if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meStripTEC_[5];
   TH1* newmeStripTEC_[5];
 
   for (Int_t i=0; i<5; i++){
      sprintf(histo,"strip_tec_wheel8_ring%d_zm;1",i+1);
      rdir->GetObject(histo ,meStripTEC_[i]);
      sdir->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1); 
      gPad->SetLogy();
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
      leg.Clear();
      leg.AddEntry(meStripTEC_[i],rver , "l");
      leg.AddEntry(newmeStripTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTECWheel8Compare_ZM.eps");
   Strip->Print("StripNumOfTECWheel8Compare_ZM.gif");
 }

//TEC 9th Wheel in ZMinus Side
if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,2);
   TH1* meAdcTEC_[4];
   TH1* newmeAdcTEC_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"adc_tec_wheel9_ring%d_zm;1",i+1);
      rdir->GetObject(histo ,meAdcTEC_[i]);
      sdir->GetObject(histo ,newmeAdcTEC_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTEC_[i],rver , "l");
      leg.AddEntry(newmeAdcTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTECWheel9Compare_ZM.eps");
   Strip->Print("AdcOfTECWheel9Compare_ZM.gif");
   } 
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,2);
   TH1* meStripTEC_[4];
   TH1* newmeStripTEC_[4];
 
   for (Int_t i=0; i<4; i++){
      sprintf(histo,"strip_tec_wheel9_ring%d_zm;1",i+1);
      rdir->GetObject(histo ,meStripTEC_[i]);
      sdir->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1); 
      gPad->SetLogy(); 
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
      leg.Clear();
      leg.AddEntry(meStripTEC_[i],rver , "l");
      leg.AddEntry(newmeStripTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTECWheel9Compare_ZM.eps");
   Strip->Print("StripNumOfTECWheel9Compare_ZM.gif");
   }

//TEC 1st Wheel in ZPlus  Side
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,4);
   TH1* meAdcTEC_[7];
   TH1* newmeAdcTEC_[7];

   for (Int_t i=0; i<7; i++){
      sprintf(histo,"adc_tec_wheel1_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTEC_[i]);
      sdir->GetObject(histo ,newmeAdcTEC_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTEC_[i],rver , "l");
      leg.AddEntry(newmeAdcTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTECWheel1Compare_ZP.eps");
   Strip->Print("AdcOfTECWheel1Compare_ZP.gif");
   } 
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,4);
   TH1* meStripTEC_[7];
   TH1* newmeStripTEC_[7];
 
   for (Int_t i=0; i<7; i++){
      sprintf(histo,"strip_tec_wheel1_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meStripTEC_[i]);
      sdir->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1); 
      gPad->SetLogy();
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
      leg.Clear();
      leg.AddEntry(meStripTEC_[i],rver , "l");
      leg.AddEntry(newmeStripTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTECWheel1Compare_ZP.eps");
   Strip->Print("StripNumOfTECWheel1Compare_ZP.gif");
   }
//TEC 2nd Wheel in ZPlus  Side
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,4);
   TH1* meAdcTEC_[7];
   TH1* newmeAdcTEC_[7];

   for (Int_t i=0; i<7; i++){
      sprintf(histo,"adc_tec_wheel2_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTEC_[i]);
      sdir->GetObject(histo ,newmeAdcTEC_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTEC_[i],rver , "l");
      leg.AddEntry(newmeAdcTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTECWheel2Compare_ZP.eps");
   Strip->Print("AdcOfTECWheel2Compare_ZP.gif");
   } 
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,4);
   TH1* meStripTEC_[7];
   TH1* newmeStripTEC_[7];
 
   for (Int_t i=0; i<7; i++){
      sprintf(histo,"strip_tec_wheel2_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meStripTEC_[i]);
      sdir->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1); 
      gPad->SetLogy();
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
      leg.Clear();
      leg.AddEntry(meStripTEC_[i],rver , "l");
      leg.AddEntry(newmeStripTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTECWheel2Compare_ZP.eps");
   Strip->Print("StripNumOfTECWheel2Compare_ZP.gif");
   }

//TEC 3rd Wheel in ZPlus  Side
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,4);
   TH1* meAdcTEC_[7];
   TH1* newmeAdcTEC_[7];

   for (Int_t i=0; i<7; i++){
      sprintf(histo,"adc_tec_wheel3_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTEC_[i]);
      sdir->GetObject(histo ,newmeAdcTEC_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTEC_[i],rver , "l");
      leg.AddEntry(newmeAdcTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTECWheel3Compare_ZP.eps");
   Strip->Print("AdcOfTECWheel3Compare_ZP.gif");
   } 
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,4);
   TH1* meStripTEC_[7];
   TH1* newmeStripTEC_[7];
 
   for (Int_t i=0; i<7; i++){
      sprintf(histo,"strip_tec_wheel3_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meStripTEC_[i]);
      sdir->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1); 
      gPad->SetLogy();
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
      leg.Clear();
      leg.AddEntry(meStripTEC_[i],rver , "l");
      leg.AddEntry(newmeStripTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTECWheel3Compare_ZP.eps");
   Strip->Print("StripNumOfTECWheel3Compare_ZP.gif");
   }

//TEC 4th Wheel in ZPlus  Side
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meAdcTEC_[6];
   TH1* newmeAdcTEC_[6];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"adc_tec_wheel4_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTEC_[i]);
      sdir->GetObject(histo ,newmeAdcTEC_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTEC_[i],rver , "l");
      leg.AddEntry(newmeAdcTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTECWheel4Compare_ZP.eps");
   Strip->Print("AdcOfTECWheel4Compare_ZP.gif");
   } 
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meStripTEC_[6];
   TH1* newmeStripTEC_[6];
 
   for (Int_t i=0; i<6; i++){
      sprintf(histo,"strip_tec_wheel4_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meStripTEC_[i]);
      sdir->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1); 
      gPad->SetLogy();
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
      leg.Clear();
      leg.AddEntry(meStripTEC_[i],rver , "l");
      leg.AddEntry(newmeStripTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTECWheel4Compare_ZP.eps");
   Strip->Print("StripNumOfTECWheel4Compare_ZP.gif");
   }

//TEC 5th Wheel in ZPlus  Side
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meAdcTEC_[6];
   TH1* newmeAdcTEC_[6];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"adc_tec_wheel5_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTEC_[i]);
      sdir->GetObject(histo ,newmeAdcTEC_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTEC_[i],rver , "l");
      leg.AddEntry(newmeAdcTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTECWheel5Compare_ZP.eps");
   Strip->Print("AdcOfTECWheel5Compare_ZP.gif");
   } 
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meStripTEC_[6];
   TH1* newmeStripTEC_[6];
 
   for (Int_t i=0; i<6; i++){
      sprintf(histo,"strip_tec_wheel5_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meStripTEC_[i]);
      sdir->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1); 
      gPad->SetLogy();
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
      leg.Clear();
      leg.AddEntry(meStripTEC_[i],rver , "l");
      leg.AddEntry(newmeStripTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTECWheel5Compare_ZP.eps");
   Strip->Print("StripNumOfTECWheel5Compare_ZP.gif");
   }

//TEC 6th WPlus in ZMinus Side
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meAdcTEC_[6];
   TH1* newmeAdcTEC_[6];

   for (Int_t i=0; i<6; i++){
      sprintf(histo,"adc_tec_wheel6_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTEC_[i]);
      sdir->GetObject(histo ,newmeAdcTEC_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTEC_[i],rver , "l");
      leg.AddEntry(newmeAdcTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTECWheel6Compare_ZP.eps");
   Strip->Print("AdcOfTECWheel6Compare_ZP.gif");
   } 
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meStripTEC_[6];
   TH1* newmeStripTEC_[6];
 
   for (Int_t i=0; i<6; i++){
      sprintf(histo,"strip_tec_wheel6_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meStripTEC_[i]);
      sdir->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1); 
      gPad->SetLogy();
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
      leg.Clear();
      leg.AddEntry(meStripTEC_[i],rver , "l");
      leg.AddEntry(newmeStripTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTECWheel6Compare_ZP.eps");
   Strip->Print("StripNumOfTECWheel6Compare_ZP.gif");
   }

//TEC 7th Wheel in ZPlus  Side
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meAdcTEC_[5];
   TH1* newmeAdcTEC_[5];

   for (Int_t i=0; i<5; i++){
      sprintf(histo,"adc_tec_wheel7_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTEC_[i]);
      sdir->GetObject(histo ,newmeAdcTEC_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTEC_[i],rver , "l");
      leg.AddEntry(newmeAdcTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTECWheel7Compare_ZP.eps");
   Strip->Print("AdcOfTECWheel7Compare_ZP.gif");
   } 
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meStripTEC_[5];
   TH1* newmeStripTEC_[5];
 
   for (Int_t i=0; i<5; i++){
      sprintf(histo,"strip_tec_wheel7_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meStripTEC_[i]);
      sdir->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1); 
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
      leg.Clear();
      leg.AddEntry(meStripTEC_[i],rver , "l");
      leg.AddEntry(newmeStripTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTECWheel7Compare_ZP.eps");
   Strip->Print("StripNumOfTECWheel7Compare_ZP.gif");
   }

//TEC 8th Wheel in ZPlus  Side
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meAdcTEC_[5];
   TH1* newmeAdcTEC_[5];

   for (Int_t i=0; i<5; i++){
      sprintf(histo,"adc_tec_wheel8_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTEC_[i]);
      sdir->GetObject(histo ,newmeAdcTEC_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTEC_[i],rver , "l");
      leg.AddEntry(newmeAdcTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTECWheel8Compare_ZP.eps");
   Strip->Print("AdcOfTECWheel8Compare_ZP.gif");
   } 
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,3);
   TH1* meStripTEC_[5];
   TH1* newmeStripTEC_[5];
 
   for (Int_t i=0; i<5; i++){
      sprintf(histo,"strip_tec_wheel8_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meStripTEC_[i]);
      sdir->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1); 
      gPad->SetLogy();
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
      leg.Clear();
      leg.AddEntry(meStripTEC_[i],rver , "l");
      leg.AddEntry(newmeStripTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTECWheel8Compare_ZP.eps");
   Strip->Print("StripNumOfTECWheel8Compare_ZP.gif");
   }

//TEC 9th Wheel in ZPLUS  Side
   if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,2);
   TH1* meAdcTEC_[4];
   TH1* newmeAdcTEC_[4];

   for (Int_t i=0; i<4; i++){
      sprintf(histo,"adc_tec_wheel9_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meAdcTEC_[i]);
      sdir->GetObject(histo ,newmeAdcTEC_[i]);
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
      leg.Clear();
      leg.AddEntry(meAdcTEC_[i],rver , "l");
      leg.AddEntry(newmeAdcTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("AdcOfTECWheel9Compare_ZP.eps");
   Strip->Print("AdcOfTECWheel9Compare_ZP.gif");
   } 
 if (1) {
   TCanvas * Strip = new TCanvas("Strip","Strip",800,1000);
   Strip->Divide(2,2);
   TH1* meStripTEC_[4];
   TH1* newmeStripTEC_[4];
 
   for (Int_t i=0; i<4; i++){
      sprintf(histo,"strip_tec_wheel9_ring%d_zp;1",i+1);
      rdir->GetObject(histo ,meStripTEC_[i]);
      sdir->GetObject(histo ,newmeStripTEC_[i]);
      meStripTEC_[i];
      newmeStripTEC_[i];
      Strip->cd(i+1); 
      gPad->SetLogy();
      meStripTEC_[i]->SetLineColor(2);
      newmeStripTEC_[i]->SetLineColor(4);
      newmeStripTEC_[i]->SetLineStyle(2);
      meStripTEC_[i]->Draw();
      newmeStripTEC_[i]->Draw("sames");
      myPV->PVCompute(meStripTEC_[i],newmeStripTEC_[i],te);
      leg.Clear();
      leg.AddEntry(meStripTEC_[i],rver , "l");
      leg.AddEntry(newmeStripTEC_[i],cver , "l");
      leg.Draw();

   }

   Strip->Print("StripNumOfTECWheel9Compare_ZP.eps");
   Strip->Print("StripNumOfTECWheel9Compare_ZP.gif");
 }
}


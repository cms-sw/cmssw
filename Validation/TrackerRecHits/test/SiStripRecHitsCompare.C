void SiStripRecHitsCompare()
{

 gROOT ->Reset();
 char*  sfilename = "sistriprechitshisto.root";
 char*  rfilename = "../sistriprechitshisto.root";

 delete gROOT->GetListOfFiles()->FindObject(rfilename);
 delete gROOT->GetListOfFiles()->FindObject(sfilename); 

 TText* te = new TText();
 TFile * rfile = new TFile(rfilename);
 TDirectory * rdir=gDirectory; 
 TFile * sfile = new TFile(sfilename);
 TDirectory * sdir=gDirectory; 

 if(rfile->cd("DQMData/Run 1/TrackerRecHitsV"))rfile->cd("DQMData/Run 1/TrackerRecHitsV/Run summary/TrackerRecHits/Strip");
 else rfile->cd("DQMData/TrackerRecHitsV/TrackerRecHits/Strip");
 rdir=gDirectory;

 if(sfile->cd("DQMData/Run 1/TrackerRecHitsV"))sfile->cd("DQMData/Run 1/TrackerRecHitsV/Run summary/TrackerRecHits/Strip");
 else sfile->cd("DQMData/TrackerRecHitsV/TrackerRecHits/Strip");
 sdir=gDirectory; 
 
  TLegend leg(0.3, 0.80, 0.55, 0.90);
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



 Char_t histo[200];

 gROOT->ProcessLine(".x HistoCompare.C");
 HistoCompare * myPV = new HistoCompare();

 TCanvas *Strip0;
 TCanvas *Strip1;
 TCanvas *Strip2;
 TCanvas *Strip3;
 TCanvas *Strip4;
 TCanvas *Strip5;
 TCanvas *Strip6;
 TCanvas *Strip7;
 TCanvas *Strip8;
 TCanvas *Strip9;
 TCanvas *Strip10;
 TCanvas *Strip11;
 TCanvas *Strip12;
 TCanvas *Strip13;
 TCanvas *Strip14;
 TCanvas *Strip15;
 TCanvas *Strip16;
 TCanvas *Strip17;
 TCanvas *Strip18;
 TCanvas *Strip19;
 TCanvas *Strip20;
 TCanvas *Strip21;
 TCanvas *Strip22;
 TCanvas *Strip23;
 TCanvas *Strip24;
 TCanvas *Strip25;
 TCanvas *Strip26;
 TCanvas *Strip27;
 TCanvas *Strip28;
 TCanvas *Strip29;
 TCanvas *Strip30;
 TCanvas *Strip31;
 TCanvas *Strip32;
 TCanvas *Strip33;
 TCanvas *Strip34;

 //=============================================================== 
// TIB
 
 TH1F* adctib[6];
 TH1F* pulltib[6];
 TH1F* nstptib[6];
 TH1F* postib[6];
 TH1F* errxtib[6];
 TH1F* restib[6];
 TH1F* chi2tib[6];
 TH1F* matchedtib[12];

 TH1F* newadctib[6];
 TH1F* newpulltib[6];
 TH1F* newnstptib[6];
 TH1F* newpostib[6];
 TH1F* newerrxtib[6];
 TH1F* newrestib[6];
 TH1F* newchi2tib[6];
 TH1F* newmatchedtib[12];
 
 rdir->GetObject("TIB/Adc_rphi_layer1tib",adctib[0]);
 rdir->GetObject("TIB/Adc_rphi_layer2tib",adctib[1]);
 rdir->GetObject("TIB/Adc_rphi_layer3tib",adctib[2]);
 rdir->GetObject("TIB/Adc_rphi_layer4tib",adctib[3]);
 rdir->GetObject("TIB/Adc_sas_layer1tib",adctib[4]);
 rdir->GetObject("TIB/Adc_sas_layer2tib",adctib[5]);
 sdir->GetObject("TIB/Adc_rphi_layer1tib",newadctib[0]);
 sdir->GetObject("TIB/Adc_rphi_layer2tib",newadctib[1]);
 sdir->GetObject("TIB/Adc_rphi_layer3tib",newadctib[2]);
 sdir->GetObject("TIB/Adc_rphi_layer4tib",newadctib[3]);
 sdir->GetObject("TIB/Adc_sas_layer1tib",newadctib[4]);
 sdir->GetObject("TIB/Adc_sas_layer2tib",newadctib[5]);
 
 Strip0 = new TCanvas("Strip0","Strip0",800,500);
 Strip0->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   Strip0->cd(i+1);
   adctib[i]->SetLineColor(2);
   newadctib[i]->SetLineColor(4);
   newadctib[i]->SetLineStyle(2);
   adctib[i]->Sumw2();
   newadctib[i]->Sumw2();
   adctib[i]->SetNormFactor(1.0);
   newadctib[i]->SetNormFactor(1.0);
   adctib[i]->Draw("h");
   newadctib[i]->Draw("sameh");
   myPV->PVCompute(adctib[i] , newadctib[i] , te );
    leg.Clear();
   leg.AddEntry(adctib[i],rver , "l");
   leg.AddEntry(newadctib[i],cver , "l");
   leg.Draw();
 }
 
 Strip0->Print("AdcTIBCompare.eps");
 Strip0->Print("AdcTIBCompare.gif");
 

 rdir->GetObject("TIB/Nstp_rphi_layer1tib",nstptib[0]);
 rdir->GetObject("TIB/Nstp_rphi_layer2tib",nstptib[1]);
 rdir->GetObject("TIB/Nstp_rphi_layer3tib",nstptib[2]);
 rdir->GetObject("TIB/Nstp_rphi_layer4tib",nstptib[3]);
 rdir->GetObject("TIB/Nstp_sas_layer1tib",nstptib[4]);
 rdir->GetObject("TIB/Nstp_sas_layer2tib",nstptib[5]);
 sdir->GetObject("TIB/Nstp_rphi_layer1tib",newnstptib[0]);
 sdir->GetObject("TIB/Nstp_rphi_layer2tib",newnstptib[1]);
 sdir->GetObject("TIB/Nstp_rphi_layer3tib",newnstptib[2]);
 sdir->GetObject("TIB/Nstp_rphi_layer4tib",newnstptib[3]);
 sdir->GetObject("TIB/Nstp_sas_layer1tib",newnstptib[4]);
 sdir->GetObject("TIB/Nstp_sas_layer2tib",newnstptib[5]);

 Strip1 = new TCanvas("Strip1","Strip1",800,500);
 Strip1->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   Strip1->cd(i+1);
   nstptib[i]->SetLineColor(2);
   newnstptib[i]->SetLineColor(4);
   newnstptib[i]->SetLineStyle(2);
   nstptib[i]->Sumw2();
   newnstptib[i]->Sumw2();
   nstptib[i]->SetNormFactor(1.0);
   newnstptib[i]->SetNormFactor(1.0);
   nstptib[i]->Draw("h");
   newnstptib[i]->Draw("sameh");
   myPV->PVCompute(nstptib[i] , newnstptib[i] , te );
   //   leg.Clear();
   leg.AddEntry(nstptib[i],rver , "l");
   leg.AddEntry(newnstptib[i],cver , "l");
   leg.Draw();
 }
 
 Strip1->Print("NstpTIBCompare.eps");
 Strip1->Print("NstpTIBCompare.gif");

 rdir->GetObject("TIB/Posx_rphi_layer1tib",postib[0]);
 rdir->GetObject("TIB/Posx_rphi_layer2tib",postib[1]);
 rdir->GetObject("TIB/Posx_rphi_layer3tib",postib[2]);
 rdir->GetObject("TIB/Posx_rphi_layer4tib",postib[3]);
 rdir->GetObject("TIB/Posx_sas_layer1tib",postib[4]);
 rdir->GetObject("TIB/Posx_sas_layer2tib",postib[5]);
 sdir->GetObject("TIB/Posx_rphi_layer1tib",newpostib[0]);
 sdir->GetObject("TIB/Posx_rphi_layer2tib",newpostib[1]);
 sdir->GetObject("TIB/Posx_rphi_layer3tib",newpostib[2]);
 sdir->GetObject("TIB/Posx_rphi_layer4tib",newpostib[3]);
 sdir->GetObject("TIB/Posx_sas_layer1tib",newpostib[4]);
 sdir->GetObject("TIB/Posx_sas_layer2tib",newpostib[5]);

 Strip2 = new TCanvas("Strip2","Strip2",800,500);
  Strip2->Divide(2,3);
  for (Int_t i=0; i<6; i++) {
    Strip2->cd(i+1);
    postib[i]->SetLineColor(2);
    newpostib[i]->SetLineColor(4);
    newpostib[i]->SetLineStyle(2);
    postib[i]->Sumw2();
    newpostib[i]->Sumw2();
    postib[i]->SetNormFactor(1.0);
    newpostib[i]->SetNormFactor(1.0);
    postib[i]->Draw("h");
    newpostib[i]->Draw("sameh");
    myPV->PVCompute(postib[i] , newpostib[i] , te );
   leg.Clear();
   leg.AddEntry(postib[i],rver , "l");
   leg.AddEntry(newpostib[i],cver , "l");
   leg.Draw();
  }
  
  Strip2->Print("PosTIBCompare.eps");
  Strip2->Print("PosTIBCompare.gif");
  

 rdir->GetObject("TIB/Errx_rphi_layer1tib",errxtib[0]);
 rdir->GetObject("TIB/Errx_rphi_layer2tib",errxtib[1]);
 rdir->GetObject("TIB/Errx_rphi_layer3tib",errxtib[2]);
 rdir->GetObject("TIB/Errx_rphi_layer4tib",errxtib[3]);
 rdir->GetObject("TIB/Errx_sas_layer1tib",errxtib[4]);
 rdir->GetObject("TIB/Errx_sas_layer2tib",errxtib[5]);
 sdir->GetObject("TIB/Errx_rphi_layer1tib",newerrxtib[0]);
 sdir->GetObject("TIB/Errx_rphi_layer2tib",newerrxtib[1]);
 sdir->GetObject("TIB/Errx_rphi_layer3tib",newerrxtib[2]);
 sdir->GetObject("TIB/Errx_rphi_layer4tib",newerrxtib[3]);
 sdir->GetObject("TIB/Errx_sas_layer1tib",newerrxtib[4]);
 sdir->GetObject("TIB/Errx_sas_layer2tib",newerrxtib[5]);

  Strip3 = new TCanvas("Strip3","Strip3",800,500);
  Strip3->Divide(2,3);
  for (Int_t i=0; i<6; i++) {
    Strip3->cd(i+1);
    errxtib[i]->SetLineColor(2);
    newerrxtib[i]->SetLineColor(4);
    newerrxtib[i]->SetLineStyle(2);
    errxtib[i]->Sumw2();
    newerrxtib[i]->Sumw2();
    errxtib[i]->SetNormFactor(1.0);
    newerrxtib[i]->SetNormFactor(1.0);
    errxtib[i]->Draw("h");
    newerrxtib[i]->Draw("sameh");
    myPV->PVCompute(errxtib[i] , newerrxtib[i] , te );
    leg.Clear();
   leg.AddEntry(errxtib[i],rver , "l");
   leg.AddEntry(newerrxtib[i],cver , "l");
   leg.Draw();
 }
  
  Strip3->Print("ErrxTIBCompare.eps");
  Strip3->Print("ErrxTIBCompare.gif");
  
 rdir->GetObject("TIB/Res_rphi_layer1tib",restib[0]);
 rdir->GetObject("TIB/Res_rphi_layer2tib",restib[1]);
 rdir->GetObject("TIB/Res_rphi_layer3tib",restib[2]);
 rdir->GetObject("TIB/Res_rphi_layer4tib",restib[3]);
 rdir->GetObject("TIB/Res_sas_layer1tib",restib[4]);
 rdir->GetObject("TIB/Res_sas_layer2tib",restib[5]);
 sdir->GetObject("TIB/Res_rphi_layer1tib",newrestib[0]);
 sdir->GetObject("TIB/Res_rphi_layer2tib",newrestib[1]);
 sdir->GetObject("TIB/Res_rphi_layer3tib",newrestib[2]);
 sdir->GetObject("TIB/Res_rphi_layer4tib",newrestib[3]);
 sdir->GetObject("TIB/Res_sas_layer1tib",newrestib[4]);
 sdir->GetObject("TIB/Res_sas_layer2tib",newrestib[5]);

  Strip4 = new TCanvas("Strip4","Strip4",800,500);
  Strip4->Divide(2,3);
  for (Int_t i=0; i<6; i++) {
    Strip4->cd(i+1);
    restib[i]->SetLineColor(2);
    newrestib[i]->SetLineColor(4);
    newrestib[i]->SetLineStyle(2);
    restib[i]->Sumw2();
    newrestib[i]->Sumw2();
    restib[i]->SetNormFactor(1.0);
    newrestib[i]->SetNormFactor(1.0);
    restib[i]->Draw("h");
    newrestib[i]->Draw("sameh");
    myPV->PVCompute(restib[i] , newrestib[i] , te );
    leg.Clear();
   leg.AddEntry(restib[i],rver , "l");
   leg.AddEntry(newrestib[i],cver , "l");
   leg.Draw();
 }
  
  Strip4->Print("ResTIBCompare.eps");
  Strip4->Print("ResTIBCompare.gif");

 rdir->GetObject("TIB/Pull_MF_rphi_layer1tib",pulltib[0]);
 rdir->GetObject("TIB/Pull_MF_rphi_layer2tib",pulltib[1]);
 rdir->GetObject("TIB/Pull_MF_rphi_layer3tib",pulltib[2]);
 rdir->GetObject("TIB/Pull_MF_rphi_layer4tib",pulltib[3]);
 rdir->GetObject("TIB/Pull_MF_sas_layer1tib",pulltib[4]);
 rdir->GetObject("TIB/Pull_MF_sas_layer2tib",pulltib[5]);
 sdir->GetObject("TIB/Pull_MF_rphi_layer1tib",newpulltib[0]);
 sdir->GetObject("TIB/Pull_MF_rphi_layer2tib",newpulltib[1]);
 sdir->GetObject("TIB/Pull_MF_rphi_layer3tib",newpulltib[2]);
 sdir->GetObject("TIB/Pull_MF_rphi_layer4tib",newpulltib[3]);
 sdir->GetObject("TIB/Pull_MF_sas_layer1tib",newpulltib[4]);
 sdir->GetObject("TIB/Pull_MF_sas_layer2tib",newpulltib[5]);
 
 Strip31 = new TCanvas("Strip31","Strip31",800,500);
 Strip31->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   Strip31->cd(i+1);
   pulltib[i]->SetLineColor(2);
   newpulltib[i]->SetLineColor(4);
   newpulltib[i]->SetLineStyle(2);
   pulltib[i]->Sumw2();
   newpulltib[i]->Sumw2();
   pulltib[i]->SetNormFactor(1.0);
   newpulltib[i]->SetNormFactor(1.0);
   pulltib[i]->Draw("h");
   newpulltib[i]->Draw("sameh");
   myPV->PVCompute(pulltib[i] , newpulltib[i] , te );
   leg.Clear();
   leg.AddEntry(pulltib[i],rver , "l");
   leg.AddEntry(newpulltib[i],cver , "l");
   leg.Draw();
 }
 
 Strip31->Print("PullTIBCompare.eps");
 Strip31->Print("PullTIBCompare.gif");

 rdir->GetObject("TIB/Chi2_rphi_layer1tib",chi2tib[0]);
 rdir->GetObject("TIB/Chi2_rphi_layer2tib",chi2tib[1]);
 rdir->GetObject("TIB/Chi2_rphi_layer3tib",chi2tib[2]);
 rdir->GetObject("TIB/Chi2_rphi_layer4tib",chi2tib[3]);
 rdir->GetObject("TIB/Chi2_sas_layer1tib",chi2tib[4]);
 rdir->GetObject("TIB/Chi2_sas_layer2tib",chi2tib[5]);
 sdir->GetObject("TIB/Chi2_rphi_layer1tib",newchi2tib[0]);
 sdir->GetObject("TIB/Chi2_rphi_layer2tib",newchi2tib[1]);
 sdir->GetObject("TIB/Chi2_rphi_layer3tib",newchi2tib[2]);
 sdir->GetObject("TIB/Chi2_rphi_layer4tib",newchi2tib[3]);
 sdir->GetObject("TIB/Chi2_sas_layer1tib",newchi2tib[4]);
 sdir->GetObject("TIB/Chi2_sas_layer2tib",newchi2tib[5]);


  Strip5 = new TCanvas("Strip5","Strip5",800,500);
  Strip5->Divide(2,3);
  for (Int_t i=0; i<6; i++) {
    Strip5->cd(i+1);
    chi2tib[i]->SetLineColor(2);
    newchi2tib[i]->SetLineColor(4);
    newchi2tib[i]->SetLineStyle(2);
    chi2tib[i]->Sumw2();
    newchi2tib[i]->Sumw2();
    chi2tib[i]->SetNormFactor(1.0);
    newchi2tib[i]->SetNormFactor(1.0);
    chi2tib[i]->Draw("h");
    newchi2tib[i]->Draw("sameh");
    myPV->PVCompute(chi2tib[i] , newchi2tib[i] , te );
     leg.Clear();
   leg.AddEntry(chi2tib[i],rver , "l");
   leg.AddEntry(newchi2tib[i],cver , "l");
   leg.Draw();
}
  
  Strip5->Print("Chi2TIBCompare.eps");
  Strip5->Print("Chi2TIBCompare.gif");


 rdir->GetObject("TIB/Posx_matched_layer1tib",matchedtib[0]);
 rdir->GetObject("TIB/Posy_matched_layer1tib",matchedtib[1]);
 rdir->GetObject("TIB/Posx_matched_layer2tib",matchedtib[2]);
 rdir->GetObject("TIB/Posy_matched_layer2tib",matchedtib[3]);
 rdir->GetObject("TIB/Errx_matched_layer1tib",matchedtib[4]);
 rdir->GetObject("TIB/Erry_matched_layer1tib",matchedtib[5]);
 rdir->GetObject("TIB/Errx_matched_layer2tib",matchedtib[6]);
 rdir->GetObject("TIB/Erry_matched_layer2tib",matchedtib[7]);
 rdir->GetObject("TIB/Resx_matched_layer1tib",matchedtib[8]);
 rdir->GetObject("TIB/Resy_matched_layer1tib",matchedtib[9]);
 rdir->GetObject("TIB/Resx_matched_layer2tib",matchedtib[10]);
 rdir->GetObject("TIB/Resy_matched_layer2tib",matchedtib[11]);
 sdir->GetObject("TIB/Posx_matched_layer1tib",newmatchedtib[0]);
 sdir->GetObject("TIB/Posy_matched_layer1tib",newmatchedtib[1]);
 sdir->GetObject("TIB/Posx_matched_layer2tib",newmatchedtib[2]);
 sdir->GetObject("TIB/Posy_matched_layer2tib",newmatchedtib[3]);
 sdir->GetObject("TIB/Errx_matched_layer1tib",newmatchedtib[4]);
 sdir->GetObject("TIB/Erry_matched_layer1tib",newmatchedtib[5]);
 sdir->GetObject("TIB/Errx_matched_layer2tib",newmatchedtib[6]);
 sdir->GetObject("TIB/Erry_matched_layer2tib",newmatchedtib[7]);
 sdir->GetObject("TIB/Resx_matched_layer1tib",newmatchedtib[8]);
 sdir->GetObject("TIB/Resy_matched_layer1tib",newmatchedtib[9]);
 sdir->GetObject("TIB/Resx_matched_layer2tib",newmatchedtib[10]);
 sdir->GetObject("TIB/Resy_matched_layer2tib",newmatchedtib[11]);

 Strip6 = new TCanvas("Strip6","Strip6",800,500);
 Strip6->Divide(4,3);
 for (Int_t i=0; i<12; i++) {
   Strip6->cd(i+1);
   matchedtib[i]->SetLineColor(2);
   newmatchedtib[i]->SetLineColor(4);
   newmatchedtib[i]->SetLineStyle(2);
   matchedtib[i]->Sumw2();
   newmatchedtib[i]->Sumw2();
   matchedtib[i]->SetNormFactor(1.0);
   newmatchedtib[i]->SetNormFactor(1.0);
   matchedtib[i]->Draw("h");
   newmatchedtib[i]->Draw("sameh");
   myPV->PVCompute(matchedtib[i] , newmatchedtib[i] , te );
    leg.Clear();
   leg.AddEntry(matchedtib[i],rver , "l");
   leg.AddEntry(newmatchedtib[i],cver , "l");
   leg.Draw();
}
 
 Strip6->Print("MatchedTIBCompare.eps");
 Strip6->Print("MatchedTIBCompare.gif");


 //======================================================================================================
// TOB
 
 TH1F* adctob[8];
 TH1F* pulltob[8];
 TH1F* nstptob[8];
 TH1F* postob[8];
 TH1F* errxtob[8];
 TH1F* restob[8];
 TH1F* chi2tob[8];
 TH1F* matchedtob[12];
 TH1F* newadctob[8];
 TH1F* newpulltob[8];
 TH1F* newnstptob[8];
 TH1F* newpostob[8];
 TH1F* newerrxtob[8];
 TH1F* newrestob[8];
 TH1F* newchi2tob[8];
 TH1F* newmatchedtob[12];
 
 rdir->GetObject("TOB/Adc_rphi_layer1tob",adctob[0]);
 rdir->GetObject("TOB/Adc_rphi_layer2tob",adctob[1]);
 rdir->GetObject("TOB/Adc_rphi_layer3tob",adctob[2]);
 rdir->GetObject("TOB/Adc_rphi_layer4tob",adctob[3]);
 rdir->GetObject("TOB/Adc_rphi_layer5tob",adctob[4]);
 rdir->GetObject("TOB/Adc_rphi_layer6tob",adctob[5]);
 rdir->GetObject("TOB/Adc_sas_layer1tob",adctob[6]);
 rdir->GetObject("TOB/Adc_sas_layer2tob",adctob[7]);
 sdir->GetObject("TOB/Adc_rphi_layer1tob",newadctob[0]);
 sdir->GetObject("TOB/Adc_rphi_layer2tob",newadctob[1]);
 sdir->GetObject("TOB/Adc_rphi_layer3tob",newadctob[2]);
 sdir->GetObject("TOB/Adc_rphi_layer4tob",newadctob[3]);
 sdir->GetObject("TOB/Adc_rphi_layer5tob",newadctob[4]);
 sdir->GetObject("TOB/Adc_rphi_layer6tob",newadctob[5]);
 sdir->GetObject("TOB/Adc_sas_layer1tob",newadctob[6]);
 sdir->GetObject("TOB/Adc_sas_layer2tob",newadctob[7]);
 
 Strip7 = new TCanvas("Strip7","Strip7",800,500);
 Strip7->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   Strip7->cd(i+1);
   adctob[i]->SetLineColor(2);
   newadctob[i]->SetLineColor(4);
   newadctob[i]->SetLineStyle(2);
   adctob[i]->Sumw2();
   newadctob[i]->Sumw2();
   adctob[i]->SetNormFactor(1.0);
   newadctob[i]->SetNormFactor(1.0);
   adctob[i]->Draw("h");
   newadctob[i]->Draw("sameh");
   myPV->PVCompute(adctob[i] , newadctob[i] , te );
   leg.Clear();
   leg.AddEntry(adctob[i],rver , "l");
   leg.AddEntry(newadctob[i],cver , "l");
   leg.Draw();
 }
 
 Strip7->Print("AdcTOBCompare.eps");
 Strip7->Print("AdcTOBCompare.gif");
 
 rdir->GetObject("TOB/Nstp_rphi_layer1tob",nstptob[0]);
 rdir->GetObject("TOB/Nstp_rphi_layer2tob",nstptob[1]);
 rdir->GetObject("TOB/Nstp_rphi_layer3tob",nstptob[2]);
 rdir->GetObject("TOB/Nstp_rphi_layer4tob",nstptob[3]);
 rdir->GetObject("TOB/Nstp_rphi_layer5tob",nstptob[4]);
 rdir->GetObject("TOB/Nstp_rphi_layer6tob",nstptob[5]);
 rdir->GetObject("TOB/Nstp_sas_layer1tob",nstptob[6]);
 rdir->GetObject("TOB/Nstp_sas_layer2tob",nstptob[7]);
 sdir->GetObject("TOB/Nstp_rphi_layer1tob",newnstptob[0]);
 sdir->GetObject("TOB/Nstp_rphi_layer2tob",newnstptob[1]);
 sdir->GetObject("TOB/Nstp_rphi_layer3tob",newnstptob[2]);
 sdir->GetObject("TOB/Nstp_rphi_layer4tob",newnstptob[3]);
 sdir->GetObject("TOB/Nstp_rphi_layer5tob",newnstptob[4]);
 sdir->GetObject("TOB/Nstp_rphi_layer6tob",newnstptob[5]);
 sdir->GetObject("TOB/Nstp_sas_layer1tob",newnstptob[6]);
 sdir->GetObject("TOB/Nstp_sas_layer2tob",newnstptob[7]);

 Strip8 = new TCanvas("Strip8","Strip8",800,500);
 Strip8->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   Strip8->cd(i+1);
   nstptob[i]->SetLineColor(2);
   newnstptob[i]->SetLineColor(4);
   newnstptob[i]->SetLineStyle(2);
   nstptob[i]->Sumw2();
   newnstptob[i]->Sumw2();
   nstptob[i]->SetNormFactor(1.0);
   newnstptob[i]->SetNormFactor(1.0);
   nstptob[i]->Draw("h");
   newnstptob[i]->Draw("sameh");
   myPV->PVCompute(nstptob[i] , newnstptob[i] , te );
   leg.Clear();
   leg.AddEntry(nstptob[i],rver , "l");
   leg.AddEntry(newnstptob[i],cver , "l");
   leg.Draw();
 }
 
 Strip8->Print("NstpTOBCompare.eps");
 Strip8->Print("NstpTOBCompare.gif");

 rdir->GetObject("TOB/Posx_rphi_layer1tob",postob[0]);
 rdir->GetObject("TOB/Posx_rphi_layer2tob",postob[1]);
 rdir->GetObject("TOB/Posx_rphi_layer3tob",postob[2]);
 rdir->GetObject("TOB/Posx_rphi_layer4tob",postob[3]);
 rdir->GetObject("TOB/Posx_rphi_layer5tob",postob[4]);
 rdir->GetObject("TOB/Posx_rphi_layer6tob",postob[5]);
 rdir->GetObject("TOB/Posx_sas_layer1tob",postob[6]);
 rdir->GetObject("TOB/Posx_sas_layer2tob",postob[7]);
 sdir->GetObject("TOB/Posx_rphi_layer1tob",newpostob[0]);
 sdir->GetObject("TOB/Posx_rphi_layer2tob",newpostob[1]);
 sdir->GetObject("TOB/Posx_rphi_layer3tob",newpostob[2]);
 sdir->GetObject("TOB/Posx_rphi_layer4tob",newpostob[3]);
 sdir->GetObject("TOB/Posx_rphi_layer5tob",newpostob[4]);
 sdir->GetObject("TOB/Posx_rphi_layer6tob",newpostob[5]);
 sdir->GetObject("TOB/Posx_sas_layer1tob",newpostob[6]);
 sdir->GetObject("TOB/Posx_sas_layer2tob",newpostob[7]);

 Strip9 = new TCanvas("Strip9","Strip9",800,500);
  Strip9->Divide(3,3);
  for (Int_t i=0; i<8; i++) {
    Strip9->cd(i+1);
    postob[i]->SetLineColor(2);
    newpostob[i]->SetLineColor(4);
    newpostob[i]->SetLineStyle(2);
    postob[i]->Sumw2();
    newpostob[i]->Sumw2();
    postob[i]->SetNormFactor(1.0);
    newpostob[i]->SetNormFactor(1.0);
    postob[i]->Draw("h");
    newpostob[i]->Draw("sameh");
    myPV->PVCompute(postob[i] , newpostob[i] , te );
     leg.Clear();
   leg.AddEntry(postob[i],rver , "l");
   leg.AddEntry(newpostob[i],cver , "l");
   leg.Draw();
}
  
  Strip9->Print("PosTOBCompare.eps");
  Strip9->Print("PosTOBCompare.gif");
  

 rdir->GetObject("TOB/Errx_rphi_layer1tob",errxtob[0]);
 rdir->GetObject("TOB/Errx_rphi_layer2tob",errxtob[1]);
 rdir->GetObject("TOB/Errx_rphi_layer3tob",errxtob[2]);
 rdir->GetObject("TOB/Errx_rphi_layer4tob",errxtob[3]);
 rdir->GetObject("TOB/Errx_rphi_layer5tob",errxtob[4]);
 rdir->GetObject("TOB/Errx_rphi_layer6tob",errxtob[5]);
 rdir->GetObject("TOB/Errx_sas_layer1tob",errxtob[6]);
 rdir->GetObject("TOB/Errx_sas_layer2tob",errxtob[7]);
 sdir->GetObject("TOB/Errx_rphi_layer1tob",newerrxtob[0]);
 sdir->GetObject("TOB/Errx_rphi_layer2tob",newerrxtob[1]);
 sdir->GetObject("TOB/Errx_rphi_layer3tob",newerrxtob[2]);
 sdir->GetObject("TOB/Errx_rphi_layer4tob",newerrxtob[3]);
 sdir->GetObject("TOB/Errx_rphi_layer5tob",newerrxtob[4]);
 sdir->GetObject("TOB/Errx_rphi_layer6tob",newerrxtob[5]);
 sdir->GetObject("TOB/Errx_sas_layer1tob",newerrxtob[6]);
 sdir->GetObject("TOB/Errx_sas_layer2tob",newerrxtob[7]);

  Strip10 = new TCanvas("Strip10","Strip10",800,500);
  Strip10->Divide(2,3);
  for (Int_t i=0; i<8; i++) {
    Strip10->cd(i+1);
    errxtob[i]->SetLineColor(2);
    newerrxtob[i]->SetLineColor(4);
    newerrxtob[i]->SetLineStyle(2);
    errxtob[i]->Sumw2();
    newerrxtob[i]->Sumw2();
    errxtob[i]->SetNormFactor(1.0);
    newerrxtob[i]->SetNormFactor(1.0);
    errxtob[i]->Draw("h");
    newerrxtob[i]->Draw("sameh");
    myPV->PVCompute(errxtob[i] , newerrxtob[i] , te );
     leg.Clear();
   leg.AddEntry(errxtob[i],rver , "l");
   leg.AddEntry(newerrxtob[i],cver , "l");
   leg.Draw();
}
  
  Strip10->Print("ErrxTOBCompare.eps");
  Strip10->Print("ErrxTOBCompare.gif");
  
 
 rdir->GetObject("TOB/Res_rphi_layer1tob",restob[0]);
 rdir->GetObject("TOB/Res_rphi_layer2tob",restob[1]);
 rdir->GetObject("TOB/Res_rphi_layer3tob",restob[2]);
 rdir->GetObject("TOB/Res_rphi_layer4tob",restob[3]);
 rdir->GetObject("TOB/Res_rphi_layer5tob",restob[4]);
 rdir->GetObject("TOB/Res_rphi_layer6tob",restob[5]);
 rdir->GetObject("TOB/Res_sas_layer1tob",restob[6]);
 rdir->GetObject("TOB/Res_sas_layer2tob",restob[7]);
 sdir->GetObject("TOB/Res_rphi_layer1tob",newrestob[0]);
 sdir->GetObject("TOB/Res_rphi_layer2tob",newrestob[1]);
 sdir->GetObject("TOB/Res_rphi_layer3tob",newrestob[2]);
 sdir->GetObject("TOB/Res_rphi_layer4tob",newrestob[3]);
 sdir->GetObject("TOB/Res_rphi_layer5tob",newrestob[4]);
 sdir->GetObject("TOB/Res_rphi_layer6tob",newrestob[5]);
 sdir->GetObject("TOB/Res_sas_layer1tob",newrestob[6]);
 sdir->GetObject("TOB/Res_sas_layer2tob",newrestob[7]);

  Strip11 = new TCanvas("Strip11","Strip11",800,500);
  Strip11->Divide(3,3);
  for (Int_t i=0; i<8; i++) {
    Strip11->cd(i+1);
    restob[i]->SetLineColor(2);
    newrestob[i]->SetLineColor(4);
    newrestob[i]->SetLineStyle(2);
    restob[i]->Sumw2();
    newrestob[i]->Sumw2();
    restob[i]->SetNormFactor(1.0);
    newrestob[i]->SetNormFactor(1.0);
    restob[i]->Draw("h");
    newrestob[i]->Draw("sameh");
    myPV->PVCompute(restob[i] , newrestob[i] , te );
    leg.Clear();
   leg.AddEntry(restob[i],rver , "l");
   leg.AddEntry(newrestob[i],cver , "l");
   leg.Draw();
 }
  
  Strip11->Print("ResTOBCompare.eps");
  Strip11->Print("ResTOBCompare.gif");

 rdir->GetObject("TOB/Pull_MF_rphi_layer1tob",pulltob[0]);
 rdir->GetObject("TOB/Pull_MF_rphi_layer2tob",pulltob[1]);
 rdir->GetObject("TOB/Pull_MF_rphi_layer3tob",pulltob[2]);
 rdir->GetObject("TOB/Pull_MF_rphi_layer4tob",pulltob[3]);
 rdir->GetObject("TOB/Pull_MF_rphi_layer5tob",pulltob[4]);
 rdir->GetObject("TOB/Pull_MF_rphi_layer6tob",pulltob[5]);
 rdir->GetObject("TOB/Pull_MF_sas_layer1tob",pulltob[6]);
 rdir->GetObject("TOB/Pull_MF_sas_layer2tob",pulltob[7]);
 sdir->GetObject("TOB/Pull_MF_rphi_layer1tob",newpulltob[0]);
 sdir->GetObject("TOB/Pull_MF_rphi_layer2tob",newpulltob[1]);
 sdir->GetObject("TOB/Pull_MF_rphi_layer3tob",newpulltob[2]);
 sdir->GetObject("TOB/Pull_MF_rphi_layer4tob",newpulltob[3]);
 sdir->GetObject("TOB/Pull_MF_rphi_layer5tob",newpulltob[4]);
 sdir->GetObject("TOB/Pull_MF_rphi_layer6tob",newpulltob[5]);
 sdir->GetObject("TOB/Pull_MF_sas_layer1tob",newpulltob[6]);
 sdir->GetObject("TOB/Pull_MF_sas_layer2tob",newpulltob[7]);
 
 Strip32 = new TCanvas("Strip32","Strip32",800,500);
 Strip32->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   Strip32->cd(i+1);
   pulltob[i]->SetLineColor(2);
   newpulltob[i]->SetLineColor(4);
   newpulltob[i]->SetLineStyle(2);
   pulltob[i]->Sumw2();
   newpulltob[i]->Sumw2();
   pulltob[i]->SetNormFactor(1.0);
   newpulltob[i]->SetNormFactor(1.0);
   pulltob[i]->Draw("h");
   newpulltob[i]->Draw("sameh");
    leg.Clear();
  myPV->PVCompute(pulltob[i] , newpulltob[i] , te );
   leg.AddEntry(pulltob[i],rver , "l");
   leg.AddEntry(newpulltob[i],cver , "l");
   leg.Draw();
 }
 
 Strip32->Print("PullTOBCompare.eps");
 Strip32->Print("PullTOBCompare.gif");


 rdir->GetObject("TOB/Chi2_rphi_layer1tob",chi2tob[0]);
 rdir->GetObject("TOB/Chi2_rphi_layer2tob",chi2tob[1]);
 rdir->GetObject("TOB/Chi2_rphi_layer3tob",chi2tob[2]);
 rdir->GetObject("TOB/Chi2_rphi_layer4tob",chi2tob[3]);
 rdir->GetObject("TOB/Chi2_rphi_layer5tob",chi2tob[4]);
 rdir->GetObject("TOB/Chi2_rphi_layer6tob",chi2tob[5]);
 rdir->GetObject("TOB/Chi2_sas_layer1tob",chi2tob[6]);
 rdir->GetObject("TOB/Chi2_sas_layer2tob",chi2tob[7]);
 sdir->GetObject("TOB/Chi2_rphi_layer1tob",newchi2tob[0]);
 sdir->GetObject("TOB/Chi2_rphi_layer2tob",newchi2tob[1]);
 sdir->GetObject("TOB/Chi2_rphi_layer3tob",newchi2tob[2]);
 sdir->GetObject("TOB/Chi2_rphi_layer4tob",newchi2tob[3]);
 sdir->GetObject("TOB/Chi2_rphi_layer5tob",newchi2tob[4]);
 sdir->GetObject("TOB/Chi2_rphi_layer6tob",newchi2tob[5]);
 sdir->GetObject("TOB/Chi2_sas_layer1tob",newchi2tob[6]);
 sdir->GetObject("TOB/Chi2_sas_layer2tob",newchi2tob[7]);

  Strip12 = new TCanvas("Strip12","Strip12",800,500);
  Strip12->Divide(3,3);
  for (Int_t i=0; i<8; i++) {
    Strip12->cd(i+1);
    chi2tob[i]->SetLineColor(2);
    newchi2tob[i]->SetLineColor(4);
    newchi2tob[i]->SetLineStyle(2);
    chi2tob[i]->Sumw2();
    newchi2tob[i]->Sumw2();
    chi2tob[i]->SetNormFactor(1.0);
    newchi2tob[i]->SetNormFactor(1.0);
    chi2tob[i]->Draw("h");
    newchi2tob[i]->Draw("sameh");
    myPV->PVCompute(chi2tob[i] , newchi2tob[i] , te );
    leg.Clear();
   leg.AddEntry(chi2tob[i],rver , "l");
   leg.AddEntry(newchi2tob[i],cver , "l");
   leg.Draw();
 }
  
  Strip12->Print("Chi2TOBCompare.eps");
  Strip12->Print("Chi2TOBCompare.gif");

  
 rdir->GetObject("TOB/Posx_matched_layer1tob",matchedtob[0]);
 rdir->GetObject("TOB/Posy_matched_layer1tob",matchedtob[1]);
 rdir->GetObject("TOB/Posx_matched_layer2tob",matchedtob[2]);
 rdir->GetObject("TOB/Posy_matched_layer2tob",matchedtob[3]);
 rdir->GetObject("TOB/Errx_matched_layer1tob",matchedtob[4]);
 rdir->GetObject("TOB/Erry_matched_layer1tob",matchedtob[5]);
 rdir->GetObject("TOB/Errx_matched_layer2tob",matchedtob[6]);
 rdir->GetObject("TOB/Erry_matched_layer2tob",matchedtob[7]);
 rdir->GetObject("TOB/Resx_matched_layer1tob",matchedtob[8]);
 rdir->GetObject("TOB/Resy_matched_layer1tob",matchedtob[9]);
 rdir->GetObject("TOB/Resx_matched_layer2tob",matchedtob[10]);
 rdir->GetObject("TOB/Resy_matched_layer2tob",matchedtob[11]);
 sdir->GetObject("TOB/Posx_matched_layer1tob",newmatchedtob[0]);
 sdir->GetObject("TOB/Posy_matched_layer1tob",newmatchedtob[1]);
 sdir->GetObject("TOB/Posx_matched_layer2tob",newmatchedtob[2]);
 sdir->GetObject("TOB/Posy_matched_layer2tob",newmatchedtob[3]);
 sdir->GetObject("TOB/Errx_matched_layer1tob",newmatchedtob[4]);
 sdir->GetObject("TOB/Erry_matched_layer1tob",newmatchedtob[5]);
 sdir->GetObject("TOB/Errx_matched_layer2tob",newmatchedtob[6]);
 sdir->GetObject("TOB/Erry_matched_layer2tob",newmatchedtob[7]);
 sdir->GetObject("TOB/Resx_matched_layer1tob",newmatchedtob[8]);
 sdir->GetObject("TOB/Resy_matched_layer1tob",newmatchedtob[9]);
 sdir->GetObject("TOB/Resx_matched_layer2tob",newmatchedtob[10]);
 sdir->GetObject("TOB/Resy_matched_layer2tob",newmatchedtob[11]);

 Strip13 = new TCanvas("Strip13","Strip13",800,500);
 Strip13->Divide(4,3);
 for (Int_t i=0; i<12; i++) {
   Strip13->cd(i+1);
   matchedtob[i]->SetLineColor(2);
   newmatchedtob[i]->SetLineColor(4);
   newmatchedtob[i]->SetLineStyle(2);
   matchedtob[i]->Sumw2();
   newmatchedtob[i]->Sumw2();
   matchedtob[i]->SetNormFactor(1.0);
   newmatchedtob[i]->SetNormFactor(1.0);
   matchedtob[i]->Draw("h");
   newmatchedtob[i]->Draw("sameh");
   myPV->PVCompute(matchedtob[i] , newmatchedtob[i] , te );
    leg.Clear();
   leg.AddEntry(matchedtob[i],rver , "l");
   leg.AddEntry(newmatchedtob[i],cver , "l");
   leg.Draw();
}
 
 Strip13->Print("MatchedTOBCompare.eps");
 Strip13->Print("MatchedTOBCompare.gif");
 

 //=============================================================== 
// TID
 
 TH1F* adctid[5];
 TH1F* pulltid[5];
 TH1F* nstptid[5];
 TH1F* postid[5];
 TH1F* errxtid[5];
 TH1F* restid[5];
 TH1F* chi2tid[5];
 TH1F* matchedtid[12];
 TH1F* matchedchi2tid[6];
 TH1F* newadctid[5];
 TH1F* newpulltid[5];
 TH1F* newnstptid[5];
 TH1F* newpostid[5];
 TH1F* newerrxtid[5];
 TH1F* newrestid[5];
 TH1F* newchi2tid[5];
 TH1F* newmatchedtid[12];
 TH1F* newmatchedchi2tid[6];
 
 rdir->GetObject("TID/Adc_rphi_layer1tid",adctid[0]);
 rdir->GetObject("TID/Adc_rphi_layer2tid",adctid[1]);
 rdir->GetObject("TID/Adc_rphi_layer3tid",adctid[2]);
 rdir->GetObject("TID/Adc_sas_layer1tid",adctid[3]);
 rdir->GetObject("TID/Adc_sas_layer2tid",adctid[4]);
 sdir->GetObject("TID/Adc_rphi_layer1tid",newadctid[0]);
 sdir->GetObject("TID/Adc_rphi_layer2tid",newadctid[1]);
 sdir->GetObject("TID/Adc_rphi_layer3tid",newadctid[2]);
 sdir->GetObject("TID/Adc_sas_layer1tid",newadctid[3]);
 sdir->GetObject("TID/Adc_sas_layer2tid",newadctid[4]);
 
 Strip14 = new TCanvas("Strip14","Strip14",800,500);
 Strip14->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   Strip14->cd(i+1);
   adctid[i]->SetLineColor(2);
   newadctid[i]->SetLineColor(4);
   newadctid[i]->SetLineStyle(2);
   adctid[i]->Sumw2();
   newadctid[i]->Sumw2();
   adctid[i]->SetNormFactor(1.0);
   newadctid[i]->SetNormFactor(1.0);
   adctid[i]->Draw("h");
   newadctid[i]->Draw("sameh");
   myPV->PVCompute(adctid[i] , newadctid[i] , te );
    leg.Clear();
   leg.AddEntry(adctid[i],rver , "l");
   leg.AddEntry(newadctid[i],cver , "l");
   leg.Draw();
}
 
 Strip14->Print("AdcTIDCompare.eps");
 Strip14->Print("AdcTIDCompare.gif");
 
 rdir->GetObject("TID/Nstp_rphi_layer1tid",nstptid[0]);
 rdir->GetObject("TID/Nstp_rphi_layer2tid",nstptid[1]);
 rdir->GetObject("TID/Nstp_rphi_layer3tid",nstptid[2]);
 rdir->GetObject("TID/Nstp_sas_layer1tid",nstptid[3]);
 rdir->GetObject("TID/Nstp_sas_layer2tid",nstptid[4]);
 sdir->GetObject("TID/Nstp_rphi_layer1tid",newnstptid[0]);
 sdir->GetObject("TID/Nstp_rphi_layer2tid",newnstptid[1]);
 sdir->GetObject("TID/Nstp_rphi_layer3tid",newnstptid[2]);
 sdir->GetObject("TID/Nstp_sas_layer1tid",newnstptid[3]);
 sdir->GetObject("TID/Nstp_sas_layer2tid",newnstptid[4]);

 Strip15 = new TCanvas("Strip15","Strip15",800,500);
 Strip15->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   Strip15->cd(i+1);
   nstptid[i]->SetLineColor(2);
   newnstptid[i]->SetLineColor(4);
   newnstptid[i]->SetLineStyle(2);
   nstptid[i]->Sumw2();
   newnstptid[i]->Sumw2();
   nstptid[i]->SetNormFactor(1.0);
   newnstptid[i]->SetNormFactor(1.0);
   nstptid[i]->Draw("h");
   newnstptid[i]->Draw("sameh");
   myPV->PVCompute(nstptid[i] , newnstptid[i] , te );
    leg.Clear();
   leg.AddEntry(nstptid[i],rver , "l");
   leg.AddEntry(newnstptid[i],cver , "l");
   leg.Draw();
}
 
 Strip15->Print("NstpTIDCompare.eps");
 Strip15->Print("NstpTIDCompare.gif");

 rdir->GetObject("TID/Posx_rphi_layer1tid",postid[0]);
 rdir->GetObject("TID/Posx_rphi_layer2tid",postid[1]);
 rdir->GetObject("TID/Posx_rphi_layer3tid",postid[2]);
 rdir->GetObject("TID/Posx_sas_layer1tid",postid[3]);
 rdir->GetObject("TID/Posx_sas_layer2tid",postid[4]);
 sdir->GetObject("TID/Posx_rphi_layer1tid",newpostid[0]);
 sdir->GetObject("TID/Posx_rphi_layer2tid",newpostid[1]);
 sdir->GetObject("TID/Posx_rphi_layer3tid",newpostid[2]);
 sdir->GetObject("TID/Posx_sas_layer1tid",newpostid[3]);
 sdir->GetObject("TID/Posx_sas_layer2tid",newpostid[4]);

 Strip16 = new TCanvas("Strip16","Strip16",800,500);
  Strip16->Divide(2,3);
  for (Int_t i=0; i<5; i++) {
    Strip16->cd(i+1);
    postid[i]->SetLineColor(2);
    newpostid[i]->SetLineColor(4);
    newpostid[i]->SetLineStyle(2);
    postid[i]->Sumw2();
    newpostid[i]->Sumw2();
    postid[i]->SetNormFactor(1.0);
    newpostid[i]->SetNormFactor(1.0);
    postid[i]->Draw("h");
    newpostid[i]->Draw("sameh");
    myPV->PVCompute(postid[i] , newpostid[i] , te );
     leg.Clear();
   leg.AddEntry(postid[i],rver , "l");
   leg.AddEntry(newpostid[i],cver , "l");
   leg.Draw();
}
  
  Strip16->Print("PosTIDCompare.eps");
  Strip16->Print("PosTIDCompare.gif");
  

 rdir->GetObject("TID/Errx_rphi_layer1tid",errxtid[0]);
 rdir->GetObject("TID/Errx_rphi_layer2tid",errxtid[1]);
 rdir->GetObject("TID/Errx_rphi_layer3tid",errxtid[2]);
 rdir->GetObject("TID/Errx_sas_layer1tid",errxtid[3]);
 rdir->GetObject("TID/Errx_sas_layer2tid",errxtid[4]);
 sdir->GetObject("TID/Errx_rphi_layer1tid",newerrxtid[0]);
 sdir->GetObject("TID/Errx_rphi_layer2tid",newerrxtid[1]);
 sdir->GetObject("TID/Errx_rphi_layer3tid",newerrxtid[2]);
 sdir->GetObject("TID/Errx_sas_layer1tid",newerrxtid[3]);
 sdir->GetObject("TID/Errx_sas_layer2tid",newerrxtid[4]);

  Strip17 = new TCanvas("Strip17","Strip17",800,500);
  Strip17->Divide(2,3);
  for (Int_t i=0; i<5; i++) {
    Strip17->cd(i+1);
    errxtid[i]->SetLineColor(2);
    newerrxtid[i]->SetLineColor(4);
    newerrxtid[i]->SetLineStyle(2);
    errxtid[i]->Sumw2();
    newerrxtid[i]->Sumw2();
    errxtid[i]->SetNormFactor(1.0);
    newerrxtid[i]->SetNormFactor(1.0);
    errxtid[i]->Draw("h");
    newerrxtid[i]->Draw("sameh");
    myPV->PVCompute(errxtid[i] , newerrxtid[i] , te );
    leg.Clear();
   leg.AddEntry(errxtid[i],rver , "l");
   leg.AddEntry(newerrxtid[i],cver , "l");
   leg.Draw();
 }
  
  Strip17->Print("ErrxTIDCompare.eps");
  Strip17->Print("ErrxTIDCompare.gif");
  
 
 rdir->GetObject("TID/Res_rphi_layer1tid",restid[0]);
 rdir->GetObject("TID/Res_rphi_layer2tid",restid[1]);
 rdir->GetObject("TID/Res_rphi_layer3tid",restid[2]);
 rdir->GetObject("TID/Res_sas_layer1tid",restid[3]);
 rdir->GetObject("TID/Res_sas_layer2tid",restid[4]);
 sdir->GetObject("TID/Res_rphi_layer1tid",newrestid[0]);
 sdir->GetObject("TID/Res_rphi_layer2tid",newrestid[1]);
 sdir->GetObject("TID/Res_rphi_layer3tid",newrestid[2]);
 sdir->GetObject("TID/Res_sas_layer1tid",newrestid[3]);
 sdir->GetObject("TID/Res_sas_layer2tid",newrestid[4]);

  Strip18 = new TCanvas("Strip18","Strip18",800,500);
  Strip18->Divide(2,3);
  for (Int_t i=0; i<5; i++) {
    Strip18->cd(i+1);
    restid[i]->SetLineColor(2);
    newrestid[i]->SetLineColor(4);
    newrestid[i]->SetLineStyle(2);
    restid[i]->Sumw2();
    newrestid[i]->Sumw2();
    restid[i]->SetNormFactor(1.0);
    newrestid[i]->SetNormFactor(1.0);
    restid[i]->Draw("h");
    newrestid[i]->Draw("sameh");
    myPV->PVCompute(restid[i] , newrestid[i] , te );
   leg.Clear();
   leg.AddEntry(restid[i],rver , "l");
   leg.AddEntry(newrestid[i],cver , "l");
   leg.Draw();
  }
  
  Strip18->Print("ResTIDCompare.eps");
  Strip18->Print("ResTIDCompare.gif");

 rdir->GetObject("TID/Pull_MF_rphi_layer1tid",pulltid[0]);
 rdir->GetObject("TID/Pull_MF_rphi_layer2tid",pulltid[1]);
 rdir->GetObject("TID/Pull_MF_rphi_layer3tid",pulltid[2]);
 rdir->GetObject("TID/Pull_MF_sas_layer1tid",pulltid[3]);
 rdir->GetObject("TID/Pull_MF_sas_layer2tid",pulltid[4]);
 sdir->GetObject("TID/Pull_MF_rphi_layer1tid",newpulltid[0]);
 sdir->GetObject("TID/Pull_MF_rphi_layer2tid",newpulltid[1]);
 sdir->GetObject("TID/Pull_MF_rphi_layer3tid",newpulltid[2]);
 sdir->GetObject("TID/Pull_MF_sas_layer1tid",newpulltid[3]);
 sdir->GetObject("TID/Pull_MF_sas_layer2tid",newpulltid[4]);
 
 Strip33 = new TCanvas("Strip33","Strip33",800,500);
 Strip33->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   Strip33->cd(i+1);
   pulltid[i]->SetLineColor(2);
   newpulltid[i]->SetLineColor(4);
   newpulltid[i]->SetLineStyle(2);
   pulltid[i]->Sumw2();
   newpulltid[i]->Sumw2();
   pulltid[i]->SetNormFactor(1.0);
   newpulltid[i]->SetNormFactor(1.0);
   pulltid[i]->Draw("h");
   newpulltid[i]->Draw("sameh");
   myPV->PVCompute(pulltid[i] , newpulltid[i] , te );
   leg.Clear();
   leg.AddEntry(pulltid[i],rver , "l");
   leg.AddEntry(newpulltid[i],cver , "l");
   leg.Draw();
 }
 
 Strip33->Print("PullTIDCompare.eps");
 Strip33->Print("PullTIDCompare.gif");

 rdir->GetObject("TID/Chi2_rphi_layer1tid",chi2tid[0]);
 rdir->GetObject("TID/Chi2_rphi_layer2tid",chi2tid[1]);
 rdir->GetObject("TID/Chi2_rphi_layer3tid",chi2tid[2]);
 rdir->GetObject("TID/Chi2_sas_layer1tid",chi2tid[3]);
 rdir->GetObject("TID/Chi2_sas_layer2tid",chi2tid[4]);
 sdir->GetObject("TID/Chi2_rphi_layer1tid",newchi2tid[0]);
 sdir->GetObject("TID/Chi2_rphi_layer2tid",newchi2tid[1]);
 sdir->GetObject("TID/Chi2_rphi_layer3tid",newchi2tid[2]);
 sdir->GetObject("TID/Chi2_sas_layer1tid",newchi2tid[3]);
 sdir->GetObject("TID/Chi2_sas_layer2tid",newchi2tid[4]);

  Strip19 = new TCanvas("Strip19","Strip19",800,500);
  Strip19->Divide(2,3);
  for (Int_t i=0; i<5; i++) {
    Strip19->cd(i+1);
    chi2tid[i]->SetLineColor(2);
    newchi2tid[i]->SetLineColor(4);
    newchi2tid[i]->SetLineStyle(2);
    chi2tid[i]->Sumw2();
    newchi2tid[i]->Sumw2();
    chi2tid[i]->SetNormFactor(1.0);
    newchi2tid[i]->SetNormFactor(1.0);
    chi2tid[i]->Draw("h");
    newchi2tid[i]->Draw("sameh");
    myPV->PVCompute(chi2tid[i] , newchi2tid[i] , te );
    leg.Clear();
   leg.AddEntry(chi2tid[i],rver , "l");
   leg.AddEntry(newchi2tid[i],cver , "l");
   leg.Draw();
 }
  
  Strip19->Print("Chi2TIDCompare.eps");
  Strip19->Print("Chi2TIDCompare.gif");
  
 rdir->GetObject("TID/Posx_matched_layer1tid",matchedtid[0]);
 rdir->GetObject("TID/Posy_matched_layer1tid",matchedtid[1]);
 rdir->GetObject("TID/Posx_matched_layer2tid",matchedtid[2]);
 rdir->GetObject("TID/Posy_matched_layer2tid",matchedtid[3]);
 rdir->GetObject("TID/Errx_matched_layer1tid",matchedtid[4]);
 rdir->GetObject("TID/Erry_matched_layer1tid",matchedtid[5]);
 rdir->GetObject("TID/Errx_matched_layer2tid",matchedtid[6]);
 rdir->GetObject("TID/Erry_matched_layer2tid",matchedtid[7]);
 rdir->GetObject("TID/Resx_matched_layer1tid",matchedtid[8]);
 rdir->GetObject("TID/Resy_matched_layer1tid",matchedtid[9]);
 rdir->GetObject("TID/Resx_matched_layer2tid",matchedtid[10]);
 rdir->GetObject("TID/Resy_matched_layer2tid",matchedtid[11]);
 sdir->GetObject("TID/Posx_matched_layer1tid",newmatchedtid[0]);
 sdir->GetObject("TID/Posy_matched_layer1tid",newmatchedtid[1]);
 sdir->GetObject("TID/Posx_matched_layer2tid",newmatchedtid[2]);
 sdir->GetObject("TID/Posy_matched_layer2tid",newmatchedtid[3]);
 sdir->GetObject("TID/Errx_matched_layer1tid",newmatchedtid[4]);
 sdir->GetObject("TID/Erry_matched_layer1tid",newmatchedtid[5]);
 sdir->GetObject("TID/Errx_matched_layer2tid",newmatchedtid[6]);
 sdir->GetObject("TID/Erry_matched_layer2tid",newmatchedtid[7]);
 sdir->GetObject("TID/Resx_matched_layer1tid",newmatchedtid[8]);
 sdir->GetObject("TID/Resy_matched_layer1tid",newmatchedtid[9]);
 sdir->GetObject("TID/Resx_matched_layer2tid",newmatchedtid[10]);
 sdir->GetObject("TID/Resy_matched_layer2tid",newmatchedtid[11]);


 Strip20 = new TCanvas("Strip20","Strip20",800,500);
 Strip20->Divide(4,3);
 for (Int_t i=0; i<12; i++) {
   Strip20->cd(i+1);
   matchedtid[i]->SetLineColor(2);
   newmatchedtid[i]->SetLineColor(4);
   newmatchedtid[i]->SetLineStyle(2);
   matchedtid[i]->Sumw2();
   newmatchedtid[i]->Sumw2();
   matchedtid[i]->SetNormFactor(1.0);
   newmatchedtid[i]->SetNormFactor(1.0);
   matchedtid[i]->Draw("h");
   newmatchedtid[i]->Draw("sameh");
   myPV->PVCompute(matchedtid[i] , newmatchedtid[i] , te );
    leg.Clear();
   leg.AddEntry(matchedtid[i],rver , "l");
   leg.AddEntry(newmatchedtid[i],cver , "l");
   leg.Draw();
}
 
 Strip20->Print("MatchedTIDCompare.eps");
 Strip20->Print("MatchedTIDCompare.gif");

 rdir->GetObject("TID/Resx_matched_layer1tid",matchedchi2tid[0]);
 rdir->GetObject("TID/Resy_matched_layer1tid",matchedchi2tid[1]);
 rdir->GetObject("TID/Resx_matched_layer2tid",matchedchi2tid[2]);
 rdir->GetObject("TID/Resy_matched_layer2tid",matchedchi2tid[3]);
 rdir->GetObject("TID/Chi2_matched_layer1tid",matchedchi2tid[4]);
 rdir->GetObject("TID/Chi2_matched_layer2tid",matchedchi2tid[5]);
 sdir->GetObject("TID/Resx_matched_layer1tid",newmatchedchi2tid[0]);
 sdir->GetObject("TID/Resy_matched_layer1tid",newmatchedchi2tid[1]);
 sdir->GetObject("TID/Resx_matched_layer2tid",newmatchedchi2tid[2]);
 sdir->GetObject("TID/Resy_matched_layer2tid",newmatchedchi2tid[3]);
 sdir->GetObject("TID/Chi2_matched_layer1tid",newmatchedchi2tid[4]);
 sdir->GetObject("TID/Chi2_matched_layer2tid",newmatchedchi2tid[5]);

  Strip21 = new TCanvas("Strip21","Strip21",800,500);
  Strip21->Divide(2,3);
  for (Int_t i=0; i<5; i++) {
    Strip21->cd(i+1);
    chi2tid[i]->SetLineColor(2);
    newchi2tid[i]->SetLineColor(4);
    newchi2tid[i]->SetLineStyle(2);
    chi2tid[i]->Sumw2();
    newchi2tid[i]->Sumw2();
    chi2tid[i]->SetNormFactor(1.0);
    newchi2tid[i]->SetNormFactor(1.0);
    chi2tid[i]->Draw("h");
    newchi2tid[i]->Draw("sameh");
    myPV->PVCompute(matchedchi2tid[i] , newmatchedchi2tid[i] , te );
    leg.Clear();
   leg.AddEntry(chi2tid[i],rver , "l");
   leg.AddEntry(newchi2tid[i],cver , "l");
   leg.Draw();
 }
  
  Strip21->Print("Chi2MatchedTIDCompare.eps");
  Strip21->Print("Chi2MatchedTIDCompare.gif");

 //======================================================================================================
// TEC
 
 TH1F* adctec[10];
 TH1F* pulltec[10];
 TH1F* nstptec[10];
 TH1F* postec[10];
 TH1F* errxtec[10];
 TH1F* restec[10];
 TH1F* chi2tec[10];
 TH1F* matchedtec[12];
 TH1F* matchedrestec[6];
 TH1F* matchedchi2tec[3];
 TH1F* newadctec[10];
 TH1F* newpulltec[10];
 TH1F* newnstptec[10];
 TH1F* newpostec[10];
 TH1F* newerrxtec[10];
 TH1F* newrestec[10];
 TH1F* newmatchedtec[12];
 TH1F* newchi2tec[10];
 TH1F* newmatchedrestec[6];
 TH1F* newmatchedchi2tec[3];
 
 rdir->GetObject("TEC/Adc_rphi_layer1tec",adctec[0]);
 rdir->GetObject("TEC/Adc_rphi_layer2tec",adctec[1]);
 rdir->GetObject("TEC/Adc_rphi_layer3tec",adctec[2]);
 rdir->GetObject("TEC/Adc_rphi_layer4tec",adctec[3]);
 rdir->GetObject("TEC/Adc_rphi_layer5tec",adctec[4]);
 rdir->GetObject("TEC/Adc_rphi_layer6tec",adctec[5]);
 rdir->GetObject("TEC/Adc_rphi_layer7tec",adctec[6]);
 rdir->GetObject("TEC/Adc_sas_layer1tec",adctec[7]);
 rdir->GetObject("TEC/Adc_sas_layer2tec",adctec[8]);
 rdir->GetObject("TEC/Adc_sas_layer5tec",adctec[9]);
 sdir->GetObject("TEC/Adc_rphi_layer1tec",newadctec[0]);
 sdir->GetObject("TEC/Adc_rphi_layer2tec",newadctec[1]);
 sdir->GetObject("TEC/Adc_rphi_layer3tec",newadctec[2]);
 sdir->GetObject("TEC/Adc_rphi_layer4tec",newadctec[3]);
 sdir->GetObject("TEC/Adc_rphi_layer5tec",newadctec[4]);
 sdir->GetObject("TEC/Adc_rphi_layer6tec",newadctec[5]);
 sdir->GetObject("TEC/Adc_rphi_layer7tec",newadctec[6]);
 sdir->GetObject("TEC/Adc_sas_layer1tec",newadctec[7]);
 sdir->GetObject("TEC/Adc_sas_layer2tec",newadctec[8]);
 sdir->GetObject("TEC/Adc_sas_layer5tec",newadctec[9]);
 
 Strip22 = new TCanvas("Strip22","Strip22",800,500);
 Strip22->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip22->cd(i+1);
   adctec[i]->SetLineColor(2);
   newadctec[i]->SetLineColor(4);
   newadctec[i]->SetLineStyle(2);
   adctec[i]->Sumw2();
   newadctec[i]->Sumw2();
   adctec[i]->SetNormFactor(1.0);
   newadctec[i]->SetNormFactor(1.0);
   adctec[i]->Draw("h");
   newadctec[i]->Draw("sameh");
   myPV->PVCompute(adctec[i] , newadctec[i] , te );
    leg.Clear();
   leg.AddEntry(adctec[i],rver , "l");
   leg.AddEntry(newadctec[i],cver , "l");
   leg.Draw();
}
 
 Strip22->Print("AdcTECCompare.eps");
 Strip22->Print("AdcTECCompare.gif");
 
 rdir->GetObject("TEC/Nstp_rphi_layer1tec",nstptec[0]);
 rdir->GetObject("TEC/Nstp_rphi_layer2tec",nstptec[1]);
 rdir->GetObject("TEC/Nstp_rphi_layer3tec",nstptec[2]);
 rdir->GetObject("TEC/Nstp_rphi_layer4tec",nstptec[3]);
 rdir->GetObject("TEC/Nstp_rphi_layer5tec",nstptec[4]);
 rdir->GetObject("TEC/Nstp_rphi_layer6tec",nstptec[5]);
 rdir->GetObject("TEC/Nstp_rphi_layer7tec",nstptec[6]);
 rdir->GetObject("TEC/Nstp_sas_layer1tec",nstptec[7]);
 rdir->GetObject("TEC/Nstp_sas_layer2tec",nstptec[8]);
 rdir->GetObject("TEC/Nstp_sas_layer5tec",nstptec[9]);
 sdir->GetObject("TEC/Nstp_rphi_layer1tec",newnstptec[0]);
 sdir->GetObject("TEC/Nstp_rphi_layer2tec",newnstptec[1]);
 sdir->GetObject("TEC/Nstp_rphi_layer3tec",newnstptec[2]);
 sdir->GetObject("TEC/Nstp_rphi_layer4tec",newnstptec[3]);
 sdir->GetObject("TEC/Nstp_rphi_layer5tec",newnstptec[4]);
 sdir->GetObject("TEC/Nstp_rphi_layer6tec",newnstptec[5]);
 sdir->GetObject("TEC/Nstp_rphi_layer7tec",newnstptec[6]);
 sdir->GetObject("TEC/Nstp_sas_layer1tec",newnstptec[7]);
 sdir->GetObject("TEC/Nstp_sas_layer2tec",newnstptec[8]);
 sdir->GetObject("TEC/Nstp_sas_layer5tec",newnstptec[9]);

 Strip23 = new TCanvas("Strip23","Strip23",800,500);
 Strip23->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip23->cd(i+1);
   nstptec[i]->SetLineColor(2);
   newnstptec[i]->SetLineColor(4);
   newnstptec[i]->SetLineStyle(2);
   nstptec[i]->Sumw2();
   newnstptec[i]->Sumw2();
   nstptec[i]->SetNormFactor(1.0);
   newnstptec[i]->SetNormFactor(1.0);
   nstptec[i]->Draw("h");
   newnstptec[i]->Draw("sameh");
   myPV->PVCompute(nstptec[i] , newnstptec[i] , te );
    leg.Clear();
   leg.AddEntry(nstptec[i],rver , "l");
   leg.AddEntry(newnstptec[i],cver , "l");
   leg.Draw();
}
 
 Strip23->Print("NstpTECCompare.eps");
 Strip23->Print("NstpTECCompare.gif");

 rdir->GetObject("TEC/Posx_rphi_layer1tec",postec[0]);
 rdir->GetObject("TEC/Posx_rphi_layer2tec",postec[1]);
 rdir->GetObject("TEC/Posx_rphi_layer3tec",postec[2]);
 rdir->GetObject("TEC/Posx_rphi_layer4tec",postec[3]);
 rdir->GetObject("TEC/Posx_rphi_layer5tec",postec[4]);
 rdir->GetObject("TEC/Posx_rphi_layer6tec",postec[5]);
 rdir->GetObject("TEC/Posx_rphi_layer7tec",postec[6]);
 rdir->GetObject("TEC/Posx_sas_layer1tec",postec[7]);
 rdir->GetObject("TEC/Posx_sas_layer2tec",postec[8]);
 rdir->GetObject("TEC/Posx_sas_layer5tec",postec[9]);
 sdir->GetObject("TEC/Posx_rphi_layer1tec",newpostec[0]);
 sdir->GetObject("TEC/Posx_rphi_layer2tec",newpostec[1]);
 sdir->GetObject("TEC/Posx_rphi_layer3tec",newpostec[2]);
 sdir->GetObject("TEC/Posx_rphi_layer4tec",newpostec[3]);
 sdir->GetObject("TEC/Posx_rphi_layer5tec",newpostec[4]);
 sdir->GetObject("TEC/Posx_rphi_layer6tec",newpostec[5]);
 sdir->GetObject("TEC/Posx_rphi_layer7tec",newpostec[6]);
 sdir->GetObject("TEC/Posx_sas_layer1tec",newpostec[7]);
 sdir->GetObject("TEC/Posx_sas_layer2tec",newpostec[8]);
 sdir->GetObject("TEC/Posx_sas_layer5tec",newpostec[9]);

 Strip24 = new TCanvas("Strip24","Strip24",800,500);
  Strip24->Divide(4,3);
  for (Int_t i=0; i<10; i++) {
    Strip24->cd(i+1);
    postec[i]->SetLineColor(2);
    newpostec[i]->SetLineColor(4);
    newpostec[i]->SetLineStyle(2);
    postec[i]->Sumw2();
    newpostec[i]->Sumw2();
    postec[i]->SetNormFactor(1.0);
    newpostec[i]->SetNormFactor(1.0);
    postec[i]->Draw("h");
    newpostec[i]->Draw("sameh");
    myPV->PVCompute(postec[i] , newpostec[i] , te );
     leg.Clear();
   leg.AddEntry(postec[i],rver , "l");
   leg.AddEntry(newpostec[i],cver , "l");
   leg.Draw();
}
  
  Strip24->Print("PosTECCompare.eps");
  Strip24->Print("PosTECCompare.gif");
  

 rdir->GetObject("TEC/Errx_rphi_layer1tec",errxtec[0]);
 rdir->GetObject("TEC/Errx_rphi_layer2tec",errxtec[1]);
 rdir->GetObject("TEC/Errx_rphi_layer3tec",errxtec[2]);
 rdir->GetObject("TEC/Errx_rphi_layer4tec",errxtec[3]);
 rdir->GetObject("TEC/Errx_rphi_layer5tec",errxtec[4]);
 rdir->GetObject("TEC/Errx_rphi_layer6tec",errxtec[5]);
 rdir->GetObject("TEC/Errx_rphi_layer7tec",errxtec[6]);
 rdir->GetObject("TEC/Errx_sas_layer1tec",errxtec[7]);
 rdir->GetObject("TEC/Errx_sas_layer2tec",errxtec[8]);
 rdir->GetObject("TEC/Errx_sas_layer5tec",errxtec[9]);
 sdir->GetObject("TEC/Errx_rphi_layer1tec",newerrxtec[0]);
 sdir->GetObject("TEC/Errx_rphi_layer2tec",newerrxtec[1]);
 sdir->GetObject("TEC/Errx_rphi_layer3tec",newerrxtec[2]);
 sdir->GetObject("TEC/Errx_rphi_layer4tec",newerrxtec[3]);
 sdir->GetObject("TEC/Errx_rphi_layer5tec",newerrxtec[4]);
 sdir->GetObject("TEC/Errx_rphi_layer6tec",newerrxtec[5]);
 sdir->GetObject("TEC/Errx_rphi_layer7tec",newerrxtec[6]);
 sdir->GetObject("TEC/Errx_sas_layer1tec",newerrxtec[7]);
 sdir->GetObject("TEC/Errx_sas_layer2tec",newerrxtec[8]);
 sdir->GetObject("TEC/Errx_sas_layer5tec",newerrxtec[9]);


  Strip25 = new TCanvas("Strip25","Strip25",800,500);
  Strip25->Divide(4,3);
  for (Int_t i=0; i<10; i++) {
    Strip25->cd(i+1);
    errxtec[i]->SetLineColor(2);
    newerrxtec[i]->SetLineColor(4);
    newerrxtec[i]->SetLineStyle(2);
    errxtec[i]->Sumw2();
    newerrxtec[i]->Sumw2();
    errxtec[i]->SetNormFactor(1.0);
    newerrxtec[i]->SetNormFactor(1.0);
    errxtec[i]->Draw("h");
    newerrxtec[i]->Draw("sameh");
    myPV->PVCompute(errxtec[i] , newerrxtec[i] , te );
    leg.Clear();
   leg.AddEntry(errxtec[i],rver , "l");
   leg.AddEntry(newerrxtec[i],cver , "l");
   leg.Draw();
 }
  
  Strip25->Print("ErrxTECCompare.eps");
  Strip25->Print("ErrxTECCompare.gif");
  
 
 rdir->GetObject("TEC/Res_rphi_layer1tec",restec[0]);
 rdir->GetObject("TEC/Res_rphi_layer2tec",restec[1]);
 rdir->GetObject("TEC/Res_rphi_layer3tec",restec[2]);
 rdir->GetObject("TEC/Res_rphi_layer4tec",restec[3]);
 rdir->GetObject("TEC/Res_rphi_layer5tec",restec[4]);
 rdir->GetObject("TEC/Res_rphi_layer6tec",restec[5]);
 rdir->GetObject("TEC/Res_rphi_layer7tec",restec[6]);
 rdir->GetObject("TEC/Res_sas_layer1tec",restec[7]);
 rdir->GetObject("TEC/Res_sas_layer2tec",restec[8]);
 rdir->GetObject("TEC/Res_sas_layer5tec",restec[9]);
 sdir->GetObject("TEC/Res_rphi_layer1tec",newrestec[0]);
 sdir->GetObject("TEC/Res_rphi_layer2tec",newrestec[1]);
 sdir->GetObject("TEC/Res_rphi_layer3tec",newrestec[2]);
 sdir->GetObject("TEC/Res_rphi_layer4tec",newrestec[3]);
 sdir->GetObject("TEC/Res_rphi_layer5tec",newrestec[4]);
 sdir->GetObject("TEC/Res_rphi_layer6tec",newrestec[5]);
 sdir->GetObject("TEC/Res_rphi_layer7tec",newrestec[6]);
 sdir->GetObject("TEC/Res_sas_layer1tec",newrestec[7]);
 sdir->GetObject("TEC/Res_sas_layer2tec",newrestec[8]);
 sdir->GetObject("TEC/Res_sas_layer5tec",newrestec[9]);

  Strip26 = new TCanvas("Strip26","Strip26",800,500);
  Strip26->Divide(4,3);
  for (Int_t i=0; i<10; i++) {
    Strip26->cd(i+1);
    restec[i]->SetLineColor(2);
    newrestec[i]->SetLineColor(4);
    newrestec[i]->SetLineStyle(2);
    restec[i]->Sumw2();
    newrestec[i]->Sumw2();
    restec[i]->SetNormFactor(1.0);
    newrestec[i]->SetNormFactor(1.0);
    restec[i]->Draw("h");
    newrestec[i]->Draw("sameh");
    myPV->PVCompute(restec[i] , newrestec[i] , te );
    leg.Clear();
   leg.AddEntry(restec[i],rver , "l");
   leg.AddEntry(newrestec[i],cver , "l");
   leg.Draw();
 }
  
  Strip26->Print("ResTECCompare.eps");
  Strip26->Print("ResTECCompare.gif");

  rdir->GetObject("TEC/Pull_MF_rphi_layer1tec",pulltec[0]);
  rdir->GetObject("TEC/Pull_MF_rphi_layer2tec",pulltec[1]);
  rdir->GetObject("TEC/Pull_MF_rphi_layer3tec",pulltec[2]);
  rdir->GetObject("TEC/Pull_MF_rphi_layer4tec",pulltec[3]);
  rdir->GetObject("TEC/Pull_MF_rphi_layer5tec",pulltec[4]);
  rdir->GetObject("TEC/Pull_MF_rphi_layer6tec",pulltec[5]);
  rdir->GetObject("TEC/Pull_MF_rphi_layer7tec",pulltec[6]);
  rdir->GetObject("TEC/Pull_MF_sas_layer1tec",pulltec[7]);
  rdir->GetObject("TEC/Pull_MF_sas_layer2tec",pulltec[8]);
  rdir->GetObject("TEC/Pull_MF_sas_layer5tec",pulltec[9]);
  sdir->GetObject("TEC/Pull_MF_rphi_layer1tec",newpulltec[0]);
  sdir->GetObject("TEC/Pull_MF_rphi_layer2tec",newpulltec[1]);
  sdir->GetObject("TEC/Pull_MF_rphi_layer3tec",newpulltec[2]);
  sdir->GetObject("TEC/Pull_MF_rphi_layer4tec",newpulltec[3]);
  sdir->GetObject("TEC/Pull_MF_rphi_layer5tec",newpulltec[4]);
  sdir->GetObject("TEC/Pull_MF_rphi_layer6tec",newpulltec[5]);
  sdir->GetObject("TEC/Pull_MF_rphi_layer7tec",newpulltec[6]);
  sdir->GetObject("TEC/Pull_MF_sas_layer1tec",newpulltec[7]);
  sdir->GetObject("TEC/Pull_MF_sas_layer2tec",newpulltec[8]);
  sdir->GetObject("TEC/Pull_MF_sas_layer5tec",newpulltec[9]);
  
  Strip34 = new TCanvas("Strip34","Strip34",800,500);
  Strip34->Divide(4,3);
  for (Int_t i=0; i<10; i++) {
    Strip34->cd(i+1);
    pulltec[i]->SetLineColor(2);
    newpulltec[i]->SetLineColor(4);
    newpulltec[i]->SetLineStyle(2);
    pulltec[i]->Sumw2();
    newpulltec[i]->Sumw2();
    pulltec[i]->SetNormFactor(1.0);
    newpulltec[i]->SetNormFactor(1.0);
    pulltec[i]->Draw("h");
    newpulltec[i]->Draw("sameh");
    myPV->PVCompute(pulltec[i] , newpulltec[i] , te );
     leg.Clear();
   leg.AddEntry(pulltec[i],rver , "l");
   leg.AddEntry(newpulltec[i],cver , "l");
   leg.Draw();
}
  
  Strip34->Print("PullTECCompare.eps");
  Strip34->Print("PullTECCompare.gif");

 rdir->GetObject("TEC/Chi2_rphi_layer1tec",chi2tec[0]);
 rdir->GetObject("TEC/Chi2_rphi_layer2tec",chi2tec[1]);
 rdir->GetObject("TEC/Chi2_rphi_layer3tec",chi2tec[2]);
 rdir->GetObject("TEC/Chi2_rphi_layer4tec",chi2tec[3]);
 rdir->GetObject("TEC/Chi2_rphi_layer5tec",chi2tec[4]);
 rdir->GetObject("TEC/Chi2_rphi_layer6tec",chi2tec[5]);
 rdir->GetObject("TEC/Chi2_rphi_layer7tec",chi2tec[6]);
 rdir->GetObject("TEC/Chi2_sas_layer1tec",chi2tec[7]);
 rdir->GetObject("TEC/Chi2_sas_layer2tec",chi2tec[8]);
 rdir->GetObject("TEC/Chi2_sas_layer5tec",chi2tec[9]);
 sdir->GetObject("TEC/Chi2_rphi_layer1tec",newchi2tec[0]);
 sdir->GetObject("TEC/Chi2_rphi_layer2tec",newchi2tec[1]);
 sdir->GetObject("TEC/Chi2_rphi_layer3tec",newchi2tec[2]);
 sdir->GetObject("TEC/Chi2_rphi_layer4tec",newchi2tec[3]);
 sdir->GetObject("TEC/Chi2_rphi_layer5tec",newchi2tec[4]);
 sdir->GetObject("TEC/Chi2_rphi_layer6tec",newchi2tec[5]);
 sdir->GetObject("TEC/Chi2_rphi_layer7tec",newchi2tec[6]);
 sdir->GetObject("TEC/Chi2_sas_layer1tec",newchi2tec[7]);
 sdir->GetObject("TEC/Chi2_sas_layer2tec",newchi2tec[8]);
 sdir->GetObject("TEC/Chi2_sas_layer5tec",newchi2tec[9]);

  Strip27 = new TCanvas("Strip27","Strip27",800,500);
  Strip27->Divide(4,3);
  for (Int_t i=0; i<10; i++) {
    Strip27->cd(i+1);
    chi2tec[i]->SetLineColor(2);
    newchi2tec[i]->SetLineColor(4);
    newchi2tec[i]->SetLineStyle(2);
    chi2tec[i]->Sumw2();
    newchi2tec[i]->Sumw2();
    chi2tec[i]->SetNormFactor(1.0);
    newchi2tec[i]->SetNormFactor(1.0);
    chi2tec[i]->Draw("h");
    newchi2tec[i]->Draw("sameh");
    myPV->PVCompute(chi2tec[i] , newchi2tec[i] , te );
     leg.Clear();
   leg.AddEntry(chi2tec[i],rver , "l");
   leg.AddEntry(newchi2tec[i],cver , "l");
   leg.Draw();
}
  
  Strip27->Print("Chi2TECCompare.eps");
  Strip27->Print("Chi2TECCompare.gif");
  
 rdir->GetObject("TEC/Posx_matched_layer1tec",matchedtec[0]);
 rdir->GetObject("TEC/Posy_matched_layer1tec",matchedtec[1]);
 rdir->GetObject("TEC/Posx_matched_layer2tec",matchedtec[2]);
 rdir->GetObject("TEC/Posy_matched_layer2tec",matchedtec[3]);
 rdir->GetObject("TEC/Posx_matched_layer5tec",matchedtec[4]);
 rdir->GetObject("TEC/Posy_matched_layer5tec",matchedtec[5]);
 rdir->GetObject("TEC/Errx_matched_layer1tec",matchedtec[6]);
 rdir->GetObject("TEC/Erry_matched_layer1tec",matchedtec[7]);
 rdir->GetObject("TEC/Errx_matched_layer2tec",matchedtec[8]);
 rdir->GetObject("TEC/Erry_matched_layer2tec",matchedtec[9]);
 rdir->GetObject("TEC/Errx_matched_layer5tec",matchedtec[10]);
 rdir->GetObject("TEC/Erry_matched_layer5tec",matchedtec[11]);
 sdir->GetObject("TEC/Posx_matched_layer1tec",newmatchedtec[0]);
 sdir->GetObject("TEC/Posy_matched_layer1tec",newmatchedtec[1]);
 sdir->GetObject("TEC/Posx_matched_layer2tec",newmatchedtec[2]);
 sdir->GetObject("TEC/Posy_matched_layer2tec",newmatchedtec[3]);
 sdir->GetObject("TEC/Posx_matched_layer5tec",newmatchedtec[4]);
 sdir->GetObject("TEC/Posy_matched_layer5tec",newmatchedtec[5]);
 sdir->GetObject("TEC/Errx_matched_layer1tec",newmatchedtec[6]);
 sdir->GetObject("TEC/Erry_matched_layer1tec",newmatchedtec[7]);
 sdir->GetObject("TEC/Errx_matched_layer2tec",newmatchedtec[8]);
 sdir->GetObject("TEC/Erry_matched_layer2tec",newmatchedtec[9]);
 sdir->GetObject("TEC/Errx_matched_layer5tec",newmatchedtec[10]);
 sdir->GetObject("TEC/Erry_matched_layer5tec",newmatchedtec[11]);

 Strip28 = new TCanvas("Strip28","Strip28",800,500);
 Strip28->Divide(4,3);
 for (Int_t i=0; i<12; i++) {
   Strip28->cd(i+1);
   matchedtec[i]->SetLineColor(2);
   newmatchedtec[i]->SetLineColor(4);
   newmatchedtec[i]->SetLineStyle(2);
   matchedtec[i]->Sumw2();
   newmatchedtec[i]->Sumw2();
   matchedtec[i]->SetNormFactor(1.0);
   newmatchedtec[i]->SetNormFactor(1.0);
   matchedtec[i]->Draw("h");
   newmatchedtec[i]->Draw("sameh");
   myPV->PVCompute(matchedtec[i] , newmatchedtec[i] , te );
    leg.Clear();
   leg.AddEntry(matchedtec[i],rver , "l");
   leg.AddEntry(newmatchedtec[i],cver , "l");
   leg.Draw();
}
 
 Strip28->Print("MatchedTECCompare.eps");
 Strip28->Print("MatchedTECCompare.gif");
 
 rdir->GetObject("TEC/Resx_matched_layer1tec",matchedrestec[0]);
 rdir->GetObject("TEC/Resy_matched_layer1tec",matchedrestec[1]);
 rdir->GetObject("TEC/Resx_matched_layer2tec",matchedrestec[2]);
 rdir->GetObject("TEC/Resy_matched_layer2tec",matchedrestec[3]);
 rdir->GetObject("TEC/Resx_matched_layer5tec",matchedrestec[4]);
 rdir->GetObject("TEC/Resy_matched_layer5tec",matchedrestec[5]);
 sdir->GetObject("TEC/Resx_matched_layer1tec",newmatchedrestec[0]);
 sdir->GetObject("TEC/Resy_matched_layer1tec",newmatchedrestec[1]);
 sdir->GetObject("TEC/Resx_matched_layer2tec",newmatchedrestec[2]);
 sdir->GetObject("TEC/Resy_matched_layer2tec",newmatchedrestec[3]);
 sdir->GetObject("TEC/Resx_matched_layer5tec",newmatchedrestec[4]);
 sdir->GetObject("TEC/Resy_matched_layer5tec",newmatchedrestec[5]);

 Strip29 = new TCanvas("Strip29","Strip29",800,500);
 Strip29->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   Strip29->cd(i+1);
   matchedrestec[i]->SetLineColor(2);
   newmatchedrestec[i]->SetLineColor(4);
   newmatchedrestec[i]->SetLineStyle(2);
   matchedrestec[i]->Sumw2();
   newmatchedrestec[i]->Sumw2();
   matchedrestec[i]->SetNormFactor(1.0);
   newmatchedrestec[i]->SetNormFactor(1.0);
   matchedrestec[i]->Draw("h");
   newmatchedrestec[i]->Draw("sameh");
   myPV->PVCompute(matchedrestec[i] , newmatchedrestec[i] , te );
    leg.Clear();
   leg.AddEntry(matchedrestec[i],rver , "l");
   leg.AddEntry(newmatchedrestec[i],cver , "l");
   leg.Draw();
}
 
 Strip29->Print("MatchedResTECCompare.eps");
 Strip29->Print("MatchedResTECCompare.gif");

 rdir->GetObject("TEC/Chi2_matched_layer1tec",matchedchi2tec[0]);
 rdir->GetObject("TEC/Chi2_matched_layer2tec",matchedchi2tec[1]);
 rdir->GetObject("TEC/Chi2_matched_layer5tec",matchedchi2tec[2]);
 sdir->GetObject("TEC/Chi2_matched_layer1tec",newmatchedchi2tec[0]);
 sdir->GetObject("TEC/Chi2_matched_layer2tec",newmatchedchi2tec[1]);
 sdir->GetObject("TEC/Chi2_matched_layer5tec",newmatchedchi2tec[2]);
 
 Strip30 = new TCanvas("Strip30","Strip30",800,500);
 Strip30->Divide(1,3);
 for (Int_t i=0; i<3; i++) {
   Strip30->cd(i+1);
   matchedchi2tec[i]->SetLineColor(2);
   newmatchedchi2tec[i]->SetLineColor(4);
   newmatchedchi2tec[i]->SetLineStyle(2);
   matchedchi2tec[i]->Sumw2();
   newmatchedchi2tec[i]->Sumw2();
   matchedchi2tec[i]->SetNormFactor(1.0);
   newmatchedchi2tec[i]->SetNormFactor(1.0);
   matchedchi2tec[i]->Draw("h");
   newmatchedchi2tec[i]->Draw("sameh");
   myPV->PVCompute(matchedchi2tec[i] , newmatchedchi2tec[i] , te );
    leg.Clear();
   leg.AddEntry(matchedchi2tec[i],rver , "l");
   leg.AddEntry(newmatchedchi2tec[i],cver , "l");
   leg.Draw();
}
 
 Strip30->Print("MatchedChi2TECCompare.eps");
 Strip30->Print("MatchedChi2TECCompare.gif");

 
}


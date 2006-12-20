void SiStripRecHitsCompare()
{

 gROOT ->Reset();
 char*  rfilename = "sistriprechitshisto.root";
 char*  sfilename = "../data/sistriprechitshisto.root";

 delete gROOT->GetListOfFiles()->FindObject(rfilename);
 delete gROOT->GetListOfFiles()->FindObject(sfilename); 

 TText* te = new TText();
 TFile * rfile = new TFile(rfilename);
 TFile * sfile = new TFile(sfilename);

 rfile->cd("DQMData");
 sfile->cd("DQMData");

 Char_t histo[200];

 gROOT->ProcessLine(".x HistoCompare.C");
 HistoCompare * myPV = new HistoCompare();

 TCanvas *Strip;

 //=============================================================== 
// TIB
 
 TH1F* adctib[6];
 TH1F* nstptib[6];
 TH1F* postib[6];
 TH1F* errxtib[6];
 TH1F* restib[6];
 TH1F* chi2tib[6];
 TH1F* matchedtib[12];

 TH1F* newadctib[6];
 TH1F* newnstptib[6];
 TH1F* newpostib[6];
 TH1F* newerrxtib[6];
 TH1F* newrestib[6];
 TH1F* newchi2tib[6];
 TH1F* newmatchedtib[12];
 
 rfile->GetObject("DQMData/TIB/Adc_rphi_layer1tib",adctib[0]);
 rfile->GetObject("DQMData/TIB/Adc_rphi_layer2tib",adctib[1]);
 rfile->GetObject("DQMData/TIB/Adc_rphi_layer3tib",adctib[2]);
 rfile->GetObject("DQMData/TIB/Adc_rphi_layer4tib",adctib[3]);
 rfile->GetObject("DQMData/TIB/Adc_sas_layer1tib",adctib[4]);
 rfile->GetObject("DQMData/TIB/Adc_sas_layer2tib",adctib[5]);
 sfile->GetObject("DQMData/TIB/Adc_rphi_layer1tib",newadctib[0]);
 sfile->GetObject("DQMData/TIB/Adc_rphi_layer2tib",newadctib[1]);
 sfile->GetObject("DQMData/TIB/Adc_rphi_layer3tib",newadctib[2]);
 sfile->GetObject("DQMData/TIB/Adc_rphi_layer4tib",newadctib[3]);
 sfile->GetObject("DQMData/TIB/Adc_sas_layer1tib",newadctib[4]);
 sfile->GetObject("DQMData/TIB/Adc_sas_layer2tib",newadctib[5]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   Strip->cd(i+1);
   adctib[i]->SetLineColor(2);
   newadctib[i]->SetLineColor(4);
    newadctib[i]->SetLineStyle(2);
    adctib[i]->Draw();
    newadctib[i]->Draw("sames");
    myPV->PVCompute(adctib[i] , newadctib[i] , te );
 }
 
 Strip->Print("AdcTIBCompare.eps");
 
 rfile->GetObject("DQMData/TIB/Nstp_rphi_layer1tib",nstptib[0]);
 rfile->GetObject("DQMData/TIB/Nstp_rphi_layer2tib",nstptib[1]);
 rfile->GetObject("DQMData/TIB/Nstp_rphi_layer3tib",nstptib[2]);
 rfile->GetObject("DQMData/TIB/Nstp_rphi_layer4tib",nstptib[3]);
 rfile->GetObject("DQMData/TIB/Nstp_sas_layer1tib",nstptib[4]);
 rfile->GetObject("DQMData/TIB/Nstp_sas_layer2tib",nstptib[5]);
 sfile->GetObject("DQMData/TIB/Nstp_rphi_layer1tib",newnstptib[0]);
 sfile->GetObject("DQMData/TIB/Nstp_rphi_layer2tib",newnstptib[1]);
 sfile->GetObject("DQMData/TIB/Nstp_rphi_layer3tib",newnstptib[2]);
 sfile->GetObject("DQMData/TIB/Nstp_rphi_layer4tib",newnstptib[3]);
 sfile->GetObject("DQMData/TIB/Nstp_sas_layer1tib",newnstptib[4]);
 sfile->GetObject("DQMData/TIB/Nstp_sas_layer2tib",newnstptib[5]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   Strip->cd(i+1);
   nstptib[i]->SetLineColor(2);
   newnstptib[i]->SetLineColor(4);
   newnstptib[i]->SetLineStyle(2);
   nstptib[i]->Draw();
   newnstptib[i]->Draw("sames");
   myPV->PVCompute(nstptib[i] , newnstptib[i] , te );
 }
 
 Strip->Print("NstpTIBCompare.eps");

 rfile->GetObject("DQMData/TIB/Posx_rphi_layer1tib",postib[0]);
 rfile->GetObject("DQMData/TIB/Posx_rphi_layer2tib",postib[1]);
 rfile->GetObject("DQMData/TIB/Posx_rphi_layer3tib",postib[2]);
 rfile->GetObject("DQMData/TIB/Posx_rphi_layer4tib",postib[3]);
 rfile->GetObject("DQMData/TIB/Posx_sas_layer1tib",postib[4]);
 rfile->GetObject("DQMData/TIB/Posx_sas_layer2tib",postib[5]);
 sfile->GetObject("DQMData/TIB/Posx_rphi_layer1tib",newpostib[0]);
 sfile->GetObject("DQMData/TIB/Posx_rphi_layer2tib",newpostib[1]);
 sfile->GetObject("DQMData/TIB/Posx_rphi_layer3tib",newpostib[2]);
 sfile->GetObject("DQMData/TIB/Posx_rphi_layer4tib",newpostib[3]);
 sfile->GetObject("DQMData/TIB/Posx_sas_layer1tib",newpostib[4]);
 sfile->GetObject("DQMData/TIB/Posx_sas_layer2tib",newpostib[5]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<6; i++) {
    Strip->cd(i+1);
    postib[i]->SetLineColor(2);
    newpostib[i]->SetLineColor(4);
    newpostib[i]->SetLineStyle(2);
    postib[i]->Draw();
    newpostib[i]->Draw("sames");
    myPV->PVCompute(postib[i] , newpostib[i] , te );
  }
  
  Strip->Print("PosTIBCompare.eps");
  

 rfile->GetObject("DQMData/TIB/Errx_rphi_layer1tib",errxtib[0]);
 rfile->GetObject("DQMData/TIB/Errx_rphi_layer2tib",errxtib[1]);
 rfile->GetObject("DQMData/TIB/Errx_rphi_layer3tib",errxtib[2]);
 rfile->GetObject("DQMData/TIB/Errx_rphi_layer4tib",errxtib[3]);
 rfile->GetObject("DQMData/TIB/Errx_sas_layer1tib",errxtib[4]);
 rfile->GetObject("DQMData/TIB/Errx_sas_layer2tib",errxtib[5]);
 sfile->GetObject("DQMData/TIB/Errx_rphi_layer1tib",newerrxtib[0]);
 sfile->GetObject("DQMData/TIB/Errx_rphi_layer2tib",newerrxtib[1]);
 sfile->GetObject("DQMData/TIB/Errx_rphi_layer3tib",newerrxtib[2]);
 sfile->GetObject("DQMData/TIB/Errx_rphi_layer4tib",newerrxtib[3]);
 sfile->GetObject("DQMData/TIB/Errx_sas_layer1tib",newerrxtib[4]);
 sfile->GetObject("DQMData/TIB/Errx_sas_layer2tib",newerrxtib[5]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<6; i++) {
    Strip->cd(i+1);
    errxtib[i]->SetLineColor(2);
    newerrxtib[i]->SetLineColor(4);
    newerrxtib[i]->SetLineStyle(2);
    errxtib[i]->Draw();
    newerrxtib[i]->Draw("sames");
    myPV->PVCompute(errxtib[i] , newerrxtib[i] , te );
  }
  
  Strip->Print("ErrxTIBCompare.eps");
  
 
 rfile->GetObject("DQMData/TIB/Res_rphi_layer1tib",restib[0]);
 rfile->GetObject("DQMData/TIB/Res_rphi_layer2tib",restib[1]);
 rfile->GetObject("DQMData/TIB/Res_rphi_layer3tib",restib[2]);
 rfile->GetObject("DQMData/TIB/Res_rphi_layer4tib",restib[3]);
 rfile->GetObject("DQMData/TIB/Res_sas_layer1tib",restib[4]);
 rfile->GetObject("DQMData/TIB/Res_sas_layer2tib",restib[5]);
 sfile->GetObject("DQMData/TIB/Res_rphi_layer1tib",newrestib[0]);
 sfile->GetObject("DQMData/TIB/Res_rphi_layer2tib",newrestib[1]);
 sfile->GetObject("DQMData/TIB/Res_rphi_layer3tib",newrestib[2]);
 sfile->GetObject("DQMData/TIB/Res_rphi_layer4tib",newrestib[3]);
 sfile->GetObject("DQMData/TIB/Res_sas_layer1tib",newrestib[4]);
 sfile->GetObject("DQMData/TIB/Res_sas_layer2tib",newrestib[5]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<6; i++) {
    Strip->cd(i+1);
    restib[i]->SetLineColor(2);
    newrestib[i]->SetLineColor(4);
    newrestib[i]->SetLineStyle(2);
    restib[i]->Draw();
    newrestib[i]->Draw("sames");
    myPV->PVCompute(restib[i] , newrestib[i] , te );
  }
  
  Strip->Print("ResTIBCompare.eps");

 rfile->GetObject("DQMData/TIB/Chi2_rphi_layer1tib",chi2tib[0]);
 rfile->GetObject("DQMData/TIB/Chi2_rphi_layer2tib",chi2tib[1]);
 rfile->GetObject("DQMData/TIB/Chi2_rphi_layer3tib",chi2tib[2]);
 rfile->GetObject("DQMData/TIB/Chi2_rphi_layer4tib",chi2tib[3]);
 rfile->GetObject("DQMData/TIB/Chi2_sas_layer1tib",chi2tib[4]);
 rfile->GetObject("DQMData/TIB/Chi2_sas_layer2tib",chi2tib[5]);
 sfile->GetObject("DQMData/TIB/Chi2_rphi_layer1tib",newchi2tib[0]);
 sfile->GetObject("DQMData/TIB/Chi2_rphi_layer2tib",newchi2tib[1]);
 sfile->GetObject("DQMData/TIB/Chi2_rphi_layer3tib",newchi2tib[2]);
 sfile->GetObject("DQMData/TIB/Chi2_rphi_layer4tib",newchi2tib[3]);
 sfile->GetObject("DQMData/TIB/Chi2_sas_layer1tib",newchi2tib[4]);
 sfile->GetObject("DQMData/TIB/Chi2_sas_layer2tib",newchi2tib[5]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<6; i++) {
    Strip->cd(i+1);
    chi2tib[i]->SetLineColor(2);
    newchi2tib[i]->SetLineColor(4);
    newchi2tib[i]->SetLineStyle(2);
    chi2tib[i]->Draw();
    newchi2tib[i]->Draw("sames");
    myPV->PVCompute(chi2tib[i] , newchi2tib[i] , te );
  }
  
  Strip->Print("Chi2TIBCompare.eps");

  
 rfile->GetObject("DQMData/TIB/Posx_matched_layer1tib",matchedtib[0]);
 rfile->GetObject("DQMData/TIB/Posy_matched_layer1tib",matchedtib[1]);
 rfile->GetObject("DQMData/TIB/Posx_matched_layer2tib",matchedtib[2]);
 rfile->GetObject("DQMData/TIB/Posy_matched_layer2tib",matchedtib[3]);
 rfile->GetObject("DQMData/TIB/Errx_matched_layer1tib",matchedtib[4]);
 rfile->GetObject("DQMData/TIB/Erry_matched_layer1tib",matchedtib[5]);
 rfile->GetObject("DQMData/TIB/Errx_matched_layer2tib",matchedtib[6]);
 rfile->GetObject("DQMData/TIB/Erry_matched_layer2tib",matchedtib[7]);
 rfile->GetObject("DQMData/TIB/Resx_matched_layer1tib",matchedtib[8]);
 rfile->GetObject("DQMData/TIB/Resy_matched_layer1tib",matchedtib[9]);
 rfile->GetObject("DQMData/TIB/Resx_matched_layer2tib",matchedtib[10]);
 rfile->GetObject("DQMData/TIB/Resy_matched_layer2tib",matchedtib[11]);
 sfile->GetObject("DQMData/TIB/Posx_matched_layer1tib",newmatchedtib[0]);
 sfile->GetObject("DQMData/TIB/Posy_matched_layer1tib",newmatchedtib[1]);
 sfile->GetObject("DQMData/TIB/Posx_matched_layer2tib",newmatchedtib[2]);
 sfile->GetObject("DQMData/TIB/Posy_matched_layer2tib",newmatchedtib[3]);
 sfile->GetObject("DQMData/TIB/Errx_matched_layer1tib",newmatchedtib[4]);
 sfile->GetObject("DQMData/TIB/Erry_matched_layer1tib",newmatchedtib[5]);
 sfile->GetObject("DQMData/TIB/Errx_matched_layer2tib",newmatchedtib[6]);
 sfile->GetObject("DQMData/TIB/Erry_matched_layer2tib",newmatchedtib[7]);
 sfile->GetObject("DQMData/TIB/Resx_matched_layer1tib",newmatchedtib[8]);
 sfile->GetObject("DQMData/TIB/Resy_matched_layer1tib",newmatchedtib[9]);
 sfile->GetObject("DQMData/TIB/Resx_matched_layer2tib",newmatchedtib[10]);
 sfile->GetObject("DQMData/TIB/Resy_matched_layer2tib",newmatchedtib[11]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<12; i++) {
   Strip->cd(i+1);
   matchedtib[i]->SetLineColor(2);
   newmatchedtib[i]->SetLineColor(4);
   newmatchedtib[i]->SetLineStyle(2);
   matchedtib[i]->Draw();
   newmatchedtib[i]->Draw("sames");
   myPV->PVCompute(matchedtib[i] , newmatchedtib[i] , te );
 }
 
 Strip->Print("MatchedTIBCompare.eps");

 
 //======================================================================================================
// TOB
 
 TH1F* adctob[8];
 TH1F* nstptob[8];
 TH1F* postob[8];
 TH1F* errxtob[8];
 TH1F* restob[8];
 TH1F* chi2tob[8];
 TH1F* matchedtob[12];
 TH1F* newadctob[8];
 TH1F* newnstptob[8];
 TH1F* newpostob[8];
 TH1F* newerrxtob[8];
 TH1F* newrestob[8];
 TH1F* newchi2tob[8];
 TH1F* newmatchedtob[12];
 
 rfile->GetObject("DQMData/TOB/Adc_rphi_layer1tob",adctob[0]);
 rfile->GetObject("DQMData/TOB/Adc_rphi_layer2tob",adctob[1]);
 rfile->GetObject("DQMData/TOB/Adc_rphi_layer3tob",adctob[2]);
 rfile->GetObject("DQMData/TOB/Adc_rphi_layer4tob",adctob[3]);
 rfile->GetObject("DQMData/TOB/Adc_rphi_layer5tob",adctob[4]);
 rfile->GetObject("DQMData/TOB/Adc_rphi_layer6tob",adctob[5]);
 rfile->GetObject("DQMData/TOB/Adc_sas_layer1tob",adctob[6]);
 rfile->GetObject("DQMData/TOB/Adc_sas_layer2tob",adctob[7]);
 sfile->GetObject("DQMData/TOB/Adc_rphi_layer1tob",newadctob[0]);
 sfile->GetObject("DQMData/TOB/Adc_rphi_layer2tob",newadctob[1]);
 sfile->GetObject("DQMData/TOB/Adc_rphi_layer3tob",newadctob[2]);
 sfile->GetObject("DQMData/TOB/Adc_rphi_layer4tob",newadctob[3]);
 sfile->GetObject("DQMData/TOB/Adc_rphi_layer5tob",newadctob[4]);
 sfile->GetObject("DQMData/TOB/Adc_rphi_layer6tob",newadctob[5]);
 sfile->GetObject("DQMData/TOB/Adc_sas_layer1tob",newadctob[6]);
 sfile->GetObject("DQMData/TOB/Adc_sas_layer2tob",newadctob[7]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   Strip->cd(i+1);
   adctob[i]->SetLineColor(2);
   newadctob[i]->SetLineColor(4);
    newadctob[i]->SetLineStyle(2);
    adctob[i]->Draw();
    newadctob[i]->Draw("sames");
    myPV->PVCompute(adctob[i] , newadctob[i] , te );
 }
 
 Strip->Print("AdcTOBCompare.eps");
 
 rfile->GetObject("DQMData/TOB/Nstp_rphi_layer1tob",nstptob[0]);
 rfile->GetObject("DQMData/TOB/Nstp_rphi_layer2tob",nstptob[1]);
 rfile->GetObject("DQMData/TOB/Nstp_rphi_layer3tob",nstptob[2]);
 rfile->GetObject("DQMData/TOB/Nstp_rphi_layer4tob",nstptob[3]);
 rfile->GetObject("DQMData/TOB/Nstp_rphi_layer5tob",nstptob[4]);
 rfile->GetObject("DQMData/TOB/Nstp_rphi_layer6tob",nstptob[5]);
 rfile->GetObject("DQMData/TOB/Nstp_sas_layer1tob",nstptob[6]);
 rfile->GetObject("DQMData/TOB/Nstp_sas_layer2tob",nstptob[7]);
 sfile->GetObject("DQMData/TOB/Nstp_rphi_layer1tob",newnstptob[0]);
 sfile->GetObject("DQMData/TOB/Nstp_rphi_layer2tob",newnstptob[1]);
 sfile->GetObject("DQMData/TOB/Nstp_rphi_layer3tob",newnstptob[2]);
 sfile->GetObject("DQMData/TOB/Nstp_rphi_layer4tob",newnstptob[3]);
 sfile->GetObject("DQMData/TOB/Nstp_rphi_layer5tob",newnstptob[4]);
 sfile->GetObject("DQMData/TOB/Nstp_rphi_layer6tob",newnstptob[5]);
 sfile->GetObject("DQMData/TOB/Nstp_sas_layer1tob",newnstptob[6]);
 sfile->GetObject("DQMData/TOB/Nstp_sas_layer2tob",newnstptob[7]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(3,3);
 for (Int_t i=0; i<8; i++) {
   Strip->cd(i+1);
   nstptob[i]->SetLineColor(2);
   newnstptob[i]->SetLineColor(4);
   newnstptob[i]->SetLineStyle(2);
   nstptob[i]->Draw();
   newnstptob[i]->Draw("sames");
   myPV->PVCompute(nstptob[i] , newnstptob[i] , te );
 }
 
 Strip->Print("NstpTOBCompare.eps");

 rfile->GetObject("DQMData/TOB/Posx_rphi_layer1tob",postob[0]);
 rfile->GetObject("DQMData/TOB/Posx_rphi_layer2tob",postob[1]);
 rfile->GetObject("DQMData/TOB/Posx_rphi_layer3tob",postob[2]);
 rfile->GetObject("DQMData/TOB/Posx_rphi_layer4tob",postob[3]);
 rfile->GetObject("DQMData/TOB/Posx_rphi_layer5tob",postob[4]);
 rfile->GetObject("DQMData/TOB/Posx_rphi_layer6tob",postob[5]);
 rfile->GetObject("DQMData/TOB/Posx_sas_layer1tob",postob[6]);
 rfile->GetObject("DQMData/TOB/Posx_sas_layer2tob",postob[7]);
 sfile->GetObject("DQMData/TOB/Posx_rphi_layer1tob",newpostob[0]);
 sfile->GetObject("DQMData/TOB/Posx_rphi_layer2tob",newpostob[1]);
 sfile->GetObject("DQMData/TOB/Posx_rphi_layer3tob",newpostob[2]);
 sfile->GetObject("DQMData/TOB/Posx_rphi_layer4tob",newpostob[3]);
 sfile->GetObject("DQMData/TOB/Posx_rphi_layer5tob",newpostob[4]);
 sfile->GetObject("DQMData/TOB/Posx_rphi_layer6tob",newpostob[5]);
 sfile->GetObject("DQMData/TOB/Posx_sas_layer1tob",newpostob[6]);
 sfile->GetObject("DQMData/TOB/Posx_sas_layer2tob",newpostob[7]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(3,3);
  for (Int_t i=0; i<8; i++) {
    Strip->cd(i+1);
    postob[i]->SetLineColor(2);
    newpostob[i]->SetLineColor(4);
    newpostob[i]->SetLineStyle(2);
    postob[i]->Draw();
    newpostob[i]->Draw("sames");
    myPV->PVCompute(postob[i] , newpostob[i] , te );
  }
  
  Strip->Print("PosTOBCompare.eps");
  

 rfile->GetObject("DQMData/TOB/Errx_rphi_layer1tob",errxtob[0]);
 rfile->GetObject("DQMData/TOB/Errx_rphi_layer2tob",errxtob[1]);
 rfile->GetObject("DQMData/TOB/Errx_rphi_layer3tob",errxtob[2]);
 rfile->GetObject("DQMData/TOB/Errx_rphi_layer4tob",errxtob[3]);
 rfile->GetObject("DQMData/TOB/Errx_rphi_layer5tob",errxtob[4]);
 rfile->GetObject("DQMData/TOB/Errx_rphi_layer6tob",errxtob[5]);
 rfile->GetObject("DQMData/TOB/Errx_sas_layer1tob",errxtob[6]);
 rfile->GetObject("DQMData/TOB/Errx_sas_layer2tob",errxtob[7]);
 sfile->GetObject("DQMData/TOB/Errx_rphi_layer1tob",newerrxtob[0]);
 sfile->GetObject("DQMData/TOB/Errx_rphi_layer2tob",newerrxtob[1]);
 sfile->GetObject("DQMData/TOB/Errx_rphi_layer3tob",newerrxtob[2]);
 sfile->GetObject("DQMData/TOB/Errx_rphi_layer4tob",newerrxtob[3]);
 sfile->GetObject("DQMData/TOB/Errx_rphi_layer5tob",newerrxtob[4]);
 sfile->GetObject("DQMData/TOB/Errx_rphi_layer6tob",newerrxtob[5]);
 sfile->GetObject("DQMData/TOB/Errx_sas_layer1tob",newerrxtob[6]);
 sfile->GetObject("DQMData/TOB/Errx_sas_layer2tob",newerrxtob[7]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<8; i++) {
    Strip->cd(i+1);
    errxtob[i]->SetLineColor(2);
    newerrxtob[i]->SetLineColor(4);
    newerrxtob[i]->SetLineStyle(2);
    errxtob[i]->Draw();
    newerrxtob[i]->Draw("sames");
    myPV->PVCompute(errxtob[i] , newerrxtob[i] , te );
  }
  
  Strip->Print("ErrxTOBCompare.eps");
  
 
 rfile->GetObject("DQMData/TOB/Res_rphi_layer1tob",restob[0]);
 rfile->GetObject("DQMData/TOB/Res_rphi_layer2tob",restob[1]);
 rfile->GetObject("DQMData/TOB/Res_rphi_layer3tob",restob[2]);
 rfile->GetObject("DQMData/TOB/Res_rphi_layer4tob",restob[3]);
 rfile->GetObject("DQMData/TOB/Res_rphi_layer5tob",restob[4]);
 rfile->GetObject("DQMData/TOB/Res_rphi_layer6tob",restob[5]);
 rfile->GetObject("DQMData/TOB/Res_sas_layer1tob",restob[6]);
 rfile->GetObject("DQMData/TOB/Res_sas_layer2tob",restob[7]);
 sfile->GetObject("DQMData/TOB/Res_rphi_layer1tob",newrestob[0]);
 sfile->GetObject("DQMData/TOB/Res_rphi_layer2tob",newrestob[1]);
 sfile->GetObject("DQMData/TOB/Res_rphi_layer3tob",newrestob[2]);
 sfile->GetObject("DQMData/TOB/Res_rphi_layer4tob",newrestob[3]);
 sfile->GetObject("DQMData/TOB/Res_rphi_layer5tob",newrestob[4]);
 sfile->GetObject("DQMData/TOB/Res_rphi_layer6tob",newrestob[5]);
 sfile->GetObject("DQMData/TOB/Res_sas_layer1tob",newrestob[6]);
 sfile->GetObject("DQMData/TOB/Res_sas_layer2tob",newrestob[7]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(3,3);
  for (Int_t i=0; i<8; i++) {
    Strip->cd(i+1);
    restob[i]->SetLineColor(2);
    newrestob[i]->SetLineColor(4);
    newrestob[i]->SetLineStyle(2);
    restob[i]->Draw();
    newrestob[i]->Draw("sames");
    myPV->PVCompute(restob[i] , newrestob[i] , te );
  }
  
  Strip->Print("ResTOBCompare.eps");


 rfile->GetObject("DQMData/TOB/Chi2_rphi_layer1tob",chi2tob[0]);
 rfile->GetObject("DQMData/TOB/Chi2_rphi_layer2tob",chi2tob[1]);
 rfile->GetObject("DQMData/TOB/Chi2_rphi_layer3tob",chi2tob[2]);
 rfile->GetObject("DQMData/TOB/Chi2_rphi_layer4tob",chi2tob[3]);
 rfile->GetObject("DQMData/TOB/Chi2_rphi_layer5tob",chi2tob[4]);
 rfile->GetObject("DQMData/TOB/Chi2_rphi_layer6tob",chi2tob[5]);
 rfile->GetObject("DQMData/TOB/Chi2_sas_layer1tob",chi2tob[6]);
 rfile->GetObject("DQMData/TOB/Chi2_sas_layer2tob",chi2tob[7]);
 sfile->GetObject("DQMData/TOB/Chi2_rphi_layer1tob",newchi2tob[0]);
 sfile->GetObject("DQMData/TOB/Chi2_rphi_layer2tob",newchi2tob[1]);
 sfile->GetObject("DQMData/TOB/Chi2_rphi_layer3tob",newchi2tob[2]);
 sfile->GetObject("DQMData/TOB/Chi2_rphi_layer4tob",newchi2tob[3]);
 sfile->GetObject("DQMData/TOB/Chi2_rphi_layer5tob",newchi2tob[4]);
 sfile->GetObject("DQMData/TOB/Chi2_rphi_layer6tob",newchi2tob[5]);
 sfile->GetObject("DQMData/TOB/Chi2_sas_layer1tob",newchi2tob[6]);
 sfile->GetObject("DQMData/TOB/Chi2_sas_layer2tob",newchi2tob[7]);

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

  
 rfile->GetObject("DQMData/TOB/Posx_matched_layer1tob",matchedtob[0]);
 rfile->GetObject("DQMData/TOB/Posy_matched_layer1tob",matchedtob[1]);
 rfile->GetObject("DQMData/TOB/Posx_matched_layer2tob",matchedtob[2]);
 rfile->GetObject("DQMData/TOB/Posy_matched_layer2tob",matchedtob[3]);
 rfile->GetObject("DQMData/TOB/Errx_matched_layer1tob",matchedtob[4]);
 rfile->GetObject("DQMData/TOB/Erry_matched_layer1tob",matchedtob[5]);
 rfile->GetObject("DQMData/TOB/Errx_matched_layer2tob",matchedtob[6]);
 rfile->GetObject("DQMData/TOB/Erry_matched_layer2tob",matchedtob[7]);
 rfile->GetObject("DQMData/TOB/Resx_matched_layer1tob",matchedtob[8]);
 rfile->GetObject("DQMData/TOB/Resy_matched_layer1tob",matchedtob[9]);
 rfile->GetObject("DQMData/TOB/Resx_matched_layer2tob",matchedtob[10]);
 rfile->GetObject("DQMData/TOB/Resy_matched_layer2tob",matchedtob[11]);
 sfile->GetObject("DQMData/TOB/Posx_matched_layer1tob",newmatchedtob[0]);
 sfile->GetObject("DQMData/TOB/Posy_matched_layer1tob",newmatchedtob[1]);
 sfile->GetObject("DQMData/TOB/Posx_matched_layer2tob",newmatchedtob[2]);
 sfile->GetObject("DQMData/TOB/Posy_matched_layer2tob",newmatchedtob[3]);
 sfile->GetObject("DQMData/TOB/Errx_matched_layer1tob",newmatchedtob[4]);
 sfile->GetObject("DQMData/TOB/Erry_matched_layer1tob",newmatchedtob[5]);
 sfile->GetObject("DQMData/TOB/Errx_matched_layer2tob",newmatchedtob[6]);
 sfile->GetObject("DQMData/TOB/Erry_matched_layer2tob",newmatchedtob[7]);
 sfile->GetObject("DQMData/TOB/Resx_matched_layer1tob",newmatchedtob[8]);
 sfile->GetObject("DQMData/TOB/Resy_matched_layer1tob",newmatchedtob[9]);
 sfile->GetObject("DQMData/TOB/Resx_matched_layer2tob",newmatchedtob[10]);
 sfile->GetObject("DQMData/TOB/Resy_matched_layer2tob",newmatchedtob[11]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<12; i++) {
   Strip->cd(i+1);
   matchedtob[i]->SetLineColor(2);
   newmatchedtob[i]->SetLineColor(4);
   newmatchedtob[i]->SetLineStyle(2);
   matchedtob[i]->Draw();
   newmatchedtob[i]->Draw("sames");
   myPV->PVCompute(matchedtob[i] , newmatchedtob[i] , te );
 }
 
 Strip->Print("MatchedTOBCompare.eps");
 

 //=============================================================== 
// TID
 
 TH1F* adctid[5];
 TH1F* nstptid[5];
 TH1F* postid[5];
 TH1F* errxtid[5];
 TH1F* restid[5];
 TH1F* chi2tid[5];
 TH1F* matchedtid[8];
 TH1F* matchedchi2tid[5];
 TH1F* newadctid[5];
 TH1F* newnstptid[5];
 TH1F* newpostid[5];
 TH1F* newerrxtid[5];
 TH1F* newrestid[5];
 TH1F* newchi2tid[5];
 TH1F* newmatchedtid[8];
 TH1F* newmatchedchi2tid[5];
 
 rfile->GetObject("DQMData/TID/Adc_rphi_layer1tid",adctid[0]);
 rfile->GetObject("DQMData/TID/Adc_rphi_layer2tid",adctid[1]);
 rfile->GetObject("DQMData/TID/Adc_rphi_layer3tid",adctid[2]);
 rfile->GetObject("DQMData/TID/Adc_sas_layer1tid",adctid[3]);
 rfile->GetObject("DQMData/TID/Adc_sas_layer2tid",adctid[4]);
 sfile->GetObject("DQMData/TID/Adc_rphi_layer1tid",newadctid[0]);
 sfile->GetObject("DQMData/TID/Adc_rphi_layer2tid",newadctid[1]);
 sfile->GetObject("DQMData/TID/Adc_rphi_layer3tid",newadctid[2]);
 sfile->GetObject("DQMData/TID/Adc_sas_layer1tid",newadctid[3]);
 sfile->GetObject("DQMData/TID/Adc_sas_layer2tid",newadctid[4]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   Strip->cd(i+1);
   adctid[i]->SetLineColor(2);
   newadctid[i]->SetLineColor(4);
    newadctid[i]->SetLineStyle(2);
    adctid[i]->Draw();
    newadctid[i]->Draw("sames");
    myPV->PVCompute(adctid[i] , newadctid[i] , te );
 }
 
 Strip->Print("AdcTIDCompare.eps");
 
 rfile->GetObject("DQMData/TID/Nstp_rphi_layer1tid",nstptid[0]);
 rfile->GetObject("DQMData/TID/Nstp_rphi_layer2tid",nstptid[1]);
 rfile->GetObject("DQMData/TID/Nstp_rphi_layer3tid",nstptid[2]);
 rfile->GetObject("DQMData/TID/Nstp_sas_layer1tid",nstptid[3]);
 rfile->GetObject("DQMData/TID/Nstp_sas_layer2tid",nstptid[4]);
 sfile->GetObject("DQMData/TID/Nstp_rphi_layer1tid",newnstptid[0]);
 sfile->GetObject("DQMData/TID/Nstp_rphi_layer2tid",newnstptid[1]);
 sfile->GetObject("DQMData/TID/Nstp_rphi_layer3tid",newnstptid[2]);
 sfile->GetObject("DQMData/TID/Nstp_sas_layer1tid",newnstptid[3]);
 sfile->GetObject("DQMData/TID/Nstp_sas_layer2tid",newnstptid[4]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<5; i++) {
   Strip->cd(i+1);
   nstptid[i]->SetLineColor(2);
   newnstptid[i]->SetLineColor(4);
   newnstptid[i]->SetLineStyle(2);
   nstptid[i]->Draw();
   newnstptid[i]->Draw("sames");
   myPV->PVCompute(nstptid[i] , newnstptid[i] , te );
 }
 
 Strip->Print("NstpTIDCompare.eps");

 rfile->GetObject("DQMData/TID/Posx_rphi_layer1tid",postid[0]);
 rfile->GetObject("DQMData/TID/Posx_rphi_layer2tid",postid[1]);
 rfile->GetObject("DQMData/TID/Posx_rphi_layer3tid",postid[2]);
 rfile->GetObject("DQMData/TID/Posx_sas_layer1tid",postid[3]);
 rfile->GetObject("DQMData/TID/Posx_sas_layer2tid",postid[4]);
 sfile->GetObject("DQMData/TID/Posx_rphi_layer1tid",newpostid[0]);
 sfile->GetObject("DQMData/TID/Posx_rphi_layer2tid",newpostid[1]);
 sfile->GetObject("DQMData/TID/Posx_rphi_layer3tid",newpostid[2]);
 sfile->GetObject("DQMData/TID/Posx_sas_layer1tid",newpostid[3]);
 sfile->GetObject("DQMData/TID/Posx_sas_layer2tid",newpostid[4]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<5; i++) {
    Strip->cd(i+1);
    postid[i]->SetLineColor(2);
    newpostid[i]->SetLineColor(4);
    newpostid[i]->SetLineStyle(2);
    postid[i]->Draw();
    newpostid[i]->Draw("sames");
    myPV->PVCompute(postid[i] , newpostid[i] , te );
  }
  
  Strip->Print("PosTIDCompare.eps");
  

 rfile->GetObject("DQMData/TID/Errx_rphi_layer1tid",errxtid[0]);
 rfile->GetObject("DQMData/TID/Errx_rphi_layer2tid",errxtid[1]);
 rfile->GetObject("DQMData/TID/Errx_rphi_layer3tid",errxtid[2]);
 rfile->GetObject("DQMData/TID/Errx_sas_layer1tid",errxtid[3]);
 rfile->GetObject("DQMData/TID/Errx_sas_layer2tid",errxtid[4]);
 sfile->GetObject("DQMData/TID/Errx_rphi_layer1tid",newerrxtid[0]);
 sfile->GetObject("DQMData/TID/Errx_rphi_layer2tid",newerrxtid[1]);
 sfile->GetObject("DQMData/TID/Errx_rphi_layer3tid",newerrxtid[2]);
 sfile->GetObject("DQMData/TID/Errx_sas_layer1tid",newerrxtid[3]);
 sfile->GetObject("DQMData/TID/Errx_sas_layer2tid",newerrxtid[4]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<5; i++) {
    Strip->cd(i+1);
    errxtid[i]->SetLineColor(2);
    newerrxtid[i]->SetLineColor(4);
    newerrxtid[i]->SetLineStyle(2);
    errxtid[i]->Draw();
    newerrxtid[i]->Draw("sames");
    myPV->PVCompute(errxtid[i] , newerrxtid[i] , te );
  }
  
  Strip->Print("ErrxTIDCompare.eps");
  
 
 rfile->GetObject("DQMData/TID/Res_rphi_layer1tid",restid[0]);
 rfile->GetObject("DQMData/TID/Res_rphi_layer2tid",restid[1]);
 rfile->GetObject("DQMData/TID/Res_rphi_layer3tid",restid[2]);
 rfile->GetObject("DQMData/TID/Res_sas_layer1tid",restid[3]);
 rfile->GetObject("DQMData/TID/Res_sas_layer2tid",restid[4]);
 sfile->GetObject("DQMData/TID/Res_rphi_layer1tid",newrestid[0]);
 sfile->GetObject("DQMData/TID/Res_rphi_layer2tid",newrestid[1]);
 sfile->GetObject("DQMData/TID/Res_rphi_layer3tid",newrestid[2]);
 sfile->GetObject("DQMData/TID/Res_sas_layer1tid",newrestid[3]);
 sfile->GetObject("DQMData/TID/Res_sas_layer2tid",newrestid[4]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<5; i++) {
    Strip->cd(i+1);
    restid[i]->SetLineColor(2);
    newrestid[i]->SetLineColor(4);
    newrestid[i]->SetLineStyle(2);
    restid[i]->Draw();
    newrestid[i]->Draw("sames");
    myPV->PVCompute(restid[i] , newrestid[i] , te );
  }
  
  Strip->Print("ResTIDCompare.eps");

 rfile->GetObject("DQMData/TID/Chi2_rphi_layer1tid",chi2tid[0]);
 rfile->GetObject("DQMData/TID/Chi2_rphi_layer2tid",chi2tid[1]);
 rfile->GetObject("DQMData/TID/Chi2_rphi_layer3tid",chi2tid[2]);
 rfile->GetObject("DQMData/TID/Chi2_sas_layer1tid",chi2tid[3]);
 rfile->GetObject("DQMData/TID/Chi2_sas_layer2tid",chi2tid[4]);
 sfile->GetObject("DQMData/TID/Chi2_rphi_layer1tid",newchi2tid[0]);
 sfile->GetObject("DQMData/TID/Chi2_rphi_layer2tid",newchi2tid[1]);
 sfile->GetObject("DQMData/TID/Chi2_rphi_layer3tid",newchi2tid[2]);
 sfile->GetObject("DQMData/TID/Chi2_sas_layer1tid",newchi2tid[3]);
 sfile->GetObject("DQMData/TID/Chi2_sas_layer2tid",newchi2tid[4]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<5; i++) {
    Strip->cd(i+1);
    chi2tid[i]->SetLineColor(2);
    newchi2tid[i]->SetLineColor(4);
    newchi2tid[i]->SetLineStyle(2);
    chi2tid[i]->Draw();
    newchi2tid[i]->Draw("sames");
    myPV->PVCompute(chi2tid[i] , newchi2tid[i] , te );
  }
  
  Strip->Print("Chi2TIDCompare.eps");
  
 rfile->GetObject("DQMData/TID/Posx_matched_layer1tid",matchedtid[0]);
 rfile->GetObject("DQMData/TID/Posy_matched_layer1tid",matchedtid[1]);
 rfile->GetObject("DQMData/TID/Posx_matched_layer2tid",matchedtid[2]);
 rfile->GetObject("DQMData/TID/Posy_matched_layer2tid",matchedtid[3]);
 rfile->GetObject("DQMData/TID/Errx_matched_layer1tid",matchedtid[4]);
 rfile->GetObject("DQMData/TID/Erry_matched_layer1tid",matchedtid[5]);
 rfile->GetObject("DQMData/TID/Errx_matched_layer2tid",matchedtid[6]);
 rfile->GetObject("DQMData/TID/Erry_matched_layer2tid",matchedtid[7]);
 sfile->GetObject("DQMData/TID/Posx_matched_layer1tid",newmatchedtid[0]);
 sfile->GetObject("DQMData/TID/Posy_matched_layer1tid",newmatchedtid[1]);
 sfile->GetObject("DQMData/TID/Posx_matched_layer2tid",newmatchedtid[2]);
 sfile->GetObject("DQMData/TID/Posy_matched_layer2tid",newmatchedtid[3]);
 sfile->GetObject("DQMData/TID/Errx_matched_layer1tid",newmatchedtid[4]);
 sfile->GetObject("DQMData/TID/Erry_matched_layer1tid",newmatchedtid[5]);
 sfile->GetObject("DQMData/TID/Errx_matched_layer2tid",newmatchedtid[6]);
 sfile->GetObject("DQMData/TID/Erry_matched_layer2tid",newmatchedtid[7]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,4);
 for (Int_t i=0; i<8; i++) {
   Strip->cd(i+1);
   matchedtid[i]->SetLineColor(2);
   newmatchedtid[i]->SetLineColor(4);
   newmatchedtid[i]->SetLineStyle(2);
   matchedtid[i]->Draw();
   newmatchedtid[i]->Draw("sames");
   myPV->PVCompute(matchedtid[i] , newmatchedtid[i] , te );
 }
 
 Strip->Print("MatchedTIDCompare.eps");

 rfile->GetObject("DQMData/TID/Resx_matched_layer1tid",matchedchi2tid[0]);
 rfile->GetObject("DQMData/TID/Resy_matched_layer1tid",matchedchi2tid[1]);
 rfile->GetObject("DQMData/TID/Resx_matched_layer2tid",matchedchi2tid[2]);
 rfile->GetObject("DQMData/TID/Resy_matched_layer2tid",matchedchi2tid[3]);
 rfile->GetObject("DQMData/TID/Chi2_matched_layer1tid",matchedchi2tid[4]);
 rfile->GetObject("DQMData/TID/Chi2_matched_layer2tid",matchedchi2tid[5]);
 sfile->GetObject("DQMData/TID/Resx_matched_layer1tid",newmatchedchi2tid[0]);
 sfile->GetObject("DQMData/TID/Resy_matched_layer1tid",newmatchedchi2tid[1]);
 sfile->GetObject("DQMData/TID/Resx_matched_layer2tid",newmatchedchi2tid[2]);
 sfile->GetObject("DQMData/TID/Resy_matched_layer2tid",newmatchedchi2tid[3]);
 sfile->GetObject("DQMData/TID/Chi2_matched_layer1tid",newmatchedchi2tid[4]);
 sfile->GetObject("DQMData/TID/Chi2_matched_layer2tid",newmatchedchi2tid[5]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<5; i++) {
    Strip->cd(i+1);
    chi2tid[i]->SetLineColor(2);
    newchi2tid[i]->SetLineColor(4);
    newchi2tid[i]->SetLineStyle(2);
    chi2tid[i]->Draw();
    newchi2tid[i]->Draw("sames");
    myPV->PVCompute(matchedchi2tid[i] , newmatchedchi2tid[i] , te );
  }
  
  Strip->Print("Chi2MatchedTIDCompare.eps");

 //======================================================================================================
// TEC
 
 TH1F* adctec[10];
 TH1F* nstptec[10];
 TH1F* postec[10];
 TH1F* errxtec[10];
 TH1F* restec[10];
 TH1F* chi2tec[10];
 TH1F* matchedtec[12];
 TH1F* matchedrestec[6];
 TH1F* matchedchi2tec[3];
 TH1F* newadctec[10];
 TH1F* newnstptec[10];
 TH1F* newpostec[10];
 TH1F* newerrxtec[10];
 TH1F* newrestec[10];
 TH1F* newmatchedtec[12];
 TH1F* newchi2tec[10];
 TH1F* newmatchedrestec[6];
 TH1F* newmatchedchi2tec[3];
 
 rfile->GetObject("DQMData/TEC/Adc_rphi_layer1tec",adctec[0]);
 rfile->GetObject("DQMData/TEC/Adc_rphi_layer2tec",adctec[1]);
 rfile->GetObject("DQMData/TEC/Adc_rphi_layer3tec",adctec[2]);
 rfile->GetObject("DQMData/TEC/Adc_rphi_layer4tec",adctec[3]);
 rfile->GetObject("DQMData/TEC/Adc_rphi_layer5tec",adctec[4]);
 rfile->GetObject("DQMData/TEC/Adc_rphi_layer6tec",adctec[5]);
 rfile->GetObject("DQMData/TEC/Adc_rphi_layer7tec",adctec[6]);
 rfile->GetObject("DQMData/TEC/Adc_sas_layer1tec",adctec[7]);
 rfile->GetObject("DQMData/TEC/Adc_sas_layer2tec",adctec[8]);
 rfile->GetObject("DQMData/TEC/Adc_sas_layer5tec",adctec[9]);
 sfile->GetObject("DQMData/TEC/Adc_rphi_layer1tec",newadctec[0]);
 sfile->GetObject("DQMData/TEC/Adc_rphi_layer2tec",newadctec[1]);
 sfile->GetObject("DQMData/TEC/Adc_rphi_layer3tec",newadctec[2]);
 sfile->GetObject("DQMData/TEC/Adc_rphi_layer4tec",newadctec[3]);
 sfile->GetObject("DQMData/TEC/Adc_rphi_layer5tec",newadctec[4]);
 sfile->GetObject("DQMData/TEC/Adc_rphi_layer6tec",newadctec[5]);
 sfile->GetObject("DQMData/TEC/Adc_rphi_layer7tec",newadctec[6]);
 sfile->GetObject("DQMData/TEC/Adc_sas_layer1tec",newadctec[7]);
 sfile->GetObject("DQMData/TEC/Adc_sas_layer2tec",newadctec[8]);
 sfile->GetObject("DQMData/TEC/Adc_sas_layer5tec",newadctec[9]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip->cd(i+1);
   adctec[i]->SetLineColor(2);
   newadctec[i]->SetLineColor(4);
    newadctec[i]->SetLineStyle(2);
    adctec[i]->Draw();
    newadctec[i]->Draw("sames");
    myPV->PVCompute(adctec[i] , newadctec[i] , te );
 }
 
 Strip->Print("AdcTECCompare.eps");
 
 rfile->GetObject("DQMData/TEC/Nstp_rphi_layer1tec",nstptec[0]);
 rfile->GetObject("DQMData/TEC/Nstp_rphi_layer2tec",nstptec[1]);
 rfile->GetObject("DQMData/TEC/Nstp_rphi_layer3tec",nstptec[2]);
 rfile->GetObject("DQMData/TEC/Nstp_rphi_layer4tec",nstptec[3]);
 rfile->GetObject("DQMData/TEC/Nstp_rphi_layer5tec",nstptec[4]);
 rfile->GetObject("DQMData/TEC/Nstp_rphi_layer6tec",nstptec[5]);
 rfile->GetObject("DQMData/TEC/Nstp_rphi_layer7tec",nstptec[6]);
 rfile->GetObject("DQMData/TEC/Nstp_sas_layer1tec",nstptec[7]);
 rfile->GetObject("DQMData/TEC/Nstp_sas_layer2tec",nstptec[8]);
 rfile->GetObject("DQMData/TEC/Nstp_sas_layer5tec",nstptec[9]);
 sfile->GetObject("DQMData/TEC/Nstp_rphi_layer1tec",newnstptec[0]);
 sfile->GetObject("DQMData/TEC/Nstp_rphi_layer2tec",newnstptec[1]);
 sfile->GetObject("DQMData/TEC/Nstp_rphi_layer3tec",newnstptec[2]);
 sfile->GetObject("DQMData/TEC/Nstp_rphi_layer4tec",newnstptec[3]);
 sfile->GetObject("DQMData/TEC/Nstp_rphi_layer5tec",newnstptec[4]);
 sfile->GetObject("DQMData/TEC/Nstp_rphi_layer6tec",newnstptec[5]);
 sfile->GetObject("DQMData/TEC/Nstp_rphi_layer7tec",newnstptec[6]);
 sfile->GetObject("DQMData/TEC/Nstp_sas_layer1tec",newnstptec[7]);
 sfile->GetObject("DQMData/TEC/Nstp_sas_layer2tec",newnstptec[8]);
 sfile->GetObject("DQMData/TEC/Nstp_sas_layer5tec",newnstptec[9]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<10; i++) {
   Strip->cd(i+1);
   nstptec[i]->SetLineColor(2);
   newnstptec[i]->SetLineColor(4);
   newnstptec[i]->SetLineStyle(2);
   nstptec[i]->Draw();
   newnstptec[i]->Draw("sames");
   myPV->PVCompute(nstptec[i] , newnstptec[i] , te );
 }
 
 Strip->Print("NstpTECCompare.eps");

 rfile->GetObject("DQMData/TEC/Posx_rphi_layer1tec",postec[0]);
 rfile->GetObject("DQMData/TEC/Posx_rphi_layer2tec",postec[1]);
 rfile->GetObject("DQMData/TEC/Posx_rphi_layer3tec",postec[2]);
 rfile->GetObject("DQMData/TEC/Posx_rphi_layer4tec",postec[3]);
 rfile->GetObject("DQMData/TEC/Posx_rphi_layer5tec",postec[4]);
 rfile->GetObject("DQMData/TEC/Posx_rphi_layer6tec",postec[5]);
 rfile->GetObject("DQMData/TEC/Posx_rphi_layer7tec",postec[6]);
 rfile->GetObject("DQMData/TEC/Posx_sas_layer1tec",postec[7]);
 rfile->GetObject("DQMData/TEC/Posx_sas_layer2tec",postec[8]);
 rfile->GetObject("DQMData/TEC/Posx_sas_layer5tec",postec[9]);
 sfile->GetObject("DQMData/TEC/Posx_rphi_layer1tec",newpostec[0]);
 sfile->GetObject("DQMData/TEC/Posx_rphi_layer2tec",newpostec[1]);
 sfile->GetObject("DQMData/TEC/Posx_rphi_layer3tec",newpostec[2]);
 sfile->GetObject("DQMData/TEC/Posx_rphi_layer4tec",newpostec[3]);
 sfile->GetObject("DQMData/TEC/Posx_rphi_layer5tec",newpostec[4]);
 sfile->GetObject("DQMData/TEC/Posx_rphi_layer6tec",newpostec[5]);
 sfile->GetObject("DQMData/TEC/Posx_rphi_layer7tec",newpostec[6]);
 sfile->GetObject("DQMData/TEC/Posx_sas_layer1tec",newpostec[7]);
 sfile->GetObject("DQMData/TEC/Posx_sas_layer2tec",newpostec[8]);
 sfile->GetObject("DQMData/TEC/Posx_sas_layer5tec",newpostec[9]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(4,3);
  for (Int_t i=0; i<10; i++) {
    Strip->cd(i+1);
    postec[i]->SetLineColor(2);
    newpostec[i]->SetLineColor(4);
    newpostec[i]->SetLineStyle(2);
    postec[i]->Draw();
    newpostec[i]->Draw("sames");
    myPV->PVCompute(postec[i] , newpostec[i] , te );
  }
  
  Strip->Print("PosTECCompare.eps");
  

 rfile->GetObject("DQMData/TEC/Errx_rphi_layer1tec",errxtec[0]);
 rfile->GetObject("DQMData/TEC/Errx_rphi_layer2tec",errxtec[1]);
 rfile->GetObject("DQMData/TEC/Errx_rphi_layer3tec",errxtec[2]);
 rfile->GetObject("DQMData/TEC/Errx_rphi_layer4tec",errxtec[3]);
 rfile->GetObject("DQMData/TEC/Errx_rphi_layer5tec",errxtec[4]);
 rfile->GetObject("DQMData/TEC/Errx_rphi_layer6tec",errxtec[5]);
 rfile->GetObject("DQMData/TEC/Errx_rphi_layer7tec",errxtec[6]);
 rfile->GetObject("DQMData/TEC/Errx_sas_layer1tec",errxtec[7]);
 rfile->GetObject("DQMData/TEC/Errx_sas_layer2tec",errxtec[8]);
 rfile->GetObject("DQMData/TEC/Errx_sas_layer5tec",errxtec[9]);
 sfile->GetObject("DQMData/TEC/Errx_rphi_layer1tec",newerrxtec[0]);
 sfile->GetObject("DQMData/TEC/Errx_rphi_layer2tec",newerrxtec[1]);
 sfile->GetObject("DQMData/TEC/Errx_rphi_layer3tec",newerrxtec[2]);
 sfile->GetObject("DQMData/TEC/Errx_rphi_layer4tec",newerrxtec[3]);
 sfile->GetObject("DQMData/TEC/Errx_rphi_layer5tec",newerrxtec[4]);
 sfile->GetObject("DQMData/TEC/Errx_rphi_layer6tec",newerrxtec[5]);
 sfile->GetObject("DQMData/TEC/Errx_rphi_layer7tec",newerrxtec[6]);
 sfile->GetObject("DQMData/TEC/Errx_sas_layer1tec",newerrxtec[7]);
 sfile->GetObject("DQMData/TEC/Errx_sas_layer2tec",newerrxtec[8]);
 sfile->GetObject("DQMData/TEC/Errx_sas_layer5tec",newerrxtec[9]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(4,3);
  for (Int_t i=0; i<10; i++) {
    Strip->cd(i+1);
    errxtec[i]->SetLineColor(2);
    newerrxtec[i]->SetLineColor(4);
    newerrxtec[i]->SetLineStyle(2);
    errxtec[i]->Draw();
    newerrxtec[i]->Draw("sames");
    myPV->PVCompute(errxtec[i] , newerrxtec[i] , te );
  }
  
  Strip->Print("ErrxTECCompare.eps");
  
 
 rfile->GetObject("DQMData/TEC/Res_rphi_layer1tec",restec[0]);
 rfile->GetObject("DQMData/TEC/Res_rphi_layer2tec",restec[1]);
 rfile->GetObject("DQMData/TEC/Res_rphi_layer3tec",restec[2]);
 rfile->GetObject("DQMData/TEC/Res_rphi_layer4tec",restec[3]);
 rfile->GetObject("DQMData/TEC/Res_rphi_layer5tec",restec[4]);
 rfile->GetObject("DQMData/TEC/Res_rphi_layer6tec",restec[5]);
 rfile->GetObject("DQMData/TEC/Res_rphi_layer7tec",restec[6]);
 rfile->GetObject("DQMData/TEC/Res_sas_layer1tec",restec[7]);
 rfile->GetObject("DQMData/TEC/Res_sas_layer2tec",restec[8]);
 rfile->GetObject("DQMData/TEC/Res_sas_layer5tec",restec[9]);
 sfile->GetObject("DQMData/TEC/Res_rphi_layer1tec",newrestec[0]);
 sfile->GetObject("DQMData/TEC/Res_rphi_layer2tec",newrestec[1]);
 sfile->GetObject("DQMData/TEC/Res_rphi_layer3tec",newrestec[2]);
 sfile->GetObject("DQMData/TEC/Res_rphi_layer4tec",newrestec[3]);
 sfile->GetObject("DQMData/TEC/Res_rphi_layer5tec",newrestec[4]);
 sfile->GetObject("DQMData/TEC/Res_rphi_layer6tec",newrestec[5]);
 sfile->GetObject("DQMData/TEC/Res_rphi_layer7tec",newrestec[6]);
 sfile->GetObject("DQMData/TEC/Res_sas_layer1tec",newrestec[7]);
 sfile->GetObject("DQMData/TEC/Res_sas_layer2tec",newrestec[8]);
 sfile->GetObject("DQMData/TEC/Res_sas_layer5tec",newrestec[9]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(4,3);
  for (Int_t i=0; i<10; i++) {
    Strip->cd(i+1);
    restec[i]->SetLineColor(2);
    newrestec[i]->SetLineColor(4);
    newrestec[i]->SetLineStyle(2);
    restec[i]->Draw();
    newrestec[i]->Draw("sames");
    myPV->PVCompute(restec[i] , newrestec[i] , te );
  }
  
  Strip->Print("ResTECCompare.eps");

 rfile->GetObject("DQMData/TEC/Chi2_rphi_layer1tec",chi2tec[0]);
 rfile->GetObject("DQMData/TEC/Chi2_rphi_layer2tec",chi2tec[1]);
 rfile->GetObject("DQMData/TEC/Chi2_rphi_layer3tec",chi2tec[2]);
 rfile->GetObject("DQMData/TEC/Chi2_rphi_layer4tec",chi2tec[3]);
 rfile->GetObject("DQMData/TEC/Chi2_rphi_layer5tec",chi2tec[4]);
 rfile->GetObject("DQMData/TEC/Chi2_rphi_layer6tec",chi2tec[5]);
 rfile->GetObject("DQMData/TEC/Chi2_rphi_layer7tec",chi2tec[6]);
 rfile->GetObject("DQMData/TEC/Chi2_sas_layer1tec",chi2tec[7]);
 rfile->GetObject("DQMData/TEC/Chi2_sas_layer2tec",chi2tec[8]);
 rfile->GetObject("DQMData/TEC/Chi2_sas_layer5tec",chi2tec[9]);
 sfile->GetObject("DQMData/TEC/Chi2_rphi_layer1tec",newchi2tec[0]);
 sfile->GetObject("DQMData/TEC/Chi2_rphi_layer2tec",newchi2tec[1]);
 sfile->GetObject("DQMData/TEC/Chi2_rphi_layer3tec",newchi2tec[2]);
 sfile->GetObject("DQMData/TEC/Chi2_rphi_layer4tec",newchi2tec[3]);
 sfile->GetObject("DQMData/TEC/Chi2_rphi_layer5tec",newchi2tec[4]);
 sfile->GetObject("DQMData/TEC/Chi2_rphi_layer6tec",newchi2tec[5]);
 sfile->GetObject("DQMData/TEC/Chi2_rphi_layer7tec",newchi2tec[6]);
 sfile->GetObject("DQMData/TEC/Chi2_sas_layer1tec",newchi2tec[7]);
 sfile->GetObject("DQMData/TEC/Chi2_sas_layer2tec",newchi2tec[8]);
 sfile->GetObject("DQMData/TEC/Chi2_sas_layer5tec",newchi2tec[9]);

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(4,3);
  for (Int_t i=0; i<10; i++) {
    Strip->cd(i+1);
    chi2tec[i]->SetLineColor(2);
    newchi2tec[i]->SetLineColor(4);
    newchi2tec[i]->SetLineStyle(2);
    chi2tec[i]->Draw();
    newchi2tec[i]->Draw("sames");
    myPV->PVCompute(chi2tec[i] , newchi2tec[i] , te );
  }
  
  Strip->Print("Chi2TECCompare.eps");
  
 rfile->GetObject("DQMData/TEC/Posx_matched_layer1tec",matchedtec[0]);
 rfile->GetObject("DQMData/TEC/Posy_matched_layer1tec",matchedtec[1]);
 rfile->GetObject("DQMData/TEC/Posx_matched_layer2tec",matchedtec[2]);
 rfile->GetObject("DQMData/TEC/Posy_matched_layer2tec",matchedtec[3]);
 rfile->GetObject("DQMData/TEC/Posx_matched_layer5tec",matchedtec[4]);
 rfile->GetObject("DQMData/TEC/Posy_matched_layer5tec",matchedtec[5]);
 rfile->GetObject("DQMData/TEC/Errx_matched_layer1tec",matchedtec[6]);
 rfile->GetObject("DQMData/TEC/Erry_matched_layer1tec",matchedtec[7]);
 rfile->GetObject("DQMData/TEC/Errx_matched_layer2tec",matchedtec[8]);
 rfile->GetObject("DQMData/TEC/Erry_matched_layer2tec",matchedtec[9]);
 rfile->GetObject("DQMData/TEC/Errx_matched_layer5tec",matchedtec[10]);
 rfile->GetObject("DQMData/TEC/Erry_matched_layer5tec",matchedtec[11]);
 sfile->GetObject("DQMData/TEC/Posx_matched_layer1tec",newmatchedtec[0]);
 sfile->GetObject("DQMData/TEC/Posy_matched_layer1tec",newmatchedtec[1]);
 sfile->GetObject("DQMData/TEC/Posx_matched_layer2tec",newmatchedtec[2]);
 sfile->GetObject("DQMData/TEC/Posy_matched_layer2tec",newmatchedtec[3]);
 sfile->GetObject("DQMData/TEC/Posx_matched_layer5tec",newmatchedtec[4]);
 sfile->GetObject("DQMData/TEC/Posy_matched_layer5tec",newmatchedtec[5]);
 sfile->GetObject("DQMData/TEC/Errx_matched_layer1tec",newmatchedtec[6]);
 sfile->GetObject("DQMData/TEC/Erry_matched_layer1tec",newmatchedtec[7]);
 sfile->GetObject("DQMData/TEC/Errx_matched_layer2tec",newmatchedtec[8]);
 sfile->GetObject("DQMData/TEC/Erry_matched_layer2tec",newmatchedtec[9]);
 sfile->GetObject("DQMData/TEC/Errx_matched_layer5tec",newmatchedtec[10]);
 sfile->GetObject("DQMData/TEC/Erry_matched_layer5tec",newmatchedtec[11]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,3);
 for (Int_t i=0; i<12; i++) {
   Strip->cd(i+1);
   matchedtec[i]->SetLineColor(2);
   newmatchedtec[i]->SetLineColor(4);
   newmatchedtec[i]->SetLineStyle(2);
   matchedtec[i]->Draw();
   newmatchedtec[i]->Draw("sames");
   myPV->PVCompute(matchedtec[i] , newmatchedtec[i] , te );
 }
 
 Strip->Print("MatchedTECCompare.eps");
 
 rfile->GetObject("DQMData/TEC/Resx_matched_layer1tec",matchedrestec[0]);
 rfile->GetObject("DQMData/TEC/Resy_matched_layer1tec",matchedrestec[1]);
 rfile->GetObject("DQMData/TEC/Resx_matched_layer2tec",matchedrestec[2]);
 rfile->GetObject("DQMData/TEC/Resy_matched_layer2tec",matchedrestec[3]);
 rfile->GetObject("DQMData/TEC/Resx_matched_layer5tec",matchedrestec[4]);
 rfile->GetObject("DQMData/TEC/Resy_matched_layer5tec",matchedrestec[5]);
 sfile->GetObject("DQMData/TEC/Resx_matched_layer1tec",newmatchedrestec[0]);
 sfile->GetObject("DQMData/TEC/Resy_matched_layer1tec",newmatchedrestec[1]);
 sfile->GetObject("DQMData/TEC/Resx_matched_layer2tec",newmatchedrestec[2]);
 sfile->GetObject("DQMData/TEC/Resy_matched_layer2tec",newmatchedrestec[3]);
 sfile->GetObject("DQMData/TEC/Resx_matched_layer5tec",newmatchedrestec[4]);
 sfile->GetObject("DQMData/TEC/Resy_matched_layer5tec",newmatchedrestec[5]);

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(2,3);
 for (Int_t i=0; i<6; i++) {
   Strip->cd(i+1);
   matchedrestec[i]->SetLineColor(2);
   newmatchedrestec[i]->SetLineColor(4);
   newmatchedrestec[i]->SetLineStyle(2);
   matchedrestec[i]->Draw();
   newmatchedrestec[i]->Draw("sames");
   myPV->PVCompute(matchedrestec[i] , newmatchedrestec[i] , te );
 }
 
 Strip->Print("MatchedResTECCompare.eps");

 rfile->GetObject("DQMData/TEC/Chi2_matched_layer1tec",matchedchi2tec[0]);
 rfile->GetObject("DQMData/TEC/Chi2_matched_layer2tec",matchedchi2tec[1]);
 rfile->GetObject("DQMData/TEC/Chi2_matched_layer5tec",matchedchi2tec[2]);
 sfile->GetObject("DQMData/TEC/Chi2_matched_layer1tec",newmatchedchi2tec[0]);
 sfile->GetObject("DQMData/TEC/Chi2_matched_layer2tec",newmatchedchi2tec[1]);
 sfile->GetObject("DQMData/TEC/Chi2_matched_layer5tec",newmatchedchi2tec[2]);
 
 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(1,3);
 for (Int_t i=0; i<3; i++) {
   Strip->cd(i+1);
   matchedchi2tec[i]->SetLineColor(2);
   newmatchedchi2tec[i]->SetLineColor(4);
   newmatchedchi2tec[i]->SetLineStyle(2);
   matchedchi2tec[i]->Draw();
   newmatchedchi2tec[i]->Draw("sames");
   myPV->PVCompute(matchedchi2tec[i] , newmatchedchi2tec[i] , te );
 }
 
 Strip->Print("MatchedChi2TECCompare.eps");
 
}


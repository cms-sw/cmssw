void SiStripRecHitsPlots()
{

 gROOT ->Reset();
 gStyle->SetOptStat(1111111);
 char*  rfilename = "sistriprechitshisto.root";

 delete gROOT->GetListOfFiles()->FindObject(rfilename);

 TText* te = new TText();
 TFile * rfile = new TFile(rfilename);
 Char_t histo[200];

 rfile->cd("DQMData/TrackerRecHits/Strip");
 gDirectory->ls();

 //reference files here
 // rfile->cd("")



////////////////////////////////////
//            TIB                 //
////////////////////////////////////

 rfile->cd("DQMData/TrackerRecHits/Strip/TIB");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(2,3);
 SiStrip->cd(1);
 Adc_rphi_layer1tib->Draw();
 SiStrip->cd(2);
 Adc_rphi_layer2tib->Draw();
 SiStrip->cd(3);
 Adc_rphi_layer3tib->Draw();
 SiStrip->cd(4);
 Adc_rphi_layer4tib->Draw();
 SiStrip->cd(5);
 Adc_sas_layer1tib->Draw();
 SiStrip->cd(6);
 Adc_sas_layer2tib->Draw();
 
 SiStrip->Print("AdcOfTIB.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(2,3);
 SiStrip->cd(1);
 Nstp_rphi_layer1tib->Draw();
 SiStrip->cd(2);
 Nstp_rphi_layer2tib->Draw();
 SiStrip->cd(3);
 Nstp_rphi_layer3tib->Draw();
 SiStrip->cd(4);
 Nstp_rphi_layer4tib->Draw();
 SiStrip->cd(5);
 Nstp_sas_layer1tib->Draw();
 SiStrip->cd(6);
 Nstp_sas_layer2tib->Draw();
 
 SiStrip->Print("NstpOfTIB.eps");
   
 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(2,3);
 SiStrip->cd(1);
 Posx_rphi_layer1tib->Draw();
 SiStrip->cd(2);
 Posx_rphi_layer2tib->Draw();
 SiStrip->cd(3);
 Posx_rphi_layer3tib->Draw();
 SiStrip->cd(4);
 Posx_rphi_layer4tib->Draw();
 SiStrip->cd(5);
 Posx_sas_layer1tib->Draw();
 SiStrip->cd(6);
 Posx_sas_layer2tib->Draw();
 
 SiStrip->Print("PosOfTIB.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");

 SiStrip->Divide(2,3);
 SiStrip->cd(1);
 Errx_rphi_layer1tib->Draw();
 SiStrip->cd(2);
 Errx_rphi_layer2tib->Draw();
 SiStrip->cd(3);
 Errx_rphi_layer3tib->Draw();
 SiStrip->cd(4);
 Errx_rphi_layer4tib->Draw();
 SiStrip->cd(5);
 Errx_sas_layer1tib->Draw();
 SiStrip->cd(6);
 Errx_sas_layer2tib->Draw();
 
 SiStrip->Print("ErrOfTIB.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");

 SiStrip->Divide(2,3);
 SiStrip->cd(1);
 Res_rphi_layer1tib->Draw();
 SiStrip->cd(2);
 Res_rphi_layer2tib->Draw();
 SiStrip->cd(3);
 Res_rphi_layer3tib->Draw();
 SiStrip->cd(4);
 Res_rphi_layer4tib->Draw();
 SiStrip->cd(5);
 Res_sas_layer1tib->Draw();
 SiStrip->cd(6);
 Res_sas_layer2tib->Draw();
 
 SiStrip->Print("ResOfTIB.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(2,3);
 SiStrip->cd(1);
 Pull_LF_rphi_layer1tib->Draw();
 SiStrip->cd(2);
 Pull_LF_rphi_layer2tib->Draw();
 SiStrip->cd(3);
 Pull_LF_rphi_layer3tib->Draw();
 SiStrip->cd(4);
 Pull_LF_rphi_layer4tib->Draw();
 SiStrip->cd(5);
 Pull_LF_sas_layer1tib->Draw();
 SiStrip->cd(6);
 Pull_LF_sas_layer2tib->Draw();

 SiStrip->Print("PullLFOfTIB.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(2,3);
 SiStrip->cd(1);
 Pull_MF_rphi_layer1tib->Draw();
 SiStrip->cd(2);
 Pull_MF_rphi_layer2tib->Draw();
 SiStrip->cd(3);
 Pull_MF_rphi_layer3tib->Draw();
 SiStrip->cd(4);
 Pull_MF_rphi_layer4tib->Draw();
 SiStrip->cd(5);
 Pull_MF_sas_layer1tib->Draw();
 SiStrip->cd(6);
 Pull_MF_sas_layer2tib->Draw();
 SiStrip->Print("PullMFOfTIB.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");

 SiStrip->Divide(2,3);
 SiStrip->cd(1);
 Chi2_rphi_layer1tib->Draw();
 SiStrip->cd(2);
 Chi2_rphi_layer2tib->Draw();
 SiStrip->cd(3);
 Chi2_rphi_layer3tib->Draw();
 SiStrip->cd(4);
 Chi2_rphi_layer4tib->Draw();
 SiStrip->cd(5);
 Chi2_sas_layer1tib->Draw();
 SiStrip->cd(6);
 Chi2_sas_layer2tib->Draw();
 
 SiStrip->Print("Chi2OfTIB.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,3);
 SiStrip->cd(1);
 Posx_matched_layer1tib->Draw();
 SiStrip->cd(2);
 Posy_matched_layer1tib->Draw();
 SiStrip->cd(3);
 Posx_matched_layer2tib->Draw();
 SiStrip->cd(4);
 Posy_matched_layer2tib->Draw();
 SiStrip->cd(5);
 Errx_matched_layer1tib->Draw();
 SiStrip->cd(6);
 Erry_matched_layer1tib->Draw();
 SiStrip->cd(7);
 Errx_matched_layer2tib->Draw();
 SiStrip->cd(8);
 Erry_matched_layer2tib->Draw();
   
 SiStrip->Print("MatchedOfTIB.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,2);

 SiStrip->cd(1);
 Resx_matched_layer1tib->Draw();
 SiStrip->cd(2);
 Resy_matched_layer1tib->Draw();
 SiStrip->cd(3);
 Chi2_matched_layer1tib->Draw();
 SiStrip->cd(4);
 Resx_matched_layer2tib->Draw();
 SiStrip->cd(5);
 Resy_matched_layer2tib->Draw();
 SiStrip->cd(6);
 Chi2_matched_layer2tib->Draw();

 SiStrip->Print("MatchedResOfTIB.eps");


////////////////////////////////////
//            TOB                 //
////////////////////////////////////


 rfile->cd("DQMData/TrackerRecHits/Strip/TOB");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,3);
 SiStrip->cd(1);
 Adc_rphi_layer1tob->Draw();
 SiStrip->cd(2);
 Adc_rphi_layer2tob->Draw();
 SiStrip->cd(3);
 Adc_rphi_layer3tob->Draw();
 SiStrip->cd(4);
 Adc_rphi_layer4tob->Draw();
 SiStrip->cd(5);
 Adc_rphi_layer5tob->Draw();
 SiStrip->cd(6);
 Adc_rphi_layer6tob->Draw();
 SiStrip->cd(7);
 Adc_sas_layer1tob->Draw();
 SiStrip->cd(8);
 Adc_sas_layer2tob->Draw();
 
 SiStrip->Print("AdcOfTOB.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,3);
 SiStrip->cd(1);
 Nstp_rphi_layer1tob->Draw();
 SiStrip->cd(2);
 Nstp_rphi_layer2tob->Draw();
 SiStrip->cd(3);
 Nstp_rphi_layer3tob->Draw();
 SiStrip->cd(4);
 Nstp_rphi_layer4tob->Draw();
 SiStrip->cd(5);
 Nstp_rphi_layer5tob->Draw();
 SiStrip->cd(6);
 Nstp_rphi_layer6tob->Draw();
 SiStrip->cd(7);
 Nstp_sas_layer1tob->Draw();
 SiStrip->cd(8);
 Nstp_sas_layer2tob->Draw();
 
 SiStrip->Print("NstpOfTOB.eps");
   
 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,3);
 SiStrip->cd(1);
 Posx_rphi_layer1tob->Draw();
 SiStrip->cd(2);
 Posx_rphi_layer2tob->Draw();
 SiStrip->cd(3);
 Posx_rphi_layer3tob->Draw();
 SiStrip->cd(4);
 Posx_rphi_layer4tob->Draw();
 SiStrip->cd(5);
 Posx_rphi_layer5tob->Draw();
 SiStrip->cd(6);
 Posx_rphi_layer6tob->Draw();
 SiStrip->cd(7);
 Posx_sas_layer1tob->Draw();
 SiStrip->cd(8);
 Posx_sas_layer2tob->Draw();
 
 SiStrip->Print("PosOfTOB.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");

 SiStrip->Divide(3,3);
 SiStrip->cd(1);
 Errx_rphi_layer1tob->Draw();
 SiStrip->cd(2);
 Errx_rphi_layer2tob->Draw();
 SiStrip->cd(3);
 Errx_rphi_layer3tob->Draw();
 SiStrip->cd(4);
 Errx_rphi_layer4tob->Draw();
 SiStrip->cd(5);
 Errx_rphi_layer5tob->Draw();
 SiStrip->cd(6);
 Errx_rphi_layer6tob->Draw();
 SiStrip->cd(7);
 Errx_sas_layer1tob->Draw();
 SiStrip->cd(8);
 Errx_sas_layer2tob->Draw();
 
 SiStrip->Print("ErrOfTOB.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");

 SiStrip->Divide(3,3);
 SiStrip->cd(1);
 Res_rphi_layer1tob->Draw();
 SiStrip->cd(2);
 Res_rphi_layer2tob->Draw();
 SiStrip->cd(3);
 Res_rphi_layer3tob->Draw();
 SiStrip->cd(4);
 Res_rphi_layer4tob->Draw();
 SiStrip->cd(5);
 Res_rphi_layer5tob->Draw();
 SiStrip->cd(6);
 Res_rphi_layer6tob->Draw();
 SiStrip->cd(7);
 Res_sas_layer1tob->Draw();
 SiStrip->cd(8);
 Res_sas_layer2tob->Draw();
 
 SiStrip->Print("ResOfTOB.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,3);
 SiStrip->cd(1);
 Pull_LF_rphi_layer1tob->Draw();
 SiStrip->cd(2);
 Pull_LF_rphi_layer2tob->Draw();
 SiStrip->cd(3);
 Pull_LF_rphi_layer3tob->Draw();
 SiStrip->cd(4);
 Pull_LF_rphi_layer4tob->Draw();
 SiStrip->cd(5);
 Pull_LF_rphi_layer5tob->Draw();
 SiStrip->cd(6);
 Pull_LF_rphi_layer6tob->Draw();
 SiStrip->cd(7);
 Pull_LF_sas_layer1tob->Draw();
 SiStrip->cd(8);
 Pull_LF_sas_layer2tob->Draw();
 SiStrip->Print("PullLFOfTOB.eps");

TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,3);
 SiStrip->cd(1);
 Pull_MF_rphi_layer1tob->Draw();
 SiStrip->cd(2);
 Pull_MF_rphi_layer2tob->Draw();
 SiStrip->cd(3);
 Pull_MF_rphi_layer3tob->Draw();
 SiStrip->cd(4);
 Pull_MF_rphi_layer4tob->Draw();
 SiStrip->cd(5);
 Pull_MF_rphi_layer5tob->Draw();
 SiStrip->cd(6);
 Pull_MF_rphi_layer6tob->Draw();
 SiStrip->cd(7);
 Pull_MF_sas_layer1tob->Draw();
 SiStrip->cd(8);
 Pull_MF_sas_layer2tob->Draw();
 SiStrip->Print("PullMFOfTOB.eps");
 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");

 SiStrip->Divide(3,3);
 SiStrip->cd(1);
 Chi2_rphi_layer1tob->Draw();
 SiStrip->cd(2);
 Chi2_rphi_layer2tob->Draw();
 SiStrip->cd(3);
 Chi2_rphi_layer3tob->Draw();
 SiStrip->cd(4);
 Chi2_rphi_layer4tob->Draw();
 SiStrip->cd(5);
 Chi2_rphi_layer5tob->Draw();
 SiStrip->cd(6);
 Chi2_rphi_layer6tob->Draw();
 SiStrip->cd(7);
 Chi2_sas_layer1tob->Draw();
 SiStrip->cd(8);
 Chi2_sas_layer2tob->Draw();
 
 SiStrip->Print("Chi2OfTOB.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,3);
 SiStrip->cd(1);
 Posx_matched_layer1tob->Draw();
 SiStrip->cd(2);
 Posy_matched_layer1tob->Draw();
 SiStrip->cd(3);
 Posx_matched_layer2tob->Draw();
 SiStrip->cd(4);
 Posy_matched_layer2tob->Draw();
 SiStrip->cd(5);
 Errx_matched_layer1tob->Draw();
 SiStrip->cd(6);
 Erry_matched_layer1tob->Draw();
 SiStrip->cd(7);
 Errx_matched_layer2tob->Draw();
 SiStrip->cd(8);
 Erry_matched_layer2tob->Draw();
   
 SiStrip->Print("MatchedOfTOB.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,2);
 SiStrip->cd(1);
 Resx_matched_layer1tob->Draw();
 SiStrip->cd(2);
 Resy_matched_layer1tob->Draw();
 SiStrip->cd(3);
 Chi2_matched_layer1tob->Draw();
 SiStrip->cd(4);
 Resx_matched_layer2tob->Draw();
 SiStrip->cd(5);
 Resy_matched_layer2tob->Draw();
 SiStrip->cd(6);
 Chi2_matched_layer2tob->Draw();
 
 SiStrip->Print("MatchedResOfTOB.eps");

////////////////////////////////////
//            TID                 //
////////////////////////////////////

 rfile->cd("DQMData/TrackerRecHits/Strip/TID");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,2);
 SiStrip->cd(1);
 Adc_rphi_layer1tid->Draw();
 SiStrip->cd(2);
 Adc_rphi_layer2tid->Draw();
 SiStrip->cd(3);
 Adc_rphi_layer3tid->Draw();
 SiStrip->cd(4);
 Adc_sas_layer1tid->Draw();
 SiStrip->cd(5);
 Adc_sas_layer2tid->Draw();
 
 SiStrip->Print("AdcOfTID.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,2);
 SiStrip->cd(1);
 Nstp_rphi_layer1tid->Draw();
 SiStrip->cd(2);
 Nstp_rphi_layer2tid->Draw();
 SiStrip->cd(3);
 Nstp_rphi_layer3tid->Draw();
 SiStrip->cd(4);
 Nstp_sas_layer1tid->Draw();
 SiStrip->cd(5);
 Nstp_sas_layer2tid->Draw();
 
 SiStrip->Print("NstpOfTID.eps");
   
 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,2);
 SiStrip->cd(1);
 Posx_rphi_layer1tid->Draw();
 SiStrip->cd(2);
 Posx_rphi_layer2tid->Draw();
 SiStrip->cd(3);
 Posx_rphi_layer3tid->Draw();
 SiStrip->cd(4);
 Posx_sas_layer1tid->Draw();
 SiStrip->cd(5);
 Posx_sas_layer2tid->Draw();
 
 SiStrip->Print("PosOfTID.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");

 SiStrip->Divide(3,2);
 SiStrip->cd(1);
 Errx_rphi_layer1tid->Draw();
 SiStrip->cd(2);
 Errx_rphi_layer2tid->Draw();
 SiStrip->cd(3);
 Errx_rphi_layer3tid->Draw();
 SiStrip->cd(4);
 Errx_sas_layer1tid->Draw();
 SiStrip->cd(5);
 Errx_sas_layer2tid->Draw();
 
 SiStrip->Print("ErrOfTID.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");

 SiStrip->Divide(3,2);
 SiStrip->cd(1);
 Res_rphi_layer1tid->Draw();
 SiStrip->cd(2);
 Res_rphi_layer2tid->Draw();
 SiStrip->cd(3);
 Res_rphi_layer3tid->Draw();
 SiStrip->cd(4);
 Res_sas_layer1tid->Draw();
 SiStrip->cd(5);
 Res_sas_layer2tid->Draw();
 
 SiStrip->Print("ResOfTID.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,2);
 SiStrip->cd(1);
 Pull_LF_rphi_layer1tid->Draw();
 SiStrip->cd(2);
 Pull_LF_rphi_layer2tid->Draw();
 SiStrip->cd(3);
 Pull_LF_rphi_layer3tid->Draw();
 SiStrip->cd(4);
 Pull_LF_sas_layer1tid->Draw();
 SiStrip->cd(5);
 Pull_LF_sas_layer2tid->Draw();
 SiStrip->Print("PullLFOfTID.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,2);
 SiStrip->cd(1);
 Pull_MF_rphi_layer1tid->Draw();
 SiStrip->cd(2);
 Pull_MF_rphi_layer2tid->Draw();
 SiStrip->cd(3);
 Pull_MF_rphi_layer3tid->Draw();
 SiStrip->cd(4);
 Pull_MF_sas_layer1tid->Draw();
 SiStrip->cd(5);
 Pull_MF_sas_layer2tid->Draw();
 SiStrip->Print("PullMFOfTID.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");

 SiStrip->Divide(3,2);
 SiStrip->cd(1);
 Chi2_rphi_layer1tid->Draw();
 SiStrip->cd(2);
 Chi2_rphi_layer2tid->Draw();
 SiStrip->cd(3);
 Chi2_rphi_layer3tid->Draw();
 SiStrip->cd(4);
 Chi2_sas_layer1tid->Draw();
 SiStrip->cd(5);
 Chi2_sas_layer2tid->Draw();
 
 SiStrip->Print("Chi2OfTID.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,3);
 SiStrip->cd(1);
 Posx_matched_layer1tid->Draw();
 SiStrip->cd(2);
 Posy_matched_layer1tid->Draw();
 SiStrip->cd(3);
 Posx_matched_layer2tid->Draw();
 SiStrip->cd(4);
 Posy_matched_layer2tid->Draw();
 SiStrip->cd(5);
 Errx_matched_layer1tid->Draw();
 SiStrip->cd(6);
 Erry_matched_layer1tid->Draw();
 SiStrip->cd(7);
 Errx_matched_layer2tid->Draw();
 SiStrip->cd(8);
 Erry_matched_layer2tid->Draw();
   
 SiStrip->Print("MatchedOfTID.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,2);
 SiStrip->cd(1);
 Resx_matched_layer1tid->Draw();
 SiStrip->cd(2);
 Resy_matched_layer1tid->Draw();
 SiStrip->cd(3);
 Chi2_matched_layer1tid->Draw();
 SiStrip->cd(4);
 Resx_matched_layer2tid->Draw();
 SiStrip->cd(5);
 Resy_matched_layer2tid->Draw();
 SiStrip->cd(6);
 Chi2_matched_layer2tid->Draw();
   
 SiStrip->Print("MatchedResOfTID.eps");

////////////////////////////////////
//            TEC                 //
////////////////////////////////////


 rfile->cd("DQMData/TrackerRecHits/Strip/TEC");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,4);
 SiStrip->cd(1);
 Adc_rphi_layer1tec->Draw();
 SiStrip->cd(2);
 Adc_rphi_layer2tec->Draw();
 SiStrip->cd(3);
 Adc_rphi_layer3tec->Draw();
 SiStrip->cd(4);
 Adc_rphi_layer4tec->Draw();
 SiStrip->cd(5);
 Adc_rphi_layer5tec->Draw();
 SiStrip->cd(6);
 Adc_rphi_layer6tec->Draw();
 SiStrip->cd(7);
 Adc_rphi_layer7tec->Draw();
 SiStrip->cd(8);
 Adc_sas_layer1tec->Draw();
 SiStrip->cd(9);
 Adc_sas_layer2tec->Draw();
 SiStrip->cd(10);
 Adc_sas_layer5tec->Draw();
 
 SiStrip->Print("AdcOfTEC.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,4);
 SiStrip->cd(1);
 Nstp_rphi_layer1tec->Draw();
 SiStrip->cd(2);
 Nstp_rphi_layer2tec->Draw();
 SiStrip->cd(3);
 Nstp_rphi_layer3tec->Draw();
 SiStrip->cd(4);
 Nstp_rphi_layer4tec->Draw();
 SiStrip->cd(5);
 Nstp_rphi_layer5tec->Draw();
 SiStrip->cd(6);
 Nstp_rphi_layer6tec->Draw();
 SiStrip->cd(7);
 Nstp_rphi_layer7tec->Draw();
 SiStrip->cd(8);
 Nstp_sas_layer1tec->Draw();
 SiStrip->cd(9);
 Nstp_sas_layer2tec->Draw();
 SiStrip->cd(10);
 Nstp_sas_layer5tec->Draw();
 
 SiStrip->Print("NstpOfTEC.eps");
   
 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 
 SiStrip->Divide(3,4);
 SiStrip->cd(1);
 Posx_rphi_layer1tec->Draw();
 SiStrip->cd(2);
 Posx_rphi_layer2tec->Draw();
 SiStrip->cd(3);
 Posx_rphi_layer3tec->Draw();
 SiStrip->cd(4);
 Posx_rphi_layer4tec->Draw();
 SiStrip->cd(5);
 Posx_rphi_layer5tec->Draw();
 SiStrip->cd(6);
 Posx_rphi_layer6tec->Draw();
 SiStrip->cd(7);
 Posx_rphi_layer7tec->Draw();
 SiStrip->cd(8);
 Posx_sas_layer1tec->Draw();
 SiStrip->cd(9);
 Posx_sas_layer2tec->Draw();
 SiStrip->cd(10);
 Posx_sas_layer5tec->Draw();
 SiStrip->Print("PosOfTEC.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");

 SiStrip->Divide(3,4);
 SiStrip->cd(1);
 Errx_rphi_layer1tec->Draw();
 SiStrip->cd(2);
 Errx_rphi_layer2tec->Draw();
 SiStrip->cd(3);
 Errx_rphi_layer3tec->Draw();
 SiStrip->cd(4);
 Errx_rphi_layer4tec->Draw();
 SiStrip->cd(5);
 Errx_rphi_layer5tec->Draw();
 SiStrip->cd(6);
 Errx_rphi_layer6tec->Draw();
 SiStrip->cd(7);
 Errx_rphi_layer7tec->Draw();
 SiStrip->cd(8);
 Errx_sas_layer1tec->Draw();
 SiStrip->cd(9);
 Errx_sas_layer2tec->Draw();
 SiStrip->cd(10);
 Errx_sas_layer5tec->Draw();
 SiStrip->Print("ErrOfTEC.eps");


 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,4);
 SiStrip->cd(1);
 Res_rphi_layer1tec->Draw();
 SiStrip->cd(2);
 Res_rphi_layer2tec->Draw();
 SiStrip->cd(3);
 Res_rphi_layer3tec->Draw();
 SiStrip->cd(4);
 Res_rphi_layer4tec->Draw();
 SiStrip->cd(5);
 Res_rphi_layer5tec->Draw();
 SiStrip->cd(6);
 Res_rphi_layer6tec->Draw();
 SiStrip->cd(7);
 Res_rphi_layer7tec->Draw();
 SiStrip->cd(8);
 Res_sas_layer1tec->Draw();
 SiStrip->cd(9);
 Res_sas_layer2tec->Draw();
 SiStrip->cd(10);
 Res_sas_layer5tec->Draw();
 
 SiStrip->Print("ResOfTEC.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,4);
 SiStrip->cd(1);
 Pull_LF_rphi_layer1tec->Draw();
 SiStrip->cd(2);
 Pull_LF_rphi_layer2tec->Draw();
 SiStrip->cd(3);
 Pull_LF_rphi_layer3tec->Draw();
 SiStrip->cd(4);
 Pull_LF_rphi_layer4tec->Draw();
 SiStrip->cd(5);
 Pull_LF_rphi_layer5tec->Draw();
 SiStrip->cd(6);
 Pull_LF_rphi_layer6tec->Draw();
 SiStrip->cd(7);
 Pull_LF_rphi_layer7tec->Draw();
 SiStrip->cd(8);
 Pull_LF_sas_layer1tec->Draw();
 SiStrip->cd(9);
 Pull_LF_sas_layer2tec->Draw();
 SiStrip->cd(10);
 Pull_LF_sas_layer5tec->Draw();
 SiStrip->Print("PullLFOfTEC.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,4);
 SiStrip->cd(1);
 Pull_MF_rphi_layer1tec->Draw();
 SiStrip->cd(2);
 Pull_MF_rphi_layer2tec->Draw();
 SiStrip->cd(3);
 Pull_MF_rphi_layer3tec->Draw();
 SiStrip->cd(4);
 Pull_MF_rphi_layer4tec->Draw();
 SiStrip->cd(5);
 Pull_MF_rphi_layer5tec->Draw();
 SiStrip->cd(6);
 Pull_MF_rphi_layer6tec->Draw();
 SiStrip->cd(7);
 Pull_MF_rphi_layer7tec->Draw();
 SiStrip->cd(8);
 Pull_MF_sas_layer1tec->Draw();
 SiStrip->cd(9);
 Pull_MF_sas_layer2tec->Draw();
 SiStrip->cd(10);
 Pull_MF_sas_layer5tec->Draw();
 SiStrip->Print("PullMFOfTEC.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,4);
 SiStrip->cd(1);
 Chi2_rphi_layer1tec->Draw();
 SiStrip->cd(2);
 Chi2_rphi_layer2tec->Draw();
 SiStrip->cd(3);
 Chi2_rphi_layer3tec->Draw();
 SiStrip->cd(4);
 Chi2_rphi_layer4tec->Draw();
 SiStrip->cd(5);
 Chi2_rphi_layer5tec->Draw();
 SiStrip->cd(6);
 Chi2_rphi_layer6tec->Draw();
 SiStrip->cd(7);
 Chi2_rphi_layer7tec->Draw();
 SiStrip->cd(8);
 Chi2_sas_layer1tec->Draw();
 SiStrip->cd(9);
 Chi2_sas_layer2tec->Draw();
 SiStrip->cd(10);
 Chi2_sas_layer5tec->Draw();
 
 SiStrip->Print("Chi2OfTEC.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(4,3);
 SiStrip->cd(1);
 Posx_matched_layer1tec->Draw();
 SiStrip->cd(2);
 Posy_matched_layer1tec->Draw();
 SiStrip->cd(3);
 Posx_matched_layer2tec->Draw();
 SiStrip->cd(4);
 Posy_matched_layer2tec->Draw();
 SiStrip->cd(5);
 Posx_matched_layer5tec->Draw();
 SiStrip->cd(6);
 Posy_matched_layer5tec->Draw();
 SiStrip->cd(7);
 Errx_matched_layer1tec->Draw();
 SiStrip->cd(8);
 Erry_matched_layer1tec->Draw();
 SiStrip->cd(9);
 Errx_matched_layer2tec->Draw();
 SiStrip->cd(10);
 Erry_matched_layer2tec->Draw();
 SiStrip->cd(11);
 Errx_matched_layer5tec->Draw();
 SiStrip->cd(12);
 Erry_matched_layer5tec->Draw();

 SiStrip->Print("MatchedOfTEC.eps");

 TCanvas * SiStrip = new TCanvas("SiStrip","SiStrip");
 SiStrip->Divide(3,3);
 SiStrip->cd(1);
 Resx_matched_layer1tec->Draw();
 SiStrip->cd(2);
 Resy_matched_layer1tec->Draw();
 SiStrip->cd(3);
 Chi2_matched_layer1tec->Draw();
 SiStrip->cd(4);
 Resx_matched_layer2tec->Draw();
 SiStrip->cd(5);
 Resy_matched_layer2tec->Draw();
 SiStrip->cd(6);
 Chi2_matched_layer2tec->Draw();
 SiStrip->cd(7);
 Resx_matched_layer5tec->Draw();
 SiStrip->cd(8);
 Resy_matched_layer5tec->Draw();
 SiStrip->cd(9);
 Chi2_matched_layer5tec->Draw();
   
 SiStrip->Print("MatchedResOfTEC.eps");
}


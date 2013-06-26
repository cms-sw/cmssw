void DoCompare( char* Energy ){

 const int NHisto = 47;
 const int NHisto2 = 4;
 const int NHisto3 = 2;

 TText* te = new TText();
 te->SetTextSize(0.1);
 
 gROOT->ProcessLine(".x HistoCompare.C");
 HistoCompare * myPV = new HistoCompare();

 char*  reffilename  = "${REFFILE}";//"../data/EcalSimHitHisto_30GeV.root";
 char*  curfilename  = "${CURFILE}";//"../data/EcalSimHitHisto_30GeV.root";

 TFile * reffile = new TFile(reffilename);
 TFile * curfile = new TFile(curfilename);

 //1-Dimension Histogram
 char* label[NHisto];
 label[0] = "Barrel_E1x1";
 label[1] = "Barrel_E2x2";
 label[2] = "Barrel_E3x3";
 label[3] = "Barrel_E4x4";
 label[4] = "Barrel_E5x5";
 label[5] = "Barrel_E1OverE4";
 label[6] = "Barrel_E4OverE9";
 label[7] = "Barrel_E9OverE16";
 label[8] = "Barrel_E16OverE25";
 label[9] = "Barrel_E1OverE25";
 label[10] = "Barrel_E9OverE25";
 label[11] = "Ecal_EBOverETotal";
 label[12] = "Ecal_EEOverETotal";
 label[13] = "Ecal_EPOverETotal";
 label[14] = "Ecal_EBEEEPOverETotal";
 label[15] = "PreShower_EHit_L1zp";
 label[16] = "PreShower_EHit_L2zp";
 label[17] = "Preshower_NHit_L1zp";
 label[18] = "Preshower_NHit_L2zp";
 label[19] = "Endcap_E1x1";
 label[20] = "Endcap_E2x2";
 label[21] = "Endcap_E3x3";
 label[22] = "Endcap_E4x4";
 label[23] = "Endcap_E5x5";
 label[24] = "Endcap_E1OverE4";
 label[25] = "Endcap_E4OverE9";
 label[26] = "Endcap_E9OverE16";
 label[27] = "Endcap_E16OverE25";
 label[28] = "Endcap_E1OverE25";
 label[29] = "Endcap_E9OverE25";
 label[30] = "PreShower_EHit_L1zm";
 label[31] = "PreShower_EHit_L2zm";
 label[32] = "Preshower_NHit_L1zm";
 label[33] = "Preshower_NHit_L2zm";
 label[34] = "Preshower_E1alphaE2_zm";
 label[35] = "Preshower_E1alphaE2_zp";
 label[36] = "Preshower_E2OverE1_zm";
 label[37] = "Preshower_E2OverE1_zp";
 label[38] = "Barrel_HitMultiplicity";
 label[39] = "Barrel_HitEnergy";
 label[40] = "Barrel_CryMultiplicity";
 label[41] = "EndcapZPlus_HitMultiplicity";
 label[42] = "EndcapZPlus_HitEnergy";
 label[43] = "EndcapZPlus_CryMultiplicity";
 label[44] = "EndcapZMinus_HitMultiplicity";
 label[45] = "EndcapZMinus_HitEnergy";
 label[46] = "EndcapZMinus_CryMultiplicity";


 TH1F* htemp1[NHisto];
 TH1F* htemp2[NHisto];
 for ( int i = 0; i< NHisto ; i++ ) {
   char title[50];
   TCanvas c1;
   htemp1[i]  = dynamic_cast<TH1F*>(reffile->Get(label[i]));
   htemp2[i]  = dynamic_cast<TH1F*>(curfile->Get(label[i]));
   if( htemp1[i] == 0 || htemp2[i] == 0) continue;
   htemp1[i]->SetLineColor(2);
   htemp2[i]->SetLineColor(4);
   htemp1[i]->SetLineStyle(3);
   htemp2[i]->SetLineStyle(5);
   TLegend leg(0.1, 0.15, 0.2, 0.25);
   leg.AddEntry(htemp1[i], "Reference", "l");
   leg.AddEntry(htemp2[i], "New ", "l");
   if (i>14 && i<19 || i>29 && i< 34 || i == 13 || i>37 ) c1.SetLogy();

   htemp1[i]->Draw();
   htemp2[i]->Draw("Same"); 
   leg.Draw();
   myPV->PVCompute(htemp1[i],htemp2[i], te);
   sprintf(title,"%s%s%s", Energy, label[i],".eps");
   c1.Print(title);
 }

 
 //2-Dimention Histograms
 char* label2[NHisto2];
 label2[0] = "Barrel_Longitudinal";
 label2[1] = "Barrel_Occupancy";
 label2[2] = "Endcap_Longitudinal";
 label2[3] = "Endcap_Occupancy";

 TH2F* h2temp1[NHisto2];
 TH2F* h2temp2[NHisto2];

 for ( int i = 0; i< NHisto2 ; i++ ) {
   char title[50];
   TCanvas c1;

   h2temp1[i]  = dynamic_cast<TH2F*>(reffile->Get(label2[i]));
   h2temp2[i]  = dynamic_cast<TH2F*>(curfile->Get(label2[i]));
   if( h2temp1[i] == 0 || h2temp2[i] == 0) continue; 
   if ( i==1 || i==3 ) {
      c1.Divide(2,1);
      c1.cd(1);
      h2temp1[i]->SetMarkerColor(2);
      h2temp1[i]->SetMarkerStyle(7);
      h2temp1[i]->Draw("COLZ");
   
      c1.cd(2);
      h2temp2[i]->SetMarkerColor(4);
      h2temp2[i]->SetMarkerStyle(7);
      h2temp2[i]->Draw("COLZ");
      myPV->PVCompute(h2temp1[i],h2temp2[i], te);
      sprintf(title,"%s%s%s", Energy, label2[i],".eps");
      c1.Print(title);
    }else {

      h2temp1[i]->SetMarkerColor(2);
      h2temp1[i]->SetMarkerStyle(22);
      h2temp1[i]->Draw();
      h2temp2[i]->SetMarkerColor(4);
      h2temp2[i]->SetMarkerStyle(23);
      h2temp2[i]->Draw("same");
      TLegend leg(0.65, 0.75, 0.75, 0.85);
      leg.AddEntry(h2temp1[i], "Reference", "p");
      leg.AddEntry(h2temp2[i], "New ", "p");
      leg.Draw();
      myPV->PVCompute(h2temp1[i],h2temp2[i], te);
      sprintf(title,"%s%s%s", Energy, label2[i],".eps");
      c1.Print(title);



   }

 }




 //TProfiles
  char* label3[NHisto3];
  label3[0] = "Preshower_EEOverES_zp";
  label3[1] = "Preshower_EEOverES_zm";

 TProfile* hpro1[NHisto3];
 TProfile* hpro2[NHisto3];

 for ( int i = 0; i< NHisto3 ; i++ ) {
   char title[50];
   TCanvas c1;
   c1.Divide(2,1);

   hpro1[i]  = dynamic_cast<TProfile*>(reffile->Get(label3[i]));
   hpro2[i]  = dynamic_cast<TProfile*>(curfile->Get(label3[i]));
   if (hpro1[i] == 0 || hpro2[i] == 0) continue ;
   TF1 *f1 = new TF1("f1","pol1");
   c1.cd(1);
   hpro1[i]->Fit(f1,"Q","",0.0, 200);
   hpro1[i]->SetLineColor(2);
   hpro1[i]->Draw();
   double gradient_ref = f1->GetParameter(1);
   std::strstream buf_ref;
   std::string value_ref;
   buf_ref<<"Gradient="<<gradient_ref<<std::endl;
   buf_ref>>value_ref;
   te->DrawTextNDC(0.1,0.2, value_ref.c_str());  
   c1.cd(2);
   hpro2[i]->Fit(f1,"Q","", 0.0, 200);
   hpro2[i]->SetLineColor(4);
   hpro2[i]->Draw();
   double gradient_cur = f1->GetParameter(1);
   std::strstream buf_cur;
   std::string value_cur;
   buf_cur<<"Gradient="<<gradient_cur<<std::endl;
   buf_cur>>value_cur;
   te->DrawTextNDC(0.1,0.2, value_cur.c_str());
   //myPV->PVCompute(hpro1[i],hpro2[i], te);
   sprintf(title,"%s%s%s", Energy, label3[i],".eps");
   c1.Print(title);
 }

}



#include "TText.h"
#include "TFile.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TPaveStats.h"
#include "TPaveLabel"

void DoCompare( ){

 static const int NHisto = 81;

 TText* te = new TText();
 te->SetTextSize(0.1);
 TPaveStats* st_1, st_2; 
 
  gROOT->ProcessLine(".x HistoCompare.C");
  HistoCompare * myPV = new HistoCompare();

 char*  reffilename  = "${REFFILE}";  //"./DTDigiPlots_ref.root";
 char*  curfilename  = "${CURFILE}";  //"./DTDigiPlots.root";

 TFile * reffile = new TFile(reffilename);
 TFile * curfile = new TFile(curfilename);

 reffile->cd("DQMData/DTDigiValidationTask");
 gDirectory->ls();

 curfile->cd("DQMData/DTDigiValidationTask");
 gDirectory->ls();

 //1-Dimension Histogram
 char* label[NHisto];
 char* label_dir[NHisto];

 label[0] = "DigiTimeBox";            
 label[1] = "DigiTimeBox_wheel2m";   
 label[2] = "DigiTimeBox_wheel1m";
 label[3] = "DigiTimeBox_wheel0";
 label[4] = "DigiTimeBox_wheel1p";
 label[5] = "DigiTimeBox_wheel2p";  
 label[6] = "DigiEfficiencyMu";
 label[7] = "DigiEfficiency";
 label[8] = "Number_Digi_per_layer";
 label[9] = "Number_simhit_vs_digi";
 label[10] = "Wire_Number_with_double_Digi";
 label[11] = "Simhit_occupancy_MB1";
 label[12] = "Digi_occupancy_MB1";             
 label[13] = "Simhit_occupancy_MB2";
 label[14] = "Digi_occupancy_MB2";
 label[15] = "Simhit_occupancy_MB3";
 label[16] = "Digi_occupancy_MB3";
 label[17] = "Simhit_occupancy_MB4";
 label[18] = "Digi_occupancy_MB4"; 
 
 char stringcham[40]; 
 char stringall[80];
 
 TH1F* htemp1[NHisto];
 TH1F* htemp2[NHisto];
 for ( int i = 0; i< NHisto ; i++ ) {
   char title[50];
   TCanvas c1;
   if ( i>18 )
    {
      sprintf(stringcham, "DigiTimeBox_slid_%d", i-19) ;
      label[i] = stringcham;
    }
      sprintf(stringall, "DQMData/DTDigiValidationTask/%s",label[i]);
      label_dir[i] = stringall;
   
   htemp1[i]  = dynamic_cast<TH1F*>(reffile->Get(label_dir[i]));
   htemp2[i]  = dynamic_cast<TH1F*>(curfile->Get(label_dir[i]));
   if( htemp1[i] == 0 ) std::cout << " reference histo is empty " << endl;
   if( htemp2[i] == 0 ) std::cout << " current version histo is empty " << endl;
   if( htemp1[i] == 0 || htemp2[i] == 0) continue;
   
   htemp1[i]->SetLineColor(2);
   htemp2[i]->SetLineColor(4);
   htemp1[i]->SetLineStyle(3);
   htemp2[i]->SetLineStyle(5);
   TLegend leg(0.1, 0.15, 0.2, 0.25);
   leg.AddEntry(htemp1[i], "Reference", "l");
   leg.AddEntry(htemp2[i], "New ", "l");

   htemp2[i]->Draw();
   gStyle->SetOptStat(1111);
   st_2 = (TPaveStats*)htemp2[i]->GetListOfFunctions()->FindObject("stats");

    htemp1[i]->Draw();
    gStyle->SetOptStat(1111);
    st_1 = (TPaveStats*)htemp1[i]->GetListOfFunctions()->FindObject("stats");

    TPaveStats* sta_1= (TPaveStats*)st_1->Clone();

    sta_1->SetTextColor(2);
    sta_1->SetX1NDC(.80);
    sta_1->SetX2NDC(0.95);
    sta_1->SetY1NDC(0.70);
    sta_1->SetY2NDC(0.85);

    TPaveStats* sta_2= (TPaveStats*)st_2->Clone();
    
    sta_2->SetTextColor(4);
    sta_2->SetX1NDC(.80);
    sta_2->SetX2NDC(0.95);
    sta_2->SetY1NDC(0.85);
    sta_2->SetY2NDC(1.0);

    gStyle->SetOptStat(000000);
    htemp2[i]->Draw();
    gStyle->SetOptStat(000000);
    htemp1[i]->Draw("Same"); 
    sta_2->Draw("Same");
    sta_1->Draw("Same"); 
    leg.Draw();
    myPV->PVCompute(htemp1[i],htemp2[i], te);
    sprintf(title,"%s%s", label[i],".eps");
    c1.Print(title);
 }


}



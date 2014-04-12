#include "TText.h"
#include "TFile.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TLegend.h"

void DoCompare_DT( ){

 static const int NHisto = 23;

 TText* te = new TText();
 te->SetTextSize(0.1);

 TPaveStats* st_1;
 TPaveStats* st_2;
 
  gROOT->ProcessLine(".x HistoCompare.C");
  HistoCompare * myPV = new HistoCompare();

 char*  reffilename  = "${REFFILE}";//"./DTSimHitsPlots_ref.root";
 char*  curfilename  = "${CURFILE}";//"./DTSimHitsPlots.root";

 TFile * reffile = new TFile(reffilename);
 TFile * curfile = new TFile(curfilename);

 curfile->cd("DQMData/MuonDTHitsV/DTHitsValidationTask");
 gDirectory->ls();


 //1-Dimension Histogram
 char* label[NHisto];
 char* label_dir[NHisto];

 label[0] = "Number_of_all_DT_hits";
 label[1] = "Number_of_muon_DT_hits";
 label[2] = "Wheel_occupancy";
 label[3] = "Station_occupancy";
 label[4] = "Sector_occupancy";
 label[5] = "SuperLayer_occupancy";
 label[6] = "Layer_occupancy";
 label[7] = "Wire_occupancy";
 label[8] = "DT_energy_loss_keV";
 label[9] = "chamber_occupancy";
 label[10] = "Momentum_at_MB1";
 label[11] = "Momentum_at_MB4";
 label[12] = "Loss_of_muon_Momentum_in_Iron";
 label[13] = "path_followed_by_muon";
 label[14] = "Tof_of_hits";
 label[15] = "radius_of_hit";
 label[16] = "costheta_of_hit";
 label[17] = "global_eta_of_hit";
 label[18] = "global_phi_of_hit";
 label[19] = "Local_x-coord_vs_local_z-coord_of_muon_hit";
 label[20] = "local_x-coord_vs_local_y-coord_of_muon_hit";
 label[21] = "Global_x-coord_vs_global_z-coord_of_muon_hit";
 label[22] = "Global_x-coord_vs_global_y-coord_of_muon_hit";

 char stringall[90];

 TH1F* htemp1[NHisto];
 TH1F* htemp2[NHisto];

 for ( int i = 0; i< NHisto ; i++ ) {
   char title[50];
   TCanvas c1;
//   cout << "label(i)" << label[i] << endl;

 
   sprintf(stringall, "DQMData/MuonDTHitsV/DTHitsValidationTask/%s",label[i]);
   label_dir[i] = stringall;

   if ( i<19 ) 
   {
     htemp1[i]  = dynamic_cast<TH1F*>(reffile->Get(label[i]));
     htemp2[i]  = dynamic_cast<TH1F*>(curfile->Get(label_dir[i]));
     if( htemp1[i] == 0 ) std::cout << " reference histo is empty " << endl;
     if( htemp2[i] == 0 ) std::cout << " current histo is empty " << endl;

     htemp1[i]->SetLineColor(2);
     htemp2[i]->SetLineColor(4);
     htemp1[i]->SetLineStyle(1);
     htemp2[i]->SetLineStyle(2);
     htemp1[i]->SetLineWidth(2);
     htemp2[i]->SetLineWidth(2);   

     TLegend leg(0.1, 0.15, 0.2, 0.25);
     leg.AddEntry(htemp1[i], "Reference", "l");
     leg.AddEntry(htemp2[i], "New ", "l");
//   if (i>14 && i<19 || i>29 && i< 34 || i == 13 || i>37 ) c1.SetLogy();

     htemp1[i]->Draw();
     gStyle->SetOptStat(1111);  
     st_1 = (TPaveStats*)htemp1[i]->GetListOfFunctions()->FindObject("stats");

     htemp2[i]->Draw();
     gStyle->SetOptStat(1111);
     st_2 = (TPaveStats*)htemp2[i]->GetListOfFunctions()->FindObject("stats");

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
   
   } else {

     htemp1[i]  = dynamic_cast<TH2F*>(reffile->Get(label[i]));
     htemp2[i]  = dynamic_cast<TH2F*>(curfile->Get(label_dir[i]));
 
     htemp1[i]->SetMarkerStyle(21);
     htemp2[i]->SetMarkerStyle(22);
     htemp1[i]->SetMarkerColor(2);
     htemp2[i]->SetMarkerColor(4);
     htemp1[i]->SetMarkerSize(0.3);
     htemp2[i]->SetMarkerSize(0.3);
 
     c1.Divide(1,2);
     c1.cd(1);
     htemp1[i]->Draw();
     leg.Draw();
     
     c1.cd(2);
     htemp2[i]->Draw();
     leg.Draw();

     sprintf(title,"%s%s", label[i],".eps");  
     c1.Print(title);
  } 
 }


}



#include "TText.h"
#include "TFile.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TLegend.h"

void DoCompare_DT( ){

 static const int NHisto = 23;

 TText* te = new TText();
 te->SetTextSize(0.1);
 
  gROOT->ProcessLine(".x HistoCompare.C");
  HistoCompare * myPV = new HistoCompare();

 char*  reffilename  = "${REFFILE}";//"./DTSimHitsPlots_ref.root";
 char*  curfilename  = "${CURFILE}";//"./DTSimHitsPlots.root";

 TFile * reffile = new TFile(reffilename);
 TFile * curfile = new TFile(curfilename);

 //1-Dimension Histogram
 char* label[NHisto];
 label[0] = "Number_of_all_DT_hits";
 label[1] = "Number_of_muon_DT_hits";
 label[2] = "Wheel_occupancy";
 label[3] = "Station_occupancy";
 label[4] = "Sector_occupancy";
 label[5] = "SuperLayer_occupancy";
 label[6] = "Layer_occupancy";
 label[7] = "Wire_occupancy";
 label[8] = "DT_energy_loss";
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



 TH1F* htemp1[NHisto];
 TH1F* htemp2[NHisto];
// cout << " entro en bucle" << endl;

 for ( int i = 0; i< NHisto ; i++ ) {
   char title[50];
   TCanvas c1;
//   cout << "label(i)" << label[i] << endl;

   if ( i<19 ) 
   {
     htemp1[i]  = dynamic_cast<TH1F*>(reffile->Get(label[i]));
     htemp2[i]  = dynamic_cast<TH1F*>(curfile->Get(label[i]));
//  if( htemp1[i] == 0 || htemp2[i] == 0) continue;
     if( htemp1[i] == 0 ) std::cout << " reference histo is empty " << endl;
     if( htemp2[i] == 0 ) std::cout << " current histo is empty " << endl;

     htemp1[i]->SetLineColor(2);
     htemp2[i]->SetLineColor(4);
     htemp1[i]->SetLineStyle(3);
     htemp2[i]->SetLineStyle(5);
     TLegend leg(0.1, 0.15, 0.2, 0.25);
     leg.AddEntry(htemp1[i], "Reference", "l");
     leg.AddEntry(htemp2[i], "New ", "l");
//   if (i>14 && i<19 || i>29 && i< 34 || i == 13 || i>37 ) c1.SetLogy();

     htemp1[i]->Draw();

     htemp2[i]->Draw("Same"); 
     leg.Draw();
     myPV->PVCompute(htemp1[i],htemp2[i], te);
     sprintf(title,"%s%s", label[i],".eps");
     c1.Print(title);
   
   } else {

     htemp1[i]  = dynamic_cast<TH2F*>(reffile->Get(label[i]));
     htemp2[i]  = dynamic_cast<TH2F*>(curfile->Get(label[i]));
 
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



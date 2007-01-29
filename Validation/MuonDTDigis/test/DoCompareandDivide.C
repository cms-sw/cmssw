#include "TText.h"
#include "TFile.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TLegend.h"

void DoCompareandDivide( ){

 static const int NHisto = 8;

 TText* te = new TText();
 te->SetTextSize(0.1);
 

  char*  reffilename  = "${REFFILE}";  // "./DTDigiPlots.root";

 TFile * reffile = new TFile(reffilename);

 //1-Dimension Histogram
 char* label[NHisto];
 label[0] = "Simhit_occupancy_MB1";
 label[1] = "Digi_occupancy_MB1";
 label[2] = "Simhit_occupancy_MB2";
 label[3] = "Digi_occupancy_MB2";
 label[4] = "Simhit_occupancy_MB3";
 label[5] = "Digi_occupancy_MB3";
 label[6] = "Simhit_occupancy_MB4";
 label[7] = "Digi_occupancy_MB4";


 TH1F* htemp1;
 TH1F* htemp2;
 for ( int i = 0; i<4 ; i++ ) {
   char title[50];
   TCanvas c1;
   htemp1  = dynamic_cast<TH1F*>(reffile->Get(label[2*i]));
   htemp2  = dynamic_cast<TH1F*>(reffile->Get(label[2*i+1]));
//   if( htemp1 == 0 || htemp2 == 0) continue;
   htemp1->SetLineColor(4);
   htemp2->SetLineColor(2);
   htemp1->SetLineStyle(1);
   htemp2->SetLineStyle(3);
   htemp1->SetLineWidth(2);
   TLegend leg(0.1, 0.15, 0.2, 0.25);
   leg.AddEntry(htemp1, "Reference", "l");
   leg.AddEntry(htemp2, "New ", "l");
 
   htemp2->Draw();

   htemp1->Draw("Same"); 
   leg.Draw();
   sprintf(title,"%s%s", label[2*i],"_and_Digi.jpg");   
   c1.Print(title);

   TH1F* hout = new TH1F(*htemp1);
   hout->Reset();
   hout->SetName((std::string(htemp1->GetName()) + std::string("_by_") + std::string(htemp2->GetName())).c_str());
   

   hout->Divide(htemp2,htemp1,1.,1.,"B");
   hout->Draw();


//   myPV->PVCompute(htemp1[i],htemp2[i], te);
   sprintf(title,"%s%s", label[2*i+1],"_over_Digi_ratio.jpg");
  c1.Print(title);
 }


}



#include "TText.h"
#include "TFile.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TLegend.h"

void DoCompareandDivide( ){

 static const int NHisto = 8;

 TText* te = new TText();
 te->SetTextSize(0.1);
 
 TPaveStats* st_1;
 TPaveStats* st_2;

  char*  reffilename  = "${REFFILE}";  // "./DTDigiPlots.root";

 TFile * reffile = new TFile(reffilename);

 reffile->cd("DQMData/DTDigiValidationTask");
 gDirectory->ls(); 

 //1-Dimension Histogram
 char* label[NHisto];
 char* label_dir[NHisto];
 char stringall[80];
 
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
 
   sprintf(stringall, "DQMData/DTDigiValidationTask/%s",label[2*i]);
   label_dir[2*i] = stringall;
   htemp1  = dynamic_cast<TH1F*>(reffile->Get(label_dir[2*i]));

   sprintf(stringall, "DQMData/DTDigiValidationTask/%s",label[2*i+1]);
   label_dir[2*i+1] = stringall; 
   htemp2  = dynamic_cast<TH1F*>(reffile->Get(label_dir[2*i+1]));
  
   if( htemp1 == 0 ) std::cout << " SimHits histo is empty " << endl;
   if( htemp2 == 0 ) std::cout << " Digis histo is empty " << endl;
   if( htemp1 == 0 || htemp2 == 0) continue;
   htemp1->SetLineColor(4);
   htemp2->SetLineColor(6);
   htemp1->SetLineStyle(1);
   htemp2->SetLineStyle(1);
   htemp1->SetLineWidth(2);
   htemp2->SetLineWidth(2);
   TLegend leg(0.1, 0.2, 0.2, 0.3);
   leg.AddEntry(htemp1, "Muon SimHits", "l");
   leg.AddEntry(htemp2, "All Digis ", "l");
 
   htemp2->Draw();
   gStyle->SetOptStat(1111);
   st_2 = (TPaveStats*)htemp2->GetListOfFunctions()->FindObject("stats");

   htemp1->Draw();
   gStyle->SetOptStat(1111);
   st_1 = (TPaveStats*)htemp1->GetListOfFunctions()->FindObject("stats");

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
   htemp2->Draw();
   gStyle->SetOptStat(000000);
   htemp1->Draw("Same");
   sta_2->Draw("Same");
   sta_1->Draw("Same");

   leg.Draw();
   sprintf(title,"%s%s", label[2*i],"_and_Digi.eps");   
   c1.Print(title);

   TH1F* hout = new TH1F(*htemp1);
   hout->Reset();
   hout->SetName((std::string(htemp1->GetName()) + std::string("_by_") + std::string(htemp2->GetName())).c_str());
   

   hout->Divide(htemp2,htemp1,1.,1.,"B");
   hout->Draw();


//   myPV->PVCompute(htemp1[i],htemp2[i], te);
   sprintf(title,"%s%s", label[2*i+1],"_over_Digi_ratio.eps");
  c1.Print(title);
 }


}



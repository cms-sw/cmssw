
#include <exception>
#include <vector>

#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TChain.h"
#include "TObject.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TLegend.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TTree.h"
#include "TF1.h"
#include "TCutG.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TMultiGraph.h"
#include "TPaveText.h"
#include "TProfile.h"
#include "tdrstyle.C"
#include "../../ICHEP_Analysis/Analysis_PlotFunction.h"

using namespace std;

void getScaleFactor(TFile* InputFile, string OutName, string ObjName1, string ObjName2);
void ExtractConstants(TH2D* input);
void compareDataMC(TH2D* inputD, TH2D* inputM);

double GetMass(double P, double I){
   const double K = 2.529; //Harm2
   const double C = 2.772; //Harm2

   return sqrt((I-C)/K)*P;
}

TF1* GetMassLine(double M)
{  
   const double K = 2.529; //Harm2
   const double C = 2.772; //Harm2

   double BetaMax = 0.9;
   double PMax = sqrt((BetaMax*BetaMax*M*M)/(1-BetaMax*BetaMax));

   double BetaMin = 0.2;
   double PMin = sqrt((BetaMin*BetaMin*M*M)/(1-BetaMin*BetaMin));

   TF1* MassLine = new TF1("MassLine","[2] + ([0]*[0]*[1])/(x*x)", PMin, PMax);
   MassLine->SetParName  (0,"M");
   MassLine->SetParName  (1,"K");
   MassLine->SetParName  (2,"C");
   MassLine->SetParameter(0, M);
   MassLine->SetParameter(1, K);
   MassLine->SetParameter(2, C);
   MassLine->SetLineWidth(2);
   return MassLine;
}

void MakePlot()
{
   setTDRStyle();
   gStyle->SetPadTopMargin   (0.06);
   gStyle->SetPadBottomMargin(0.10);
   gStyle->SetPadRightMargin (0.18);
   gStyle->SetPadLeftMargin  (0.125);
   gStyle->SetTitleSize(0.04, "XYZ");
   gStyle->SetTitleXOffset(1.1);
   gStyle->SetTitleYOffset(1.35);
//   gStyle->SetPalette(51); 
   gStyle->SetPalette(1); 
   gStyle->SetNdivisions(510,"X");

   TF1* PionLine = GetMassLine(0.140);
   PionLine->SetLineColor(2);
   PionLine->SetLineWidth(2);

   TF1* KaonLine = GetMassLine(0.494);
   KaonLine->SetLineColor(2);
   KaonLine->SetLineWidth(2);

   TF1* ProtonLine = GetMassLine(0.938);
   ProtonLine->SetLineColor(2);
   ProtonLine->SetLineWidth(2);

   TF1* DeuteronLine = GetMassLine(1.88);
   DeuteronLine->SetLineColor(2);
   DeuteronLine->SetLineWidth(2);

   TF1* TritonLine = GetMassLine(2.80);
   TritonLine->SetLineColor(2);
   TritonLine->SetLineWidth(2);


   TFile* InputFile = new TFile("pictures/Histos.root");
   std::vector<string> ObjName;
   ObjName.push_back("dedx");


   for(unsigned int i=0;i<ObjName.size();i++){
      TH1D*       HdedxMIP        = (TH1D*)    GetObjectFromPath(InputFile, (ObjName[i] + "_MIP"      ).c_str() );
      TH1D*       HMass           = (TH1D*)    GetObjectFromPath(InputFile, (ObjName[i] + "_Mass"     ).c_str() );
      TH2D*       HdedxVsP        = (TH2D*)    GetObjectFromPath(InputFile, (ObjName[i] + "_dedxVsP"  ).c_str() );
      TH2D*       HdedxVsQP       = (TH2D*)    GetObjectFromPath(InputFile, (ObjName[i] + "_dedxVsQP" ).c_str() );
      TProfile*   HdedxVsPProfile = (TProfile*)GetObjectFromPath(InputFile, (ObjName[i] + "_Profile"  ).c_str() );

      //ExtractConstants(HdedxVsP);

      std::cout << "TESTA\n";
      TCanvas* c1 = new TCanvas("c1", "c1", 600,600);
      c1->SetLogz(true);
      HdedxVsP->SetStats(kFALSE);
      HdedxVsP->GetXaxis()->SetTitle("track momentum (GeV/c)");
      HdedxVsP->GetYaxis()->SetTitle("dE/dx (MeV/cm)");
      HdedxVsP->SetAxisRange(0,5,"X");
      HdedxVsP->SetAxisRange(0,15,"Y");
      HdedxVsP->Draw("COLZ");

      KaonLine->Draw("same");
      ProtonLine->Draw("same");
      DeuteronLine->Draw("same");
      TritonLine->Draw("same");
      DrawPreliminary("",8,-1);
      SaveCanvas(c1, "pictures/", ObjName[i] + "_dedxVsP", true);
      delete c1;


      c1 = new TCanvas("c1", "c1", 600,600);
      c1->SetLogz(true);
      HdedxVsQP->SetStats(kFALSE);
      HdedxVsQP->GetXaxis()->SetTitle("charge X track momentum (GeV/c)");
      HdedxVsQP->GetYaxis()->SetTitle("dE/dx (MeV/cm)");
      HdedxVsQP->SetAxisRange(-5,5,"X");
      HdedxVsQP->SetAxisRange(0,15,"Y");
      HdedxVsQP->Draw("COLZ");

      //KaonLine->Draw("same");
      //ProtonLine->Draw("same");
      //DeuteronLine->Draw("same");
      //TritonLine->Draw("same");
      DrawPreliminary("",8,-1);
      SaveCanvas(c1, "pictures/", ObjName[i] + "_dedxVsQP", true);
      delete c1;



      std::cout << "TESTB\n";


      c1 = new TCanvas("c1", "c1", 600,600);
      HdedxVsPProfile->SetStats(kFALSE);
      HdedxVsPProfile->SetAxisRange(2.5,5,"Y");
      HdedxVsPProfile->GetXaxis()->SetTitle("track momentum (GeV/c)");
      HdedxVsPProfile->GetYaxis()->SetTitle("dE/dx (MeV/cm)");
      HdedxVsPProfile->Draw("");
      SaveCanvas(c1, "pictures/", ObjName[i] + "_Profile");
      delete c1;

      c1 = new TCanvas("c1", "c1", 600,600);
      c1->SetLogy(true);
      c1->SetGridx(true);
      HdedxMIP->SetStats(kFALSE);
      HdedxMIP->GetXaxis()->SetTitle("dE/dx (MeV/cm)");
      HdedxMIP->GetYaxis()->SetTitle("number of tracks");
      HdedxMIP->SetAxisRange(0,5,"X");
      HdedxMIP->Draw("");
      SaveCanvas(c1, "pictures/", ObjName[i] + "_MIP", true);
      delete c1;

      std::cout << "TESTC\n";


      TLine* lineKaon = new TLine(0.493667, HMass->GetMinimum(), 0.493667, HMass->GetMaximum());
      lineKaon->SetLineWidth(2);
      lineKaon->SetLineStyle(2);
      lineKaon->SetLineColor(9);
      TLine* lineProton = new TLine(0.938272, HMass->GetMinimum(), 0.938272, HMass->GetMaximum());
      lineProton->SetLineWidth(2);
      lineProton->SetLineStyle(2);
      lineProton->SetLineColor(9);
      TLine* lineDeuteron = new TLine(1.88, HMass->GetMinimum(), 1.88, HMass->GetMaximum());
      lineDeuteron->SetLineWidth(2);
      lineDeuteron->SetLineStyle(2);
      lineDeuteron->SetLineColor(9);
      TLine* lineTriton = new TLine(2.80, HMass->GetMinimum(), 2.80, HMass->GetMaximum());
      lineTriton->SetLineWidth(2);
      lineTriton->SetLineStyle(2);
      lineTriton->SetLineColor(9);

      c1 = new TCanvas("c1", "c1", 600,600);
      c1->SetLogy(true);
      c1->SetGridx(true);
      HMass->Reset();
      for(int x=1;x<=HdedxVsP->GetNbinsX();x++){
      if(HdedxVsP->GetXaxis()->GetBinCenter(x)>3.0)continue;
      for(int y=1;y<=HdedxVsP->GetNbinsY();y++){
        if(HdedxVsP->GetYaxis()->GetBinCenter(y)<4.5)continue;
        HMass->Fill(GetMass(HdedxVsP->GetXaxis()->GetBinCenter(x),HdedxVsP->GetYaxis()->GetBinCenter(y)),HdedxVsP->GetBinContent(x,y));
      }}
      HMass->SetStats(kFALSE);
      HMass->GetXaxis()->SetTitle("Mass (GeV/c^{2})");
      HMass->GetYaxis()->SetTitle("number of tracks");
      HMass->SetAxisRange(0,5,"X");
      HMass->Draw("");
      lineKaon->Draw("same");
      lineProton->Draw("same");
      lineDeuteron->Draw("same");
      lineTriton->Draw("same");
      SaveCanvas(c1, "pictures/", ObjName[i] + "_Mass", true);
      delete c1;

      std::cout << "TESTD\n";

   }

//   getScaleFactor(InputFile, "All_Rescale", "All_SelTrack_Pixel", "All_SelTrack_StripCleaned");


   TFile* InputFileD = new TFile("pictures_data/Histos.root");
   TH2D*       HIasVsPM_D        = (TH2D*)    GetObjectFromPath(InputFileD, "dedx_IasVsPM");
   TFile* InputFileM = new TFile("pictures_MC/Histos.root");
   TH2D*       HIasVsPM_M        = (TH2D*)    GetObjectFromPath(InputFileM, "dedx_IasVsPM");

    compareDataMC(HIasVsPM_D, HIasVsPM_M);
}



void getScaleFactor(TFile* InputFile, string OutName, string ObjName1, string ObjName2){
   TProfile*   HdedxVsPProfile1 = (TProfile*)GetObjectFromPath(InputFile, (ObjName1 + "_Profile"  ).c_str() );
   TProfile*   HdedxVsPProfile2 = (TProfile*)GetObjectFromPath(InputFile, (ObjName2 + "_Profile"  ).c_str() );

   TH1D*       HdedxMIP1        = (TProfile*)GetObjectFromPath(InputFile, (ObjName1 + "_MIP"  ).c_str() );
   TH1D*       HdedxMIP2        = (TProfile*)GetObjectFromPath(InputFile, (ObjName2 + "_MIP"  ).c_str() );

   TF1* mygausMIP = new TF1("mygausMIP","gaus", 1, 5);
   HdedxMIP1->Fit("mygausMIP","Q0","");
   double peakMIP1  = mygausMIP->GetParameter(1);
   HdedxMIP2->Fit("mygausMIP","Q0","");
   double peakMIP2  = mygausMIP->GetParameter(1);

   std::cout << "SCALE FACTOR WITH MIP     = " << peakMIP1/peakMIP2 << endl;

   TH1D* Chi2Dist = new TH1D("Chi2Dist","Chi2Dist",300, 0.9 ,1.15);

   double Minimum = 999999;
   double AbsGain = -1;

   for(int i=1;i<=Chi2Dist->GetNbinsX();i++){
      double ScaleFactor = Chi2Dist->GetXaxis()->GetBinCenter(i);
      TProfile* Rescaled = (TProfile*)HdedxVsPProfile2->Clone("Cloned");
      Rescaled->Scale(ScaleFactor);
      double Dist = 0;
      double Error = 0;
      for(int x=1;x<=HdedxVsPProfile1->GetNbinsX();x++){
         double Momentum = HdedxVsPProfile1->GetXaxis()->GetBinCenter(x);
         if(Momentum<5)continue;//|| Momentum>20)continue;
         if(HdedxVsPProfile1->GetBinError(x)<=0)continue;
         Dist += pow(HdedxVsPProfile1->GetBinContent(x) - Rescaled->GetBinContent(x),2) / std::max(1E-8,pow(HdedxVsPProfile1->GetBinError(x),2));
         Error += pow(HdedxVsPProfile1->GetBinError(x),2);
      }
      Dist *= Error;

      if(Dist<Minimum){Minimum=Dist;AbsGain=ScaleFactor;}

      //std::cout << "Rescale = " << ScaleFactor << " --> SquareDist = " << Dist << endl;
      Chi2Dist->Fill(ScaleFactor,Dist);
      delete Rescaled;
   }

   std::cout << "SCALE FACTOR WITH PROFILE = " << AbsGain << endl;

   TCanvas* c1 = new TCanvas("c1", "c1", 600,600);
   HdedxMIP2->SetStats(kFALSE);
   HdedxMIP2->SetAxisRange(0,10,"X");
   HdedxMIP2->GetXaxis()->SetNdivisions(516);
   HdedxMIP2->GetXaxis()->SetTitle("dE/dx (MeV/cm)");
   HdedxMIP2->GetYaxis()->SetTitle("Tracks");
   HdedxMIP2->SetLineColor(1);
   HdedxMIP2->Draw("");
   TH1D* HdedxMIP3 = (TH1D*)HdedxMIP2->Clone("aaa");
   HdedxMIP3->SetLineColor(8);
   HdedxMIP3->GetXaxis()->Set(HdedxMIP3->GetXaxis()->GetNbins(), HdedxMIP3->GetXaxis()->GetXmin()*2.0, HdedxMIP3->GetXaxis()->GetXmax()*(peakMIP1/peakMIP2) );
   HdedxMIP3->Draw("same");
   TH1D* HdedxMIP4 = (TH1D*)HdedxMIP2->Clone("bbb");
   HdedxMIP4->SetLineColor(4);
   HdedxMIP4->GetXaxis()->Set(HdedxMIP4->GetXaxis()->GetNbins(), HdedxMIP4->GetXaxis()->GetXmin()*2.0, HdedxMIP4->GetXaxis()->GetXmax()*(AbsGain) );
   HdedxMIP4->Draw("same");
   HdedxMIP1->SetLineColor(2);
   HdedxMIP1->Draw("same");
   c1->SetLogy(true);
   c1->SetGridx(true); 
   SaveCanvas(c1, "pictures/", OutName + "_MIP");
   delete c1;


   c1 = new TCanvas("c1", "c1", 600,600);
   Chi2Dist->SetStats(kFALSE);
   Chi2Dist->GetXaxis()->SetNdivisions(504);
   Chi2Dist->GetXaxis()->SetTitle("Rescale Factor");
   Chi2Dist->GetYaxis()->SetTitle("Weighted Square Distance");
   Chi2Dist->Draw("");
   c1->SetLogy(true);
   c1->SetGridx(true); 
   SaveCanvas(c1, "pictures/", OutName + "_Dist");
   delete c1;

   c1 = new TCanvas("c1", "c1", 600,600);
   HdedxVsPProfile1->SetStats(kFALSE);
   HdedxVsPProfile1->SetAxisRange(5,50,"X");
   HdedxVsPProfile1->SetAxisRange(2.5,3.5,"Y");
   HdedxVsPProfile1->GetXaxis()->SetTitle("track momentum (GeV/c)");
   HdedxVsPProfile1->GetYaxis()->SetTitle("dE/dx (MeV/cm)");
   HdedxVsPProfile1->SetMarkerColor(2);
   HdedxVsPProfile1->Draw("");

   HdedxVsPProfile2->SetMarkerColor(1);
   HdedxVsPProfile2->Draw("same");
   TProfile* HdedxVsPProfile3 = (TProfile*)HdedxVsPProfile2->Clone("abc");
   HdedxVsPProfile3->SetMarkerColor(8);
   HdedxVsPProfile3->Scale(peakMIP1/peakMIP2);
   HdedxVsPProfile3->Draw("same");
   TProfile* HdedxVsPProfile4 = (TProfile*)HdedxVsPProfile2->Clone("afs");
   HdedxVsPProfile4->SetMarkerColor(4);
   HdedxVsPProfile4->Scale(AbsGain);
   HdedxVsPProfile4->Draw("same");

   SaveCanvas(c1, "pictures/", OutName + "_Profile");
   delete c1;
}


void compareDataMC(TH2D* inputD, TH2D* inputM){
       inputD->Rebin2D(5,10);
       inputM->Rebin2D(5,10);
       char buffer[2048];

       TH1D* FitResult = new TH1D("FitResult"       , "FitResult"      ,inputD->GetXaxis()->GetNbins(),inputD->GetXaxis()->GetXmin(),inputD->GetXaxis()->GetXmax());
       FitResult->SetTitle("");
       FitResult->SetStats(kFALSE);  
       FitResult->GetXaxis()->SetTitle("P [GeV/c]");
       FitResult->GetYaxis()->SetTitle("dE/dx");
       FitResult->GetYaxis()->SetTitleOffset(1.20);
       FitResult->Reset();
       TH1D* FitResultD = (TH1D*)FitResult->Clone("FitResultD");
       TH1D* FitResultM = (TH1D*)FitResult->Clone("FitResultM");
       TH1D* FitResultSD = (TH1D*)FitResult->Clone("FitResultSD");
       TH1D* FitResultSM = (TH1D*)FitResult->Clone("FitResultSM");

       TGraphErrors* GResultD  = new TGraphErrors(100);
       TGraphErrors* GResultM  = new TGraphErrors(100);
       TGraphErrors* GResultSD = new TGraphErrors(100);
       TGraphErrors* GResultSM = new TGraphErrors(100);

      TCanvas* c1 = new TCanvas("c1", "c1", 600,600);
      c1->SetLogz(true);
      inputD->SetStats(kFALSE);
      inputD->GetXaxis()->SetTitle("track momentum (GeV/c)");
      inputD->GetYaxis()->SetTitle("dE/dx (MeV/cm)");
      inputD->SetAxisRange(0,5,"X");
      inputD->SetAxisRange(0,15,"Y");
      inputD->Draw("COLZ");
      SaveCanvas(c1, "pictures/", "DataMC_D_dedxVsP");
      delete c1;

      c1 = new TCanvas("c1", "c1", 600,600);
      c1->SetLogz(true);
      inputM->SetStats(kFALSE);
      inputM->GetXaxis()->SetTitle("track momentum (GeV/c)");
      inputM->GetYaxis()->SetTitle("dE/dx (MeV/cm)");
      inputM->SetAxisRange(0,5,"X");
      inputM->SetAxisRange(0,15,"Y");
      inputM->Draw("COLZ");
      SaveCanvas(c1, "pictures/", "DataMC_M_dedxVsP");
      delete c1;

       int NM=0; int ND=0;
       for(int x=1;x<inputD->GetXaxis()->FindBin(5);x++){
          double P       = inputD->GetXaxis()->GetBinCenter(x);
          double PErr    = inputD->GetXaxis()->GetBinWidth(x);

    
          TH1D* Projection = (TH1D*)(inputD->ProjectionY("proj",x,x))->Clone();
          if(Projection->Integral()>100){
             Projection->SetAxisRange(0.1,25,"X");
             Projection->Sumw2();
             Projection->Scale(1.0/Projection->Integral());

             TF1* mygaus = new TF1("mygaus","gaus", 0, 15);
             Projection->Fit("mygaus","Q0 RME");
             double chiFromFit  = (mygaus->GetChisquare())/(mygaus->GetNDF());
             FitResultD->SetBinContent(x, mygaus->GetParameter(1));
             FitResultD->SetBinError  (x, mygaus->GetParError (1));
             FitResultSD->SetBinContent(x, mygaus->GetParameter(2));
             FitResultSD->SetBinError  (x, mygaus->GetParError (2));
             GResultD->SetPoint(ND, P, mygaus->GetParameter(1));
             GResultD->SetPointError  (ND, PErr, mygaus->GetParError (1));
             GResultSD->SetPoint(ND, P, mygaus->GetParameter(2));
             GResultSD->SetPointError  (ND, PErr, mygaus->GetParError (2));
             ND++;
             mygaus->SetLineColor(2);
             mygaus->SetLineWidth(2);

             c1  = new TCanvas("canvas", "canvas", 600,600);
             Projection->Draw();
             Projection->SetTitle("");
             Projection->SetStats(kFALSE);
             Projection->GetXaxis()->SetTitle("dE/dx Estimator [MeV/cm]");
             Projection->GetYaxis()->SetTitle("#Entries");
             Projection->GetYaxis()->SetTitleOffset(1.30);
             Projection->SetAxisRange(1E-5,1.0,"Y");

             mygaus->Draw("same");

             TPaveText* stt = new TPaveText(0.55,0.82,0.79,0.92, "NDC");
             stt->SetFillColor(0);
             stt->SetTextAlign(31);
             sprintf(buffer,"Proton  #mu:%5.1fMeV/cm",mygaus->GetParameter(1));      stt->AddText(buffer);
             sprintf(buffer,"Proton  #sigma:%5.1fMeV/cm",mygaus->GetParameter(2));      stt->AddText(buffer);
             stt->Draw("same");

             //std::cout << "P = " << P << "  --> Proton dE/dx = " << mygaus->GetParameter(1) << endl;

             c1->SetLogy(true);
             sprintf(buffer,"%s_ProjectionFit_P%03i_%03i","DataMC_D",(int)(100*FitResult->GetXaxis()->GetBinLowEdge(x)),(int)(100*FitResult->GetXaxis()->GetBinUpEdge(x)) );
             SaveCanvas(c1,"pictures/",buffer);
             delete c1;
          }
 
          Projection = (TH1D*)(inputM->ProjectionY("proj",x,x))->Clone();
          if(Projection->Integral()>100){
             Projection->SetAxisRange(0.1,25,"X");
             Projection->Sumw2();
             Projection->Scale(1.0/Projection->Integral());

             TF1* mygaus = new TF1("mygaus","gaus", 0, 15);
             Projection->Fit("mygaus","Q0 RME");
             double chiFromFit  = (mygaus->GetChisquare())/(mygaus->GetNDF());
             FitResultM->SetBinContent(x, mygaus->GetParameter(1));
             FitResultM->SetBinError  (x, mygaus->GetParError (1));
             FitResultSM->SetBinContent(x, mygaus->GetParameter(2));
             FitResultSM->SetBinError  (x, mygaus->GetParError (2));
             GResultM->SetPoint(NM, P, mygaus->GetParameter(1));
             GResultM->SetPointError  (NM, PErr, mygaus->GetParError (1));
             GResultSM->SetPoint(NM, P, mygaus->GetParameter(2));
             GResultSM->SetPointError  (NM, PErr, mygaus->GetParError (2));
             NM++;

             mygaus->SetLineColor(2);
             mygaus->SetLineWidth(2);

             c1  = new TCanvas("canvas", "canvas", 600,600);
             Projection->Draw();
             Projection->SetTitle("");
             Projection->SetStats(kFALSE);
             Projection->GetXaxis()->SetTitle("dE/dx Estimator [MeV/cm]");
             Projection->GetYaxis()->SetTitle("#Entries");
             Projection->GetYaxis()->SetTitleOffset(1.30);
             Projection->SetAxisRange(1E-5,1.0,"Y");

             mygaus->Draw("same");

             TPaveText* stt = new TPaveText(0.55,0.82,0.79,0.92, "NDC");
             stt->SetFillColor(0);
             stt->SetTextAlign(31);
             sprintf(buffer,"Proton  #mu:%5.1fMeV/cm",mygaus->GetParameter(1));      stt->AddText(buffer);
             sprintf(buffer,"Proton  #sigma:%5.1fMeV/cm",mygaus->GetParameter(2));      stt->AddText(buffer);
             stt->Draw("same");

             //std::cout << "P = " << P << "  --> Proton dE/dx = " << mygaus->GetParameter(1) << endl;

             c1->SetLogy(true);
             sprintf(buffer,"%s_ProjectionFit_P%03i_%03i","DataMC_M",(int)(100*FitResult->GetXaxis()->GetBinLowEdge(x)),(int)(100*FitResult->GetXaxis()->GetBinUpEdge(x)) );
             SaveCanvas(c1,"pictures/",buffer);
             delete c1;
          }
       }
       c1  = new TCanvas("canvas", "canvas", 600,600);
       FitResultD->SetAxisRange(0,2.5,"X");
       FitResultD->SetAxisRange(0,1,"Y");
       FitResultD->Draw("");
       FitResultM->SetAxisRange(0,2.5,"X");
       FitResultM->SetAxisRange(0,1,"Y");
       FitResultM->Draw("same");
       SaveCanvas(c1,"pictures/","DataMC_Mean");
       delete c1;

      GResultD->Set(ND-1);
      GResultSD->Set(ND-1);
      GResultM->Set(NM-1);
      GResultSM->Set(NM-1);


      c1 = new TCanvas("c1", "c1",600,600);
      TMultiGraph* GraphM = new TMultiGraph();
      GResultD->SetLineColor(1);       GResultD->SetMarkerColor(1);
      GResultM->SetLineColor(4);       GResultM->SetMarkerColor(4);
      GraphM->Add(GResultD     ,"LP");
      GraphM->Add(GResultM     ,"LP");
      GraphM->Draw("A");
      GraphM->SetTitle("");
      GraphM->GetXaxis()->SetTitle("P");
      GraphM->GetYaxis()->SetTitle("#mu(I_{as})");
      GraphM->GetYaxis()->SetTitleOffset(1.70);
      GraphM->GetYaxis()->SetRangeUser(0,1);      
      SaveCanvas(c1,"pictures/","DataMC_MeanG");
      delete c1;


      c1 = new TCanvas("c1", "c1",600,600);
      TMultiGraph* GraphS = new TMultiGraph();
      GResultSD->SetLineColor(1);       GResultSD->SetMarkerColor(1);
      GResultSM->SetLineColor(4);       GResultSM->SetMarkerColor(4);
      GraphS->Add(GResultSD     ,"LP");
      GraphS->Add(GResultSM     ,"LP");
      GraphS->Draw("A");
      GraphS->SetTitle("");
      GraphS->GetXaxis()->SetTitle("P");
      GraphS->GetYaxis()->SetTitle("#sigma(I_{as})");
      GraphS->GetYaxis()->SetTitleOffset(1.70);
      GraphS->GetYaxis()->SetRangeUser(0,0.2);
      SaveCanvas(c1,"pictures/","DataMC_SigmaG");
      delete c1;

}





void ExtractConstants(TH2D* input){
       input->Rebin2D(5,10);
       double MinRange = 0.60;
       double MaxRange = 1.20;
          char buffer[2048];


       TH1D* FitResult = new TH1D("FitResult"       , "FitResult"      ,input->GetXaxis()->GetNbins(),input->GetXaxis()->GetXmin(),input->GetXaxis()->GetXmax());
       FitResult->SetTitle("");
       FitResult->SetStats(kFALSE);  
       FitResult->GetXaxis()->SetTitle("P [GeV/c]");
       FitResult->GetYaxis()->SetTitle("dE/dx Estimator [MeV/cm]");
       FitResult->GetYaxis()->SetTitleOffset(1.20);
       FitResult->Reset();


      TH2D* inputnew = (TH2D*)input->Clone("tempTH2D");
      inputnew->Reset();
      for(int x=1;x<=input->GetNbinsX();x++){
      for(int y=1;y<=input->GetNbinsY();y++){
        double Mass = GetMass(input->GetXaxis()->GetBinCenter(x),input->GetYaxis()->GetBinCenter(y));
        if(isnan(Mass) || Mass<0.94-0.3 || Mass>0.94+0.3)continue;
        inputnew->SetBinContent(x,y,input->GetBinContent(x,y));
      }}


      TCanvas* c1 = new TCanvas("c1", "c1", 600,600);
      c1->SetLogz(true);
      inputnew->SetStats(kFALSE);
      inputnew->GetXaxis()->SetTitle("track momentum (GeV/c)");
      inputnew->GetYaxis()->SetTitle("dE/dx (MeV/cm)");
      inputnew->SetAxisRange(0,5,"X");
      inputnew->SetAxisRange(0,15,"Y");
      inputnew->Draw("COLZ");

//      KaonLine->Draw("same");
//      ProtonLine->Draw("same");
//      DeuteronLine->Draw("same");
//      TritonLine->Draw("same");
      SaveCanvas(c1, "./", "tmp_dedxVsP");
      delete c1;

       for(int x=1;x<inputnew->GetXaxis()->FindBin(5);x++){
          double P       = inputnew->GetXaxis()->GetBinCenter(x);
    
          TH1D* Projection = (TH1D*)(inputnew->ProjectionY("proj",x,x))->Clone();
          if(Projection->Integral()<100)continue;
          Projection->SetAxisRange(0.1,25,"X");
          Projection->Sumw2();
          Projection->Scale(1.0/Projection->Integral());

          TF1* mygaus = new TF1("mygaus","gaus", 2.5, 15);
          Projection->Fit("mygaus","Q0 RME");
          double chiFromFit  = (mygaus->GetChisquare())/(mygaus->GetNDF());
          FitResult->SetBinContent(x, mygaus->GetParameter(1));
          FitResult->SetBinError  (x, mygaus->GetParError (1));
          mygaus->SetLineColor(2);
          mygaus->SetLineWidth(2);

          c1  = new TCanvas("canvas", "canvas", 600,600);
          Projection->Draw();
          Projection->SetTitle("");
          Projection->SetStats(kFALSE);
          Projection->GetXaxis()->SetTitle("dE/dx Estimator [MeV/cm]");
          Projection->GetYaxis()->SetTitle("#Entries");
          Projection->GetYaxis()->SetTitleOffset(1.30);
          Projection->SetAxisRange(1E-5,1.0,"Y");

          mygaus->Draw("same");


          TPaveText* stt = new TPaveText(0.55,0.82,0.79,0.92, "NDC");
          stt->SetFillColor(0);
          stt->SetTextAlign(31);
          sprintf(buffer,"Proton  #mu:%5.1fMeV/cm",mygaus->GetParameter(1));      stt->AddText(buffer);
          sprintf(buffer,"Proton  #sigma:%5.1fMeV/cm",mygaus->GetParameter(2));      stt->AddText(buffer);
          stt->Draw("same");

          //std::cout << "P = " << P << "  --> Proton dE/dx = " << mygaus->GetParameter(1) << endl;

          c1->SetLogy(true);
          sprintf(buffer,"%s_ProjectionFit_P%03i_%03i","tmp",(int)(100*FitResult->GetXaxis()->GetBinLowEdge(x)),(int)(100*FitResult->GetXaxis()->GetBinUpEdge(x)) );
          if(P>=MinRange && P<=MaxRange){SaveCanvas(c1,"./",buffer);}
          delete c1;
       }
       c1  = new TCanvas("canvas", "canvas", 600,600);
       FitResult->SetAxisRange(0,2.5,"X");
       FitResult->SetAxisRange(0,15,"Y");
       FitResult->Draw("");

       TLine* line1 = new TLine(MinRange, FitResult->GetMinimum(), MinRange, FitResult->GetMaximum());
       line1->SetLineWidth(2);
       line1->SetLineStyle(2);
       line1->Draw();

       TLine* line2 = new TLine(MaxRange, FitResult->GetMinimum(), MaxRange, FitResult->GetMaximum());
       line2->SetLineWidth(2);
       line2->SetLineStyle(2);
       line2->Draw();

       //   TF1* myfit = new TF1("myfit","[1]+(pow(0.93827,2) + x*x)/([0]*x*x)", MinRange, MaxRange);
       TF1* myfit = new TF1("myfit","[0]*pow(0.93827/x,2) + [1]", MinRange, MaxRange);
       myfit->SetParName  (0,"K");
       myfit->SetParName  (1,"C");
       myfit->SetParameter(0, 2.7);
       myfit->SetParameter(1, 2.7);
       myfit->SetParLimits(0, 2.00,4.0);
       myfit->SetParLimits(1, 2.00,4.0);
       myfit->SetLineWidth(2);
       myfit->SetLineColor(2);
       FitResult->Fit("myfit", "M R E I 0");
       myfit->SetRange(MinRange,MaxRange);
       myfit->Draw("same");

       TPaveText* st = new TPaveText(0.40,0.78,0.79,0.89, "NDC");
       st->SetFillColor(0);
//       K   [i] = myfit->GetParameter(0);
//       C   [i] = myfit->GetParameter(1);
//       KErr[i] = myfit->GetParError(0);
//       CErr[i] = myfit->GetParError(1);
       sprintf(buffer,"K = %3.2f +- %6.3f",myfit->GetParameter(0), myfit->GetParError(0));
       st->AddText(buffer);
       sprintf(buffer,"C = %3.2f +- %6.3f",myfit->GetParameter(1), myfit->GetParError(1));
       st->AddText(buffer);
       st->Draw("same");
       sprintf(buffer,"%s_Fit","tmp");
       SaveCanvas(c1,"./",buffer);
       delete c1;
}

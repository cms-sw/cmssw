#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdio.h>

#include "TROOT.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TRandom.h"

using namespace std;

void makemap()
{
  // 2012 mean (offset), RMS values in ns

  // DT: 0.5, 1 (from Anna Meneguzzo)

  // RPC: 0.06, 5.14 (from Malgorzata Kazana)
  // Same as 2011 values

  // CSC: should be split into two (from Chris Farrell)
  // Region1 (station==1 && (ring==1 || ring==4)): 0.5, 1
  // Region2 (all other CSC): 2.3, 1

  TRandom *numberGenerator = new TRandom();
  
  float meanRPC = 0.06;
  float sigmaRPC = 5.14;

  float meanDT = 0.5;
  float sigmaDT = 1.0;

  float meanCSC = 2.3;
  float sigmaCSC = 1.0;

  std::ofstream RPCtext;     RPCtext.open("RPC.dat");
  std::ofstream DTtext;      DTtext.open("DT.dat");
  std::ofstream CSCtext;     CSCtext.open("CSC.dat");


  TH1F * histoRPC = new TH1F("histoRPC", "RPC dToF", 101, meanRPC-3*sigmaRPC, meanRPC+3*sigmaRPC);
  TH1F * histoDT  = new TH1F("histoDT", "DT dToF", 101, meanDT-3*sigmaDT, meanDT+3*sigmaDT);
  TH1F * histoCSC = new TH1F("histoCSC", "CSC dToF", 101, meanCSC-3*sigmaCSC, meanCSC+3*sigmaCSC);

  for(int i=1;i<=2316;i++)
    {
      double valueRPC = numberGenerator->Gaus(meanRPC,sigmaRPC);
      histoRPC->Fill(valueRPC);
      RPCtext<<valueRPC<<endl;
      //      cout<<valueRPC<<endl;
    }

  for(int i=1;i<=250;i++)
    {
      double valueDT = numberGenerator->Gaus(meanDT,sigmaDT);
      histoDT->Fill(valueDT);
      DTtext<<valueDT<<endl;
      //      cout<<valueDT<<endl;
    }

  for(int i=1;i<=540;i++)
    {
      double valueCSC = numberGenerator->Gaus(meanCSC,sigmaCSC);
      histoCSC->Fill(valueCSC);
      CSCtext<<valueCSC<<endl;
      //      cout<<valueCSC<<endl;
    }



  RPCtext.close();
  DTtext.close();
  CSCtext.close();

  TCanvas * Ca0 = new TCanvas("Ca0","Canvas",800,600);
  histoRPC->Draw();
  histoRPC->GetXaxis()->SetTitle("RPC");
  Ca0->SaveAs("RPC.png");
  Ca0->Clear();

  histoDT->Draw();
  histoDT->GetXaxis()->SetTitle("DT");
  Ca0->SaveAs("DT.png");
  Ca0->Clear();

  histoCSC->Draw();
  histoCSC->GetXaxis()->SetTitle("CSC");
  Ca0->SaveAs("CSC.png");
  Ca0->Clear();

  gSystem->Exec("source toSourceToMergeAllMuon");

}

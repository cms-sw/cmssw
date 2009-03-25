#include <iostream>
#include "TCanvas.h"
#include "TDirectoryFile.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TIterator.h"
#include "TKey.h"
#include "TLegend.h"
#include "TList.h"
#include "TObjString.h"
#include "TPaveStats.h"
#include "TPRegexp.h"
#include "TRegexp.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"

void muonIdVal(char* filename1, char* filename2 = 0, bool make2DPlots = true, bool printPng = true, bool printHtml = true, bool printEps = false) {
   TFile* f1 = TFile::Open(filename1);
   if (! f1) {
      std::cout << "Error: unable to open " << filename1 << std::endl;
      return;
   }
   TDirectoryFile* d1 = (TDirectoryFile*)f1->FindObjectAny("MuonIdVal");
   if (! d1) {
      std::cout << "Error: MuonIdVal not found in " << filename1 << std::endl;
      return;
   }

   TFile* f2 = 0;
   TDirectoryFile* d2 = 0;
   if (filename2) {
      f2 = TFile::Open(filename2);
      if (! f2) {
         std::cout << "Error: unable to open " << filename2 << std::endl;
         return;
      }
      d2 = (TDirectoryFile*)f2->FindObjectAny("MuonIdVal");
      if (! d2) {
         std::cout << "Error: MuonIdVal not found in " << filename2 << std::endl;
         return;
      }
   }

   TList* list = 0;
   TIterator* iter = 0;
   TKey* key = 0;
   TObject* obj1 = 0;
   TObject* obj2 = 0;

   gROOT->SetStyle("Plain");
   gStyle->SetOptStat(111111);
   gStyle->SetOptFit(1);

   // Retrieve CMSSW versions for TLegend
   TString version1 = ((TObjString*)f1->GetListOfKeys()->At(0))->GetString().Remove(TString::kLeading, '"').Remove(TString::kTrailing, '"');
   TString version2 = "";
   if (f2)
      version2 = ((TObjString*)f2->GetListOfKeys()->At(0))->GetString().Remove(TString::kLeading, '"').Remove(TString::kTrailing, '"');
   //std::cout << "version1: " << version1.Data() << std::endl;
   //std::cout << "version2: " << version2.Data() << std::endl;

   TDirectoryFile* tmd1 = (TDirectoryFile*)d1->Get("TrackerMuons");
   TDirectoryFile* gmd1 = (TDirectoryFile*)d1->Get("GlobalMuons");
   TDirectoryFile* tmd2 = 0;
   TDirectoryFile* gmd2 = 0;
   if (d2) {
      tmd2 = (TDirectoryFile*)d2->Get("TrackerMuons");
      gmd2 = (TDirectoryFile*)d2->Get("GlobalMuons");
   }

   TCanvas* c1 = new TCanvas("c1");
   c1->Draw();
   TPaveStats* s1 = 0;
   TPaveStats* s2 = 0;
   TLegend* leg = 0;
   TRegexp re("*Pull[dxy]*", kTRUE);

   // Access histograms in three different locations:
   //   d1 -> MuonIdVal
   // tmd1 -> MuonIdVal/TrackerMuons
   // gmd1 -> MuonIdVal/GlobalMuons
   char pfx[3][4] = {"", "tm_", "gm_"};

   for(unsigned int i = 0; i < 3; i++) {
      if (i == 0) {}
      else if (i == 1 && tmd1) { d1 = tmd1; d2 = tmd2; }
      else if (i == 2 && gmd1) { d1 = gmd1; d2 = gmd2; }
      else continue;

      list = d1->GetListOfKeys();
      iter = list->MakeIterator();
      while((key = (TKey*)iter->Next())) {
         obj1 = key->ReadObj();
         obj2 = 0;
         s2   = 0;
         if (d2)
            obj2 = d2->Get(obj1->GetName());
         // For backwards compatibility with old dqm files where histograms may be
         // stored in different directories, try FindObjectAny before giving up
         if (! obj2)
            obj2 = f2->FindObjectAny(obj1->GetName());

         if (obj1->InheritsFrom(TH1::Class()) || (obj1->InheritsFrom(TH2::Class()) && make2DPlots)) {
            // If there are two TH1Fs better normalize them
            // If just one don't bother
            if (obj2 && obj1->InheritsFrom(TH1F::Class())) {
               ((TH1F*)obj1)->Scale(1./((TH1F*)obj1)->Integral());
               ((TH1F*)obj2)->Scale(1./((TH1F*)obj2)->Integral());
            }

            if ((TString(obj1->GetName()).Index(re) >= 0)) {
               ((TH1F*)obj1)->Fit("gaus", "", "", -2., 2.);
               if (obj2)
                  ((TH1F*)obj2)->Fit("gaus", "", "", -2., 2.);
            }

            ((TH1*)obj1)->SetLineColor(4);
            ((TH1*)obj1)->SetMarkerColor(4);
            ((TH1*)obj1)->Draw();
            c1->Update();
            s1 = (TPaveStats*)obj1->FindObject("stats");
            s1->SetFillStyle(0);

            if (obj2) {
               ((TH1*)obj2)->SetLineColor(2);
               ((TH1*)obj2)->SetMarkerColor(2);
               ((TH1*)obj2)->Draw("sames");
               c1->Update();

               s2 = (TPaveStats*)obj2->FindObject("stats");
               s2->SetFillStyle(0);
               double height = s2->GetY2NDC()-s2->GetY1NDC();
               s2->SetY2NDC(s1->GetY1NDC()-.005);
               s2->SetY1NDC(s2->GetY2NDC()-height);

               // If obj2 has a higher maximum, set obj1 to that
               // Again only an issue for TH1F
               if (obj2->InheritsFrom(TH1F::Class()))
                  if (((TH1F*)obj2)->GetMaximum() > ((TH1F*)obj1)->GetMaximum())
                     ((TH1F*)obj1)->SetMaximum(((TH1F*)obj2)->GetMaximum()*1.05);
            }

            leg = 0;
            leg = new TLegend(0., 0., .22, .06);
            leg->SetFillColor(0);
            leg->AddEntry(obj1, version1.Data(), "l");
            if (obj2)
               leg->AddEntry(obj2, version2.Data(), "l");
            leg->Draw();
            c1->Update();

            // Output png files
            if (printPng)
               c1->Print(Form("%s%s.png", pfx[i], obj1->GetName()));

            delete leg;
         }
      }
   }

   f1->Close();
   if (f2)
      f2->Close();

   TPRegexp preg = TPRegexp("(\\S+).root");
   TString output = ((TObjString*)preg.MatchS(TString(filename1))->At(1))->GetString();

   // Output html file
   if (printHtml) {
   }

   // Output eps file
   if (printEps) {
   }
}

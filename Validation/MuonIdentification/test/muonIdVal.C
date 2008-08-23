#include <iostream>
#include "TCanvas.h"
#include "TDirectoryFile.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TIterator.h"
#include "TKey.h"
#include "TList.h"
#include "TPRegexp.h"
#include "TRegexp.h"
#include "TString.h"
#include "TStyle.h"
#include "TText.h"

void muonIdVal(char* filename, bool make2DPlots = false, bool makeAllChamberPlots = false) {
   TFile* f = TFile::Open(filename);
   if (! f) {
      std::cout << "Error: unable to open " << filename << std::endl;
      return;
   }
   TDirectoryFile* d = (TDirectoryFile*)f->FindObjectAny("MuonIdVal");
   if (! d) {
      std::cout << "Error: MuonIdVal not found in " << filename << std::endl;
      return;
   }

   bool useTrackerMuons = false;
   bool useGlobalMuons = false;
   bool makeEnergyPlots = false;
   bool makeIsoPlots = false;
   if (d->Get("TrackerMuons")) useTrackerMuons = true;
   if (d->Get("GlobalMuons")) useGlobalMuons = true;
   if (d->FindObjectAny("hEnergyEMBarrel")) makeEnergyPlots = true;
   if (d->FindObjectAny("hIso03sumPt")) makeIsoPlots = true;
   if (! d->FindObjectAny("hCaloSegmentCompat")) make2DPlots = false;
   if (! d->FindObjectAny("hDTChamberDx_1_-2_1")) makeAllChamberPlots = false;

   TDirectoryFile* tmd = (TDirectoryFile*)d->Get("TrackerMuons");
   TDirectoryFile* gmd = (TDirectoryFile*)d->Get("GlobalMuons");

   // Fit trackerMuon pull distributions
   if (useTrackerMuons) {
      gStyle->SetOptFit(1);

      TList* list = tmd->GetListOfKeys();
      TIterator* iter = list->MakeIterator();
      TRegexp re("*Pull[dxy]*", kTRUE);

      TKey* key = 0;
      while(key = (TKey*)iter->Next()) {
         if (tmd->Get(key->GetName())->InheritsFrom(TH1F::Class()) &&
               (TString(key->GetName()).Index(re) >= 0))
         {
            TH1F* h = (TH1F*)tmd->Get(key->GetName()) ;
            h->Fit("gaus", "", "", -2., 2.);
         }
      }

      delete iter;
   }

   TPRegexp preg("(\\S+).root");
   TString output = ((TObjString*)preg.MatchS(TString(filename))->At(1))->GetString();
   output += ".ps";

   const int cwidth = 1000;
   const int cheight = 682;
   char types[2][8] = {"Tracker", "Global"};

   // trackerMuons == 0, globalMuons == 1
   for(unsigned int typeIdx = 0; typeIdx < 2; ++typeIdx) {
      if ((typeIdx == 0 && ! useTrackerMuons) || (typeIdx == 1 && ! useGlobalMuons)) continue;
      if (typeIdx == 0) d = tmd;
      if (typeIdx == 1) d = gmd;

      TCanvas* cTitlePage = new TCanvas("cTitlePage", "Title Page", cwidth, cheight);
      TText text(.4, .5, Form("%s Muons", types[typeIdx]));
      text.Draw();
      if (typeIdx == 0) cTitlePage->Print(Form("%s[", output.Data()));
      cTitlePage->Print(output.Data());

      TCanvas* cMuon = new TCanvas("cMuon", "Muon Information", cwidth, cheight);
      cMuon->Divide(3, 2);
      cMuon->cd(1);
      ((TH1F*)d->Get("hNumChambers"))->Draw();
      cMuon->cd(2);
      ((TH1F*)d->Get("hNumMatches"))->Draw();
      cMuon->cd(3);
      ((TH1F*)d->Get("hCaloCompat"))->Draw();
      cMuon->cd(4);
      ((TH1F*)d->Get("hSegmentCompat"))->Draw();
      if (make2DPlots) {
         cMuon->cd(5);
         ((TH2F*)d->Get("hCaloSegmentCompat"))->Draw();
      }
      cMuon->Print(output.Data());

      TCanvas* cMuonId = new TCanvas("cMuonId", "Muon Identification Algorithms", cwidth, cheight);
      cMuonId->Divide(3, 2);
      cMuonId->cd(1);
      ((TH1F*)d->Get("hGlobalMuonPromptTightBool"))->Draw();
      cMuonId->cd(2);
      ((TH1F*)d->Get("hTMLastStationLooseBool"))->Draw();
      cMuonId->cd(3);
      ((TH1F*)d->Get("hTMLastStationTightBool"))->Draw();
      cMuonId->cd(4);
      ((TH1F*)d->Get("hTM2DCompatibilityLooseBool"))->Draw();
      cMuonId->cd(5);
      ((TH1F*)d->Get("hTM2DCompatibilityTightBool"))->Draw();
      cMuonId->Print(output.Data());

      if (makeEnergyPlots) {
         TCanvas* cEnergy = new TCanvas("cEnergy", "Energy Depositions", cwidth, cheight);
         cEnergy->Divide(3, 2);
         cEnergy->cd(1);
         ((TH1F*)d->Get("hEnergyEMBarrel"))->Draw();
         cEnergy->cd(2);
         ((TH1F*)d->Get("hEnergyHABarrel"))->Draw();
         cEnergy->cd(3);
         ((TH1F*)d->Get("hEnergyHO"))->Draw();
         cEnergy->cd(4);
         ((TH1F*)d->Get("hEnergyEMEndcap"))->Draw();
         cEnergy->cd(5);
         ((TH1F*)d->Get("hEnergyHAEndcap"))->Draw();
         cEnergy->Print(output.Data());
      }

      if (makeIsoPlots) {
         TCanvas* cIso03 = new TCanvas("cIso03", "Muon Isolation", cwidth, cheight);
         cIso03->Divide(3, 2);
         cIso03->cd(1);
         ((TH1F*)d->Get("hIso03sumPt"))->Draw();
         cIso03->cd(2);
         ((TH1F*)d->Get("hIso03emEt"))->Draw();
         cIso03->cd(3);
         ((TH1F*)d->Get("hIso03hadEt"))->Draw();
         cIso03->cd(4);
         ((TH1F*)d->Get("hIso03hoEt"))->Draw();
         cIso03->cd(5);
         ((TH1F*)d->Get("hIso03nTracks"))->Draw();
         cIso03->cd(6);
         ((TH1F*)d->Get("hIso03nJets"))->Draw();
         cIso03->Print(output.Data());

         TCanvas* cIso05 = new TCanvas("cIso05", "Muon Isolation", cwidth, cheight);
         cIso05->Divide(3, 2);
         cIso05->cd(1);
         ((TH1F*)d->Get("hIso05sumPt"))->Draw();
         cIso05->cd(2);
         ((TH1F*)d->Get("hIso05emEt"))->Draw();
         cIso05->cd(3);
         ((TH1F*)d->Get("hIso05hadEt"))->Draw();
         cIso05->cd(4);
         ((TH1F*)d->Get("hIso05hoEt"))->Draw();
         cIso05->cd(5);
         ((TH1F*)d->Get("hIso05nTracks"))->Draw();
         cIso05->cd(6);
         ((TH1F*)d->Get("hIso05nJets"))->Draw();
         cIso05->Print(output.Data());
      }

      if (typeIdx == 0 && useTrackerMuons) {
         TCanvas* cDTNumSegments = new TCanvas("cDTNumSegments", "DT Number of Segments", cwidth, cheight);
         cDTNumSegments->Divide(2, 2);
         for(int i = 0; i < 4; ++i) {
            cDTNumSegments->cd(i+1);
            ((TH1F*)d->Get(Form("hDT%iNumSegments", i+1)))->Draw();
         }
         cDTNumSegments->Print(output.Data());

         TCanvas* cDTDx = new TCanvas("cDTDx", "DT Delta X, Pull X, Pull X w/ Propagation Error Only", cwidth, cheight);
         cDTDx->Divide(4, 3);
         for(int i = 0; i < 12; ++i) {
            cDTDx->cd(i+1);
            if (i < 4)
               ((TH1F*)d->Get(Form("hDT%iDx", i+1)))->Draw();
            if (i > 3 && i < 8)
               ((TH1F*)d->Get(Form("hDT%iPullx", i-3)))->Draw();
            if (i > 7)
               ((TH1F*)d->Get(Form("hDT%iPullxPropErr", i-7)))->Draw();
         }
         cDTDx->Print(output.Data());

         TCanvas* cDTDy = new TCanvas("cDTDy", "DT Delta Y, Pull Y, Pull Y w/ Propagation Error Only", cwidth, cheight);
         cDTDy->Divide(3, 3);
         for(int i = 0; i < 9; ++i) {
            cDTDy->cd(i+1);
            if (i < 3)
               ((TH1F*)d->Get(Form("hDT%iDy", i+1)))->Draw();
            if (i > 2 && i < 6)
               ((TH1F*)d->Get(Form("hDT%iPully", i-2)))->Draw();
            if (i > 5)
               ((TH1F*)d->Get(Form("hDT%iPullyPropErr", i-5)))->Draw();
         }
         cDTDy->Print(output.Data());

         TCanvas* cDTDdXdZ = new TCanvas("cDTDdXdZ", "DT Delta DxDz, Pull DxDz, Pull DxDz w/ Propagation Error Only", cwidth, cheight);
         cDTDdXdZ->Divide(4, 3);
         for(int i = 0; i < 12; ++i) {
            cDTDdXdZ->cd(i+1);
            if (i < 4)
               ((TH1F*)d->Get(Form("hDT%iDdXdZ", i+1)))->Draw();
            if (i > 3 && i < 8)
               ((TH1F*)d->Get(Form("hDT%iPulldXdZ", i-3)))->Draw();
            if (i > 7)
               ((TH1F*)d->Get(Form("hDT%iPulldXdZPropErr", i-7)))->Draw();
         }
         cDTDdXdZ->Print(output.Data());

         TCanvas* cDTDdYdZ = new TCanvas("cDTDdYdZ", "DT Delta DyDz, Pull DyDz, Pull DyDz w/ Propagation Error Only", cwidth, cheight);
         cDTDdYdZ->Divide(3, 3);
         for(int i = 0; i < 9; ++i) {
            cDTDdYdZ->cd(i+1);
            if (i < 3)
               ((TH1F*)d->Get(Form("hDT%iDdYdZ", i+1)))->Draw();
            if (i > 2 && i < 6)
               ((TH1F*)d->Get(Form("hDT%iPulldYdZ", i-2)))->Draw();
            if (i > 5)
               ((TH1F*)d->Get(Form("hDT%iPulldYdZPropErr", i-5)))->Draw();
         }
         cDTDdYdZ->Print(output.Data());

         TCanvas* cDTDist = new TCanvas("cDTDist", "DT Track Distance to Nearest Chamber Edge", cwidth, cheight);
         cDTDist->Divide(4, 4);
         for(int i = 0; i < 16; ++i) {
            cDTDist->cd(i+1);
            if (i < 4)
               ((TH1F*)d->Get(Form("hDT%iDistWithSegment", i+1)))->Draw();
            if (i > 3 && i < 8)
               ((TH1F*)d->Get(Form("hDT%iDistWithNoSegment", i-3)))->Draw();
            if (i > 7 && i < 12)
               ((TH1F*)d->Get(Form("hDT%iPullDistWithSegment", i-7)))->Draw();
            if (i > 11)
               ((TH1F*)d->Get(Form("hDT%iPullDistWithNoSegment", i-11)))->Draw();
         }
         cDTDist->Print(output.Data());

         TCanvas* cCSCNumSegments = new TCanvas("cCSCNumSegments", "CSC Number of Segments", cwidth, cheight);
         cCSCNumSegments->Divide(2, 2);
         for(int i = 0; i < 4; ++i) {
            cCSCNumSegments->cd(i+1);
            ((TH1F*)d->Get(Form("hCSC%iNumSegments", i+1)))->Draw();
         }
         cCSCNumSegments->Print(output.Data());

         TCanvas* cCSCDx = new TCanvas("cCSCDx", "CSC Delta X, Pull X, Pull X w/ Propagation Error Only", cwidth, cheight);
         cCSCDx->Divide(4, 3);
         for(int i = 0; i < 12; ++i) {
            cCSCDx->cd(i+1);
            if (i < 4)
               ((TH1F*)d->Get(Form("hCSC%iDx", i+1)))->Draw();
            if (i > 3 && i < 8)
               ((TH1F*)d->Get(Form("hCSC%iPullx", i-3)))->Draw();
            if (i > 7)
               ((TH1F*)d->Get(Form("hCSC%iPullxPropErr", i-7)))->Draw();
         }
         cCSCDx->Print(output.Data());

         TCanvas* cCSCDy = new TCanvas("cCSCDy", "CSC Delta Y, Pull Y, Pull Y w/ Propagation Error Only", cwidth, cheight);
         cCSCDy->Divide(4, 3);
         for(int i = 0; i < 12; ++i) {
            cCSCDy->cd(i+1);
            if (i < 4)
               ((TH1F*)d->Get(Form("hCSC%iDy", i+1)))->Draw();
            if (i > 3 && i < 8)
               ((TH1F*)d->Get(Form("hCSC%iPully", i-3)))->Draw();
            if (i > 7)
               ((TH1F*)d->Get(Form("hCSC%iPullyPropErr", i-7)))->Draw();
         }
         cCSCDy->Print(output.Data());

         TCanvas* cCSCDdXdZ = new TCanvas("cCSCDdXdZ", "CSC Delta DxDz, Pull DxDz, Pull DxDz w/ Propagation Error Only", cwidth, cheight);
         cCSCDdXdZ->Divide(4, 3);
         for(int i = 0; i < 12; ++i) {
            cCSCDdXdZ->cd(i+1);
            if (i < 4)
               ((TH1F*)d->Get(Form("hCSC%iDdXdZ", i+1)))->Draw();
            if (i > 3 && i < 8)
               ((TH1F*)d->Get(Form("hCSC%iPulldXdZ", i-3)))->Draw();
            if (i > 7)
               ((TH1F*)d->Get(Form("hCSC%iPulldXdZPropErr", i-7)))->Draw();
         }
         cCSCDdXdZ->Print(output.Data());

         TCanvas* cCSCDdYdZ = new TCanvas("cCSCDdYdZ", "CSC Delta DyDz, Pull DyDz, Pull DyDz w/ Propagation Error Only", cwidth, cheight);
         cCSCDdYdZ->Divide(4, 3);
         for(int i = 0; i < 12; ++i) {
            cCSCDdYdZ->cd(i+1);
            if (i < 4)
               ((TH1F*)d->Get(Form("hCSC%iDdYdZ", i+1)))->Draw();
            if (i > 3 && i < 8)
               ((TH1F*)d->Get(Form("hCSC%iPulldYdZ", i-3)))->Draw();
            if (i > 7)
               ((TH1F*)d->Get(Form("hCSC%iPulldYdZPropErr", i-7)))->Draw();
         }
         cCSCDdYdZ->Print(output.Data());

         TCanvas* cCSCDist = new TCanvas("cCSCDist", "CSC Track Distance to Nearest Chamber Edge", cwidth, cheight);
         cCSCDist->Divide(4, 4);
         for(int i = 0; i < 16; ++i) {
            cCSCDist->cd(i+1);
            if (i < 4)
               ((TH1F*)d->Get(Form("hCSC%iDistWithSegment", i+1)))->Draw();
            if (i > 3 && i < 8)
               ((TH1F*)d->Get(Form("hCSC%iDistWithNoSegment", i-3)))->Draw();
            if (i > 7 && i < 12)
               ((TH1F*)d->Get(Form("hCSC%iPullDistWithSegment", i-7)))->Draw();
            if (i > 11)
               ((TH1F*)d->Get(Form("hCSC%iPullDistWithNoSegment", i-11)))->Draw();
         }
         cCSCDist->Print(output.Data());

         TCanvas* cSegmentAssociation = new TCanvas("cSegmentAssociation", "Segment Association", cwidth, cheight);
         cSegmentAssociation->Divide(3, 2);
         cSegmentAssociation->cd(1);
         ((TH1F*)d->Get("hSegmentIsAssociatedBool"))->Draw();
         if (make2DPlots) {
            cSegmentAssociation->cd(2);
            ((TH2F*)d->Get("hSegmentIsAssociatedRZ"))->Draw();
            cSegmentAssociation->cd(3);
            ((TH2F*)d->Get("hSegmentIsAssociatedXY"))->Draw();
            cSegmentAssociation->cd(5);
            ((TH2F*)d->Get("hSegmentIsNotAssociatedRZ"))->Draw();
            cSegmentAssociation->cd(6);
            ((TH2F*)d->Get("hSegmentIsNotAssociatedXY"))->Draw();
            cSegmentAssociation->Print(output.Data());

            TCanvas* cSegmentBestDrAssociation = new TCanvas("cSegmentBestDrAssociation", "Segment Best in Station by #deltaR Association", cwidth, cheight);
            cSegmentBestDrAssociation->Divide(3, 2);
            cSegmentBestDrAssociation->cd(2);
            ((TH2F*)d->Get("hSegmentIsBestDrAssociatedRZ"))->Draw();
            cSegmentBestDrAssociation->cd(3);
            ((TH2F*)d->Get("hSegmentIsBestDrAssociatedXY"))->Draw();
            cSegmentBestDrAssociation->cd(5);
            ((TH2F*)d->Get("hSegmentIsBestDrNotAssociatedRZ"))->Draw();
            cSegmentBestDrAssociation->cd(6);
            ((TH2F*)d->Get("hSegmentIsBestDrNotAssociatedXY"))->Draw();
            cSegmentBestDrAssociation->Print(output.Data());
         }

         if (! makeAllChamberPlots) continue;

         char name[100], title[200];

         // by chamber
         for(int station = 0; station < 4; ++station) {
            // DT wheels: -2 -> 2
            for(int wheel = 0; wheel < 5; ++wheel) {
               sprintf(name, "cDTChamberDx_%i_%i", station+1, wheel-2);
               sprintf(title, "DT Chamber Delta X: Station %i Wheel %i", station+1, wheel-2);
               TCanvas* cDTChamberDx = new TCanvas(name, title, cwidth, cheight);
               cDTChamberDx->Divide(4, 4);

               TCanvas* cDTChamberDy = 0;
               if (station < 3) {
                  sprintf(name, "cDTChamberDy_%i_%i", station+1, wheel-2);
                  sprintf(title, "DT Chamber Delta Y: Station %i Wheel %i", station+1, wheel-2);
                  cDTChamberDy = new TCanvas(name, title, cwidth, cheight);
                  cDTChamberDy->Divide(4, 4);
               }

               sprintf(name, "cDTChamberEdgeXWithSegment_%i_%i", station+1, wheel-2);
               sprintf(title, "DT Chamber Edge X When There Is A Segment: Station %i Wheel %i", station+1, wheel-2);
               TCanvas* cDTChamberEdgeXWithSegment = new TCanvas(name, title, cwidth, cheight);
               cDTChamberEdgeXWithSegment->Divide(4, 4);

               sprintf(name, "cDTChamberEdgeXWithNoSegment_%i_%i", station+1, wheel-2);
               sprintf(title, "DT Chamber Edge X When There Is No Segment: Station %i Wheel %i", station+1, wheel-2);
               TCanvas* cDTChamberEdgeXWithNoSegment = new TCanvas(name, title, cwidth, cheight);
               cDTChamberEdgeXWithNoSegment->Divide(4, 4);

               sprintf(name, "cDTChamberEdgeYWithSegment_%i_%i", station+1, wheel-2);
               sprintf(title, "DT Chamber Edge Y When There Is A Segment: Station %i Wheel %i", station+1, wheel-2);
               TCanvas* cDTChamberEdgeYWithSegment = new TCanvas(name, title, cwidth, cheight);
               cDTChamberEdgeYWithSegment->Divide(4, 4);

               sprintf(name, "cDTChamberEdgeYWithNoSegment_%i_%i", station+1, wheel-2);
               sprintf(title, "DT Chamber Edge Y When There Is No Segment: Station %i Wheel %i", station+1, wheel-2);
               TCanvas* cDTChamberEdgeYWithNoSegment = new TCanvas(name, title, cwidth, cheight);
               cDTChamberEdgeYWithNoSegment->Divide(4, 4);

               // DT sectors: 1 -> 14
               for(int sector = 0; sector < 14; ++sector) {
                  cDTChamberDx->cd(sector+1);
                  ((TH1F*)d->Get(Form("hDTChamberDx_%i_%i_%i", station+1, wheel-2, sector+1)))->Draw();
                  if (station < 3) {
                     cDTChamberDy->cd(sector+1);
                     ((TH1F*)d->Get(Form("hDTChamberDy_%i_%i_%i", station+1, wheel-2, sector+1)))->Draw();
                  }
                  cDTChamberEdgeXWithSegment->cd(sector+1);
                  ((TH1F*)d->Get(Form("hDTChamberEdgeXWithSegment_%i_%i_%i", station+1, wheel-2, sector+1)))->Draw();
                  cDTChamberEdgeXWithNoSegment->cd(sector+1);
                  ((TH1F*)d->Get(Form("hDTChamberEdgeXWithNoSegment_%i_%i_%i", station+1, wheel-2, sector+1)))->Draw();
                  cDTChamberEdgeYWithSegment->cd(sector+1);
                  ((TH1F*)d->Get(Form("hDTChamberEdgeYWithSegment_%i_%i_%i", station+1, wheel-2, sector+1)))->Draw();
                  cDTChamberEdgeYWithNoSegment->cd(sector+1);
                  ((TH1F*)d->Get(Form("hDTChamberEdgeYWithNoSegment_%i_%i_%i", station+1, wheel-2, sector+1)))->Draw();
               }// sector
               cDTChamberDx->Print(output.Data());
               if (station < 3)
                  cDTChamberDy->Print(output.Data());
               cDTChamberEdgeXWithSegment->Print(output.Data());
               cDTChamberEdgeXWithNoSegment->Print(output.Data());
               cDTChamberEdgeYWithSegment->Print(output.Data());
               cDTChamberEdgeYWithNoSegment->Print(output.Data());
            }// wheel
         }// station

         for(int station = 0; station < 4; ++station) {
            // CSC endcaps: 1 -> 2
            for(int endcap = 0; endcap < 2; ++endcap) {
               // CSC rings: 1 -> 4
               for(int ring = 0; ring < 4; ++ring) {
                  sprintf(name, "cCSCChamberDx_%i_%i_%i", endcap+1, station+1, ring+1);
                  sprintf(title, "CSC Chamber Delta X: Endcap %i Station %i Ring %i", endcap+1, station+1, ring+1);
                  TCanvas* cCSCChamberDx = new TCanvas(name, title, cwidth, cheight);
                  cCSCChamberDx->Divide(6, 6);

                  sprintf(name, "cCSCChamberDy_%i_%i_%i", endcap+1, station+1, ring+1);
                  sprintf(title, "CSC Chamber Delta Y: Endcap %i Station %i Ring %i", endcap+1, station+1, ring+1);
                  TCanvas* cCSCChamberDy = new TCanvas(name, title, cwidth, cheight);
                  cCSCChamberDy->Divide(6, 6);

                  sprintf(name, "cCSCChamberEdgeXWithSegment_%i_%i_%i", endcap+1, station+1, ring+1);
                  sprintf(title, "CSC Chamber Edge X When There Is A Segment: Endcap %i Station %i Ring %i", endcap+1, station+1, ring+1);
                  TCanvas* cCSCChamberEdgeXWithSegment = new TCanvas(name, title, cwidth, cheight);
                  cCSCChamberEdgeXWithSegment->Divide(6, 6);

                  sprintf(name, "cCSCChamberEdgeXWithNoSegment_%i_%i_%i", endcap+1, station+1, ring+1);
                  sprintf(title, "CSC Chamber Edge X When There Is No Segment: Endcap %i Station %i Ring %i", endcap+1, station+1, ring+1);
                  TCanvas* cCSCChamberEdgeXWithNoSegment = new TCanvas(name, title, cwidth, cheight);
                  cCSCChamberEdgeXWithNoSegment->Divide(6, 6);

                  sprintf(name, "cCSCChamberEdgeYWithSegment_%i_%i_%i", endcap+1, station+1, ring+1);
                  sprintf(title, "CSC Chamber Edge Y When There Is A Segment: Endcap %i Station %i Ring %i", endcap+1, station+1, ring+1);
                  TCanvas* cCSCChamberEdgeYWithSegment = new TCanvas(name, title, cwidth, cheight);
                  cCSCChamberEdgeYWithSegment->Divide(6, 6);

                  sprintf(name, "cCSCChamberEdgeYWithNoSegment_%i_%i_%i", endcap+1, station+1, ring+1);
                  sprintf(title, "CSC Chamber Edge Y When There Is No Segment: Endcap %i Station %i Ring %i", endcap+1, station+1, ring+1);
                  TCanvas* cCSCChamberEdgeYWithNoSegment = new TCanvas(name, title, cwidth, cheight);
                  cCSCChamberEdgeYWithNoSegment->Divide(6, 6);

                  // CSC chambers: 1 -> 36
                  for(int chamber = 0; chamber < 36; ++chamber) {
                     cCSCChamberDx->cd(chamber+1);
                     ((TH1F*)d->Get(Form("hCSCChamberDx_%i_%i_%i_%i", endcap+1, station+1, ring+1, chamber+1)))->Draw();
                     cCSCChamberDy->cd(chamber+1);
                     ((TH1F*)d->Get(Form("hCSCChamberDy_%i_%i_%i_%i", endcap+1, station+1, ring+1, chamber+1)))->Draw();
                     cCSCChamberEdgeXWithSegment->cd(chamber+1);
                     ((TH1F*)d->Get(Form("hCSCChamberEdgeXWithSegment_%i_%i_%i_%i", endcap+1, station+1, ring+1, chamber+1)))->Draw();
                     cCSCChamberEdgeXWithNoSegment->cd(chamber+1);
                     ((TH1F*)d->Get(Form("hCSCChamberEdgeXWithNoSegment_%i_%i_%i_%i", endcap+1, station+1, ring+1, chamber+1)))->Draw();
                     cCSCChamberEdgeYWithSegment->cd(chamber+1);
                     ((TH1F*)d->Get(Form("hCSCChamberEdgeYWithSegment_%i_%i_%i_%i", endcap+1, station+1, ring+1, chamber+1)))->Draw();
                     cCSCChamberEdgeYWithNoSegment->cd(chamber+1);
                     ((TH1F*)d->Get(Form("hCSCChamberEdgeYWithNoSegment_%i_%i_%i_%i", endcap+1, station+1, ring+1, chamber+1)))->Draw();
                  }
                  cCSCChamberDx->Print(output.Data());
                  cCSCChamberDy->Print(output.Data());
                  cCSCChamberEdgeXWithSegment->Print(output.Data());
                  cCSCChamberEdgeXWithNoSegment->Print(output.Data());
                  cCSCChamberEdgeYWithSegment->Print(output.Data());
                  cCSCChamberEdgeYWithNoSegment->Print(output.Data());
               }// ring
            }// endcap
         }// station
      }

      if (typeIdx == 1) cTitlePage->Print(Form("%s]", output.Data()));
   }

   f->Close();
}

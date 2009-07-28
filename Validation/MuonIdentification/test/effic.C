#include "TCanvas.h"
#include "TDirectoryFile.h"
#include "TFile.h"
#include "TH1F.h"
#include "TLegend.h"
#include "TObjString.h"
#include "TString.h"
#include "TStyle.h"
#include <iostream>

void effic(char* filename1, char* filename2 = 0) {
    TFile* f1 = TFile::Open(filename1);
    if (! f1) {
        std::cout << "Error: unable to open " << filename1 << std::endl;
        return;
    }
    TDirectoryFile* d1 = (TDirectoryFile*)f1->FindObjectAny("MuonIdVal");
    if (! d1)
        if (f1->cd("/DQMData/Run 1/MuonIdentificationV/Run summary"))
            d1 = (TDirectoryFile*)gDirectory;
    if (! d1) {
        std::cout << "Error: MuonIdVal/MuonIdentificationV not found in " << filename1 << std::endl;
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
        if (! d2)
            if (f2->cd("/DQMData/Run 1/MuonIdentificationV/Run summary"))
                d2 = (TDirectoryFile*)gDirectory;
        if (! d2) {
            std::cout << "Error: MuonIdVal/MuonIdentificationV not found in " << filename2 << std::endl;
            return;
        }
    }

    gStyle->SetOptStat(0);

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

    TCanvas* c = new TCanvas("c");
    c->Draw();
    TLegend* leg = 0;

    // Access histograms in three different locations:
    //   d1 -> MuonIdVal
    // tmd1 -> MuonIdVal/TrackerMuons
    // gmd1 -> MuonIdVal/GlobalMuons
    char pfx[2][4] = {"tm_", "gm_"};
    char det[2][4] = {"DT", "CSC"};

    TH1F* eff1 = 0;
    TH1F* num1 = 0;
    TH1F* den1 = 0;
    TH1F* eff2 = 0;
    TH1F* num2 = 0;
    TH1F* den2 = 0;

    for(unsigned int i = 0; i < 2; i++) {
        if (i == 0 && tmd1) { d1 = tmd1; d2 = tmd2; }
        else if (i == 1 && gmd1) { d1 = gmd1; d2 = gmd2; }
        else continue;

        for(unsigned int j = 0; j < 2; j++) { // detector
            eff1 = (TH1F*)d1->Get(Form("h%s1DistWithSegment", det[j]))->Clone(Form("%sh%sDistEffic1", pfx[i], det[j]));
            eff1->Reset();
            num1 = (TH1F*)eff1->Clone(Form("%sh%sDistNum1", pfx[i], det[j]));
            den1 = (TH1F*)eff1->Clone(Form("%sh%sDistDen1", pfx[i], det[j]));
            eff1->Sumw2();
            num1->Sumw2();
            den1->Sumw2();

            if (d2) {
                eff2 = (TH1F*)eff1->Clone(Form("%sh%sDistEffic2", pfx[i], det[j]));
                num2 = (TH1F*)eff1->Clone(Form("%sh%sDistNum2", pfx[i], det[j]));
                den2 = (TH1F*)eff1->Clone(Form("%sh%sDistDen2", pfx[i], det[j]));
                if (eff2 && num2 && den2) {
                    eff2->Sumw2();
                    num2->Sumw2();
                    den2->Sumw2();
                } else
                    eff2 = 0;
            }

            for (int k = 1; k <= 4; k++) { // station
                num1->Add((TH1F*)d1->Get(Form("h%s%iDistWithSegment", det[j], k)));
                den1->Add((TH1F*)d1->Get(Form("h%s%iDistWithSegment", det[j], k)));
                den1->Add((TH1F*)d1->Get(Form("h%s%iDistWithNoSegment", det[j], k)));
                if (eff2) {
                    num2->Add((TH1F*)d2->Get(Form("h%s%iDistWithSegment", det[j], k)));
                    den2->Add((TH1F*)d2->Get(Form("h%s%iDistWithSegment", det[j], k)));
                    den2->Add((TH1F*)d2->Get(Form("h%s%iDistWithNoSegment", det[j], k)));
                }
            }

            eff1->SetTitle("");
            eff1->GetXaxis()->SetTitle("distance to nearest chamber edge [cm]");
            eff1->GetYaxis()->SetTitle("efficiency");
            eff1->GetYaxis()->SetTitleOffset(1.1);
            eff1->Divide(num1, den1, 1., 1., "B");
            eff1->SetLineColor(1);
            eff1->SetLineStyle(1);
            eff1->Draw("histe");
            eff1->SetMaximum(1.1);
            if (eff2) {
                eff2->Divide(num2, den2, 1., 1., "B");
                eff2->SetLineColor(2);
                eff2->SetLineStyle(2);
                eff2->Draw("histesame");
                double ks = eff1->KolmogorovTest(eff2);
                eff1->SetTitle(Form("KS: %f", ks));
            }

            leg = 0;
            leg = new TLegend(0., 0., .22, .06);
            leg->SetFillColor(0);
            leg->AddEntry(eff1, version1.Data(), "l");
            if (eff2)
               leg->AddEntry(eff2, version2.Data(), "l");
            leg->Draw();
            c->Update();
            c->Print(Form("%sh%sDistEffic.png", pfx[i], det[j]));
        } // for unsigned int j
    } // for unsigned int i
} // void effic

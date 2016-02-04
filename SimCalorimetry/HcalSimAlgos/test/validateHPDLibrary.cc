#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseData.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseReader.h"

#include <iostream>
#include <string>
#include <sstream>
#include "TClass.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TFile.h"
#include "TAxis.h"

std::string itos(int i);

int main(int argc, const char* argv[]) {

    int runno = 0;
    if(argc != 2){
        std::cout << "You forgot to enter run number" << std::endl;
    }else{
        runno = atoi(argv[1]);
    }
    char f[128];
    sprintf(f,"validate_%i.root",runno);
    TFile *file = new TFile(f, "RECREATE");
    TH1F *h1 = new TH1F("h1", "Ion feedback first peak rate", 144, -72, 72);
    TH1F *h2 = new TH1F("h2", "Ion feedback second peak rate", 144, -72, 72);
    TH1F *h3 = new TH1F("h3", "Thermal/Field Electron Emission rate", 144, -72, 72);
    TH1F *h4 = new TH1F("h4", "Discharge rate", 144, -72, 72);
    TH1F *h5 = new TH1F("h5", "Number of noisy events per i#phi", 144, -72, 72);
    TH1F *h6 = new TH1F("h6", "Noise Distribution (fC)", 100, 0., 10000.);
    std::string ff = "/uscmst1b_scratch/lpc1/lpcphys/tyetkin/NoiseLibrary/hpdNoiseLibrary_Run_" + itos(runno)+ ".root";
    HPDNoiseReader reader(ff);
    int NHPD = 0;
    for (int i = -1; i < 2; ++i) {
        if (i == 0)
            continue;
        for (int iphi = 1; iphi <= 72; ++iphi) {
            std::string name;
            if (i == -1) {
                name = "ZMinusHPD" + itos(iphi);
            }else if (i == 1) {
                name = "ZPlusHPD" + itos(iphi);
            }
	    TAxis *xaxis = h1->GetXaxis();
	    Int_t bin  = xaxis->FindBin(i*iphi)-1;
	    std::cout << i*iphi << " " << bin << std::endl; 
            HPDNoiseReader::Handle hpdObj = reader.getHandle(name);
            if (reader.valid(hpdObj)) {
		h1->SetBinContent(bin, reader.ionFeedbackFirstPeakRate(hpdObj));
                h2->SetBinContent(bin, reader.ionFeedbackSecondPeakRate(hpdObj));
                h3->SetBinContent(bin, reader.emissionRate(hpdObj));
                h4->SetBinContent(bin, reader.dischargeRate(hpdObj));
                h5->SetBinContent(bin, reader.totalEntries(hpdObj));
                ++NHPD;
		for (int evt = 0; evt < int (reader.totalEntries(hpdObj)); ++evt) {
                    HPDNoiseData *data;
                    reader.getEntry(hpdObj, evt, &data);   // each HPD may have more than 1 noisy channel
                    for (unsigned int idata = 0; idata < data->size(); ++idata) {
                        // HcalDetId id = data->getDataFrame(idata).id();
                        const float *noise = data->getDataFrame(idata).getFrame();
                        float charge = 0;

                        for (int ic = 0; ic < 10; ic++)
                            charge += noise[ic];
                        h6->Fill(charge);
                    }
                }
            }
        }
    }
    std::cout << "number of hpds " << NHPD << std::endl; 
    TCanvas *c1 = new TCanvas("c1", "c1", 85, 70, 800, 600);
    h1->Draw();
    c1->SaveAs("h1.gif");
    TCanvas *c2 = new TCanvas("c2", "c21", 85, 70, 800, 600);
    h2->Draw(); 
    c2->SaveAs("h2.gif");
    TCanvas *c3 = new TCanvas("c3", "c3", 85, 70, 800, 600);
    h3->Draw();
    c3->SaveAs("h3.gif");
    TCanvas *c4 = new TCanvas("c4", "c4", 85, 70, 800, 600);
    h4->Draw();
    c4->SaveAs("h4.gif");
    TCanvas *c5 = new TCanvas("c5", "c5", 85, 70, 800, 600);
    h5->Draw();
    c5->SaveAs("h5.gif");
    TCanvas *c6 = new TCanvas("c6", "c6", 85, 70, 800, 600);
    c6->SetLogy();
    h6->Draw();
    c6->SaveAs("h6.gif");
    file->Write();
    file->Close();
    return 0;
}
std::string itos(int i) {
    std::stringstream s;

    s << i;
    return s.str();
}

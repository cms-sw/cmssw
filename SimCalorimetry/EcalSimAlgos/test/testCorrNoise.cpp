#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCorrelatedNoiseMatrix.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<iostream>
#include<iomanip>
#include<fstream>

#include "CLHEP/Matrix/SymMatrix.h"

#include "TROOT.h"
#include "TStyle.h"
#include "TH1F.h"
#include "TCanvas.h"

int main() {

  edm::MessageDrop::instance()->debugEnabled = false;

  const int readoutFrameSize = 10;

  EcalCorrelatedNoiseMatrix theNoiseMatrix(readoutFrameSize);
  HepSymMatrix thisMatrix(readoutFrameSize,1);
  theNoiseMatrix.getMatrix(thisMatrix);
  CorrelatedNoisifier theCorrNoise(thisMatrix);

  HepSymMatrix thisTrivialMatrix(readoutFrameSize,1);
  CorrelatedNoisifier theUncorrNoise(thisTrivialMatrix);

  std::cout << "Using the correlation matrix: " << thisMatrix << std::cout;

  std::cout << "\n And the unit matrix: " << thisTrivialMatrix << std::cout;

  EBDetId detId(1,1);

  TH1F* uncorr = new TH1F("uncorr","first 3 samples, uncorrelated distribution",200,-10.,10.);
  TH1F* corr = new TH1F("corr","first 3 samples, correlated distribution",200,-10.,10.);

  TH1F* uncorrPed = new TH1F("uncorrPed","first 3 samples, uncorrelated average",200,-10.,10.);
  TH1F* corrPed = new TH1F("corrPed","first 3 samples, correlated average",200,-10.,10.);

  TH1F* frame[readoutFrameSize];
  Char_t histo[200];
  for ( Int_t i = 0; i < readoutFrameSize; ++i) {
    sprintf (histo,"frame %02d",i) ;
    frame[i] = new TH1F(histo,histo,200,-10.,10.);
  }

  for ( int i = 0; i < 100000; ++i ) {
    CaloSamples noiseframe(detId, readoutFrameSize);
    theCorrNoise.noisify(noiseframe);
    CaloSamples flatframe(detId, readoutFrameSize);
    theUncorrNoise.noisify(flatframe);
    for ( int j = 0; j < 3; ++j ) {
      uncorr->Fill(flatframe[j]);
      corr->Fill(noiseframe[j]);
    }
    for ( int j = 0; j < readoutFrameSize; ++j ) {
      frame[j]->Fill(noiseframe[j]);
    }
    float thisUncorrPed = (flatframe[0]+flatframe[1]+flatframe[2])/3.;
    float thisCorrPed = (noiseframe[0]+noiseframe[1]+noiseframe[2])/3.;
    uncorrPed->Fill(thisUncorrPed);
    corrPed->Fill(thisCorrPed);
  }

  uncorr->Fit("gaus","VE");
  corr->Fit("gaus","VE");

  uncorrPed->Fit("gaus","VE");
  corrPed->Fit("gaus","VE");

  const int csize = 500;
  TCanvas * showNoise = new TCanvas("showNoise","showNoise",2*csize,2*csize);

  showNoise->Divide(2,2);
  
  gStyle->SetOptFit(1111);

  showNoise->cd(1);
  uncorr->Draw();
  showNoise->cd(2);
  corr->Draw();
  showNoise->cd(3);
  uncorrPed->Draw();
  showNoise->cd(4);
  corrPed->Draw();
  showNoise->SaveAs("EcalNoise.jpg");

  TCanvas * showNoiseFrame = new TCanvas("showNoiseFrame","showNoiseFrame",2*csize,2*csize);
  showNoiseFrame->Divide(2,5);
  for ( int i = 0; i < readoutFrameSize; ++i) {
    showNoiseFrame->cd(i+1);
    frame[i]->Fit("gaus","Q");
    frame[i]->Draw();
  }
  showNoiseFrame->SaveAs("EcalNoiseFrame.jpg");


  delete uncorr;
  delete corr;
  delete uncorrPed;
  delete corrPed;
  delete showNoise;
  delete showNoiseFrame;

  return 0;

}

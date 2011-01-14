#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCorrelatedNoiseMatrix.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Event.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/JamesRandom.h"
#include "SimDataFormats/RandomEngine/interface/RandomEngineState.h"
#include<iostream>
#include<iomanip>
#include<fstream>

#include "DataFormats/Math/interface/Error.h"

#include "TROOT.h"
#include "TStyle.h"
#include "TH1F.h"
#include "TCanvas.h"

class MyRandomNumberGenerator : public edm::RandomNumberGenerator 
{
   public:

      MyRandomNumberGenerator() : edm::RandomNumberGenerator(),
				  m_seed (123456789),
				  m_engine (new CLHEP::HepJamesRandom(m_seed)) {}
      virtual ~MyRandomNumberGenerator() {}
 
      virtual CLHEP::HepRandomEngine& getEngine() const { return *m_engine ; }
      virtual uint32_t mySeed() const { return m_seed; }
      virtual void preBeginLumi(edm::LuminosityBlock const& lumi) {}
      virtual void postEventRead(edm::Event const& event) {}

      virtual std::vector<RandomEngineState> const& getLumiCache() const {
	 return m_states ; }
      virtual std::vector<RandomEngineState> const& getEventCache() const {
	 return m_states ; }
      virtual void print() {}

   private:
      MyRandomNumberGenerator(const MyRandomNumberGenerator&); // stop default
      const MyRandomNumberGenerator& operator=(const MyRandomNumberGenerator&); // stop default

      long m_seed ;
      CLHEP::HepRandomEngine* m_engine ;
      std::vector<RandomEngineState > m_states ;
};


int main() 
{
   edm::MessageDrop::instance()->debugEnabled = false;

   std::auto_ptr<edm::RandomNumberGenerator> slcptr( new MyRandomNumberGenerator() ) ;

   boost::shared_ptr<edm::serviceregistry::ServiceWrapper<edm::RandomNumberGenerator > > slc ( new edm::serviceregistry::ServiceWrapper<edm::RandomNumberGenerator >( slcptr ) ) ; 
   edm::ServiceToken token = edm::ServiceRegistry::createContaining( slc ) ;
   edm::ServiceRegistry::Operate operate( token ) ; 

/*  std::vector<edm::ParameterSet> serviceConfigs;
  edm::ServiceToken token = edm::ServiceRegistry::createSet(serviceConfigs);
  edm::ServiceRegistry::Operate operate(token); 
*/

  const unsigned int readoutFrameSize = CaloSamples::MAXSAMPLES;

  EcalCorrMatrix thisMatrix;

  thisMatrix(0,0) = 1.;
  thisMatrix(0,1) = 0.67;
  thisMatrix(0,2) = 0.53;
  thisMatrix(0,3) = 0.44;
  thisMatrix(0,4) = 0.39;
  thisMatrix(0,5) = 0.36;
  thisMatrix(0,6) = 0.38;
  thisMatrix(0,7) = 0.35;
  thisMatrix(0,8) = 0.36;
  thisMatrix(0,9) = 0.32;
  
  thisMatrix(1,0) = 0.67;
  thisMatrix(1,1) = 1.;
  thisMatrix(1,2) = 0.67;
  thisMatrix(1,3) = 0.53;
  thisMatrix(1,4) = 0.44;
  thisMatrix(1,5) = 0.39;
  thisMatrix(1,6) = 0.36;
  thisMatrix(1,7) = 0.38;
  thisMatrix(1,8) = 0.35;
  thisMatrix(1,9) = 0.36;
  
  thisMatrix(2,0) = 0.53;
  thisMatrix(2,1) = 0.67;
  thisMatrix(2,2) = 1.;
  thisMatrix(2,3) = 0.67;
  thisMatrix(2,4) = 0.53;
  thisMatrix(2,5) = 0.44;
  thisMatrix(2,6) = 0.39;
  thisMatrix(2,7) = 0.36;
  thisMatrix(2,8) = 0.38;
  thisMatrix(2,9) = 0.35;
  
  thisMatrix(3,0) = 0.44;
  thisMatrix(3,1) = 0.53;
  thisMatrix(3,2) = 0.67;
  thisMatrix(3,3) = 1.;
  thisMatrix(3,4) = 0.67;
  thisMatrix(3,5) = 0.53;
  thisMatrix(3,6) = 0.44;
  thisMatrix(3,7) = 0.39;
  thisMatrix(3,8) = 0.36;
  thisMatrix(3,9) = 0.38;
  
  thisMatrix(4,0) = 0.39;
  thisMatrix(4,1) = 0.44;
  thisMatrix(4,2) = 0.53;
  thisMatrix(4,3) = 0.67;
  thisMatrix(4,4) = 1.;
  thisMatrix(4,5) = 0.67;
  thisMatrix(4,6) = 0.53;
  thisMatrix(4,7) = 0.44;
  thisMatrix(4,8) = 0.39;
  thisMatrix(4,9) = 0.36;
  
  thisMatrix(5,0) = 0.36;
  thisMatrix(5,1) = 0.39;
  thisMatrix(5,2) = 0.44;
  thisMatrix(5,3) = 0.53;
  thisMatrix(5,4) = 0.67;
  thisMatrix(5,5) = 1.;
  thisMatrix(5,6) = 0.67;
  thisMatrix(5,7) = 0.53;
  thisMatrix(5,8) = 0.44;
  thisMatrix(5,9) = 0.39;
  
  thisMatrix(6,0) = 0.38;
  thisMatrix(6,1) = 0.36;
  thisMatrix(6,2) = 0.39;
  thisMatrix(6,3) = 0.44;
  thisMatrix(6,4) = 0.53;
  thisMatrix(6,5) = 0.67;
  thisMatrix(6,6) = 1.;
  thisMatrix(6,7) = 0.67;
  thisMatrix(6,8) = 0.53;
  thisMatrix(6,9) = 0.44;
  
  thisMatrix(7,0) = 0.35;
  thisMatrix(7,1) = 0.38;
  thisMatrix(7,2) = 0.36;
  thisMatrix(7,3) = 0.39;
  thisMatrix(7,4) = 0.44;
  thisMatrix(7,5) = 0.53;
  thisMatrix(7,6) = 0.67;
  thisMatrix(7,7) = 1.;
  thisMatrix(7,8) = 0.67;
  thisMatrix(7,9) = 0.53;
  
  thisMatrix(8,0) = 0.36;
  thisMatrix(8,1) = 0.35;
  thisMatrix(8,2) = 0.38;
  thisMatrix(8,3) = 0.36;
  thisMatrix(8,4) = 0.39;
  thisMatrix(8,5) = 0.44;
  thisMatrix(8,6) = 0.53;
  thisMatrix(8,7) = 0.67;
  thisMatrix(8,8) = 1.;
  thisMatrix(8,9) = 0.67;
  
  thisMatrix(9,0) = 0.32;
  thisMatrix(9,1) = 0.36;
  thisMatrix(9,2) = 0.35;
  thisMatrix(9,3) = 0.38;
  thisMatrix(9,4) = 0.36;
  thisMatrix(9,5) = 0.39;
  thisMatrix(9,6) = 0.44;
  thisMatrix(9,7) = 0.53;
  thisMatrix(9,8) = 0.67;
  thisMatrix(9,9) = 1.;

  CorrelatedNoisifier<EcalCorrMatrix> theCorrNoise(thisMatrix);

  EcalCorrMatrix thisTrivialMatrix;
  for (unsigned int i = 0; i < readoutFrameSize; i++ )
    {
      thisTrivialMatrix(i,i) = 1.;
      for ( unsigned int j = i+1; j < readoutFrameSize; j++ )
        {
          thisTrivialMatrix(i,j) = 0.;
          thisTrivialMatrix(j,i) = 0.;
        }
    }
  CorrelatedNoisifier<EcalCorrMatrix> theUncorrNoise(thisTrivialMatrix);

  std::cout << "Using the correlation matrix: " << thisMatrix << std::endl;

  std::cout << "\n And the unit matrix: " << thisTrivialMatrix << std::endl;

  EBDetId detId(1,1);

  TH1F* uncorr = new TH1F("uncorr","first 3 samples, uncorrelated distribution",200,-10.,10.);
  TH1F* corr = new TH1F("corr","first 3 samples, correlated distribution",200,-10.,10.);

  TH1F* uncorrPed = new TH1F("uncorrPed","first 3 samples, uncorrelated average",200,-10.,10.);
  TH1F* corrPed = new TH1F("corrPed","first 3 samples, correlated average",200,-10.,10.);

  TH1F* frame[(int)readoutFrameSize];
  Char_t histo[200];
  for ( Int_t i = 0; i < (int)readoutFrameSize; ++i) {
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
    for ( int j = 0; j < (int)readoutFrameSize; ++j ) {
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
  for ( int i = 0; i < (int)readoutFrameSize; ++i) {
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

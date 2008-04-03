#ifndef EcalSimAlgos_ESFastTDigitizer_h
#define EcalSimAlgos_ESFastTDigitizer_h

/*  Based on CaloTDigitizer
    Turns hits into digis.  Assumes that 
    there's an ESElectronicsSimFast class with the
    interface analogToDigital(const CaloSamples &, Digi &);
*/
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoiseHitGenerator.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

#include "CLHEP/Random/RandGeneral.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandFlat.h"
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_sf_result.h>

#include "TH3F.h"
#include <vector>
#include <cstdlib>

using namespace edm;

class ESFastTDigitizer
{
 public:
  
  ESFastTDigitizer(CaloHitResponse * hitResponse, ESElectronicsSimFast * electronicsSim, bool addNoise, int numESdetId, double zsThreshold, std::string refFile)
    :  theHitResponse(hitResponse),
    theNoiseHitGenerator(0),              
    theElectronicsSim(electronicsSim),
    theDetIds(0),
    addNoise_(addNoise),
    numESdetId_(numESdetId),  
    zsThreshold_(zsThreshold),
    refFile_(refFile) {        
    
    // reference distributions
    if (addNoise_) readHistosFromFile () ;  
  }
  
  /// doesn't delete the pointers passed in 
  // ~ESFastTDigitizer() { delete refHistos_; }
  ~ESFastTDigitizer() { delete histoDistribution_; }
  
  /// taking reference histos
  void readHistosFromFile( ) {

    m_histofile = new ifstream (edm::FileInPath(refFile_).fullPath().c_str());
    if (m_histofile == 0){ 
      throw edm::Exception(edm::errors::InvalidReference,"NullPointer")
        << "Reference histos file not opened" ;
      return ;
    }
    
    // number of bins
    char buffer[200];
    int thisLine = 0;
    while( thisLine==0 ) {
      m_histofile->getline(buffer,400);
      if (!strstr(buffer,"#") && !(strspn(buffer," ") == strlen(buffer))){	
	float histoBin; 
	sscanf(buffer,"%f",&histoBin); 
	histoBin_ = (double)histoBin;
	thisLine++;
      }
    }
    int histoBin3 = (int)(histoBin_*histoBin_*histoBin_);
    refHistos_ = new double[histoBin3];    
    
    // all info
    int thisBin = -2;
    while( !(m_histofile->eof()) ){
      m_histofile->getline(buffer,400);
      if (!strstr(buffer,"#") && !(strspn(buffer," ") == strlen(buffer))){
	if(thisBin==-2){ float histoInf; sscanf(buffer,"%f",&histoInf); histoInf_ = (double)histoInf; }
	if(thisBin==-1){ float histoSup; sscanf(buffer,"%f",&histoSup); histoSup_ = (double)histoSup; }
	
	if (thisBin>=0){ 
	  float refBin; 
	  sscanf(buffer,"%f",&refBin);
	  refHistos_[thisBin] = (double)refBin;
	}
	thisBin++;
      }
    }

    // creating the reference distribution to extract random numbers
    edm::Service<edm::RandomNumberGenerator> rng;   
    if ( ! rng.isAvailable()) {
      throw cms::Exception("Configuration")
	<< "ESFastTDigitizer requires the RandomNumberGeneratorService\n"
	"which is not present in the configuration file.  You must add the service\n"
	"in the configuration file or remove the modules that require it.";
    }
    CLHEP::HepRandomEngine& engine = rng->getEngine();
    histoDistribution_ = new CLHEP::RandGeneral(engine, refHistos_, histoBin3, 0);

    m_histofile->close();
    delete m_histofile;
  }

  /// preparing the list of channels where the noise has to be generated
  void createNoisyList(std::vector<int> & abThreshCh) {
    
    edm::Service<edm::RandomNumberGenerator> rng;   
    if ( ! rng.isAvailable()) {
      throw cms::Exception("Configuration")
	<< "ESFastTDigitizer requires the RandomNumberGeneratorService\n"
	"which is not present in the configuration file.  You must add the service\n"
	"in the configuration file or remove the modules that require it.";
    }
    CLHEP::RandPoissonQ poissonDistribution_(rng->getEngine());
    CLHEP::RandFlat flatDistribution_(rng->getEngine());
    
    gsl_sf_result result;
    int status  = gsl_sf_erf_Q_e(zsThreshold_, &result);
    if (status != 0) std::cerr<<"ESFastTDigitizer::could not compute gaussian tail probability for the threshold chosen"<<std::endl;
    double probabilityLeft = result.val; 
    double meanNumberOfNoisyChannels = probabilityLeft * numESdetId_;
    int numberOfNoisyChannels = poissonDistribution_.fire(meanNumberOfNoisyChannels);
    abThreshCh.reserve(numberOfNoisyChannels);
    for (int i = 0; i < numberOfNoisyChannels; i++) {
      std::vector<int>::iterator theChannel;
      int theChannelNumber = 0;
      do {
        theChannelNumber = (int)flatDistribution_.fire(numESdetId_);
        theChannel = find(abThreshCh.begin(), abThreshCh.end(), theChannelNumber);
      }
      while ( theChannel!=abThreshCh.end() );

      abThreshCh.push_back(theChannelNumber);
    }
  }
  
  
  /// tell the digitizer which cells exist
  void setDetIds(const std::vector<DetId> & detIds) {theDetIds = detIds;}
  
  void setNoiseHitGenerator(CaloVNoiseHitGenerator * generator) {
    theNoiseHitGenerator = generator;
  }
  
  /// turns hits into digis
  void run(MixCollection<PCaloHit> & input, ESDigiCollection & output) {

    assert(theDetIds.size() != 0);
    
    theHitResponse->run(input);

    if(theNoiseHitGenerator != 0) addNoiseHits();  

    theElectronicsSim->newEvent();

    // reserve space for how many digis we expect 
    int nDigisExpected = addNoise_ ? theDetIds.size() : theHitResponse->nSignals();
    output.reserve(nDigisExpected);

    // random generation of channel above threshold
    std::vector<int> abThreshCh;                                            
    if (addNoise_) createNoisyList(abThreshCh);

    // make a raw digi for evey cell where we have noise
    int idxDetId=0;
    for(std::vector<DetId>::const_iterator idItr = theDetIds.begin();
        idItr != theDetIds.end(); ++idItr,++idxDetId) {

      bool needToDeleteSignal = false;
      CaloSamples * analogSignal = theHitResponse->findSignal(*idItr);
	
      // signal or just noise?
      bool wasEmpty = false;
      
      if (!analogSignal){   	// no signal here
	wasEmpty = true;
	if (!addNoise_) continue; 
	else { 
	  std::vector<int>::iterator thisChannel;
	  thisChannel = find( abThreshCh.begin(), abThreshCh.end(), idxDetId);            
	  if( thisChannel != abThreshCh.end() ) {
	    analogSignal = new CaloSamples(theHitResponse->makeBlankSignal(*idItr));
	    needToDeleteSignal = true;
	  }
	}
      }
      
      if (analogSignal != 0){ 
	// either we have a signal or we need to generate noise samples 	  
	  
	ESDataFrame digi(*idItr);
	theElectronicsSim->analogToDigital(*analogSignal , digi, wasEmpty, histoDistribution_, histoInf_, histoSup_, histoBin_);	
	output.push_back(digi);  
	if (needToDeleteSignal) delete analogSignal;
      }
    }	
    
    // free up some memory
    theHitResponse->clear();
  }
  
  
  void addNoiseHits() {
    std::vector<PCaloHit> noiseHits;
    theNoiseHitGenerator->getNoiseHits(noiseHits);
    for(std::vector<PCaloHit>::const_iterator hitItr = noiseHits.begin(),
	  hitEnd = noiseHits.end(); hitItr != hitEnd; ++hitItr) {
      theHitResponse->add(*hitItr);
    }
  }
  
  
 private:

  CaloHitResponse * theHitResponse;
  CaloVNoiseHitGenerator * theNoiseHitGenerator;  
  ESElectronicsSimFast * theElectronicsSim;
  std::vector<DetId> theDetIds;
  bool addNoise_;
  int numESdetId_;
  double zsThreshold_;

  std::string refFile_;
  ifstream *m_histofile;
  double* refHistos_;
  double  histoBin_;
  double  histoInf_;
  double  histoSup_;

  CLHEP::RandGeneral *histoDistribution_;
};

#endif


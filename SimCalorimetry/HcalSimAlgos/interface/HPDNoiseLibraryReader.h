
// --------------------------------------------------------
// A class to read HPD noise from the library.
// The deliverable of the class is the collection of
// noisy HcalDetIds with associated noise in units of fC for
// 10 time samples. During the library production a higher
// theshold is used to find a noisy HPD. A lower threshold is
// used to eliminate adding unnecessary quite channels to HPD 
// noise event collection. Therefore user may not see whole 18 
// channels for noisy HPD.
//
// Project: HPD noise library reader
// Author: T.Yetkin University of Iowa, Feb. 7, 2008
// $Id: HPDNoiseLibraryReader.h,v 1.4 2012/08/28 14:50:42 yana Exp $
// --------------------------------------------------------

#ifndef HcalSimAlgos_HPDNoiseLibraryReader_h
#define HcalSimAlgos_HPDNoiseLibraryReader_h

#include <memory>
#include <utility>
#include <iostream>
#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/Exception.h"
// Hcal Geometry
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
// HPD Noise Data Frame
#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseReader.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseData.h"
// CLHEP Random numbers
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "TMath.h"

class HPDNoiseLibraryReader{
  
  public:
    HPDNoiseLibraryReader(const edm::ParameterSet&);
    ~HPDNoiseLibraryReader();
    // collection of noisy detIds 
    std::vector<std::pair <HcalDetId, const float* > > getNoisyHcalDetIds();
    // collection of noisy detIds. At least one HcalDetId is alwasy noiosy
    std::vector<std::pair <HcalDetId, const float* > > getBiasedNoisyHcalDetIds();


    std::vector<std::pair <HcalDetId, const float* > > getNoisyHcalDetIds(int timeSliceId);
    // collection of noisy detIds. At least one HcalDetId is alwasy noiosy
    std::vector < std::pair < HcalDetId, const float *> >getBiasedNoisyHcalDetIds(int timeSliceId);
    /** 
    HPD Ion feedback simulation based on LED data. A simple simulation
    which uses gaussian fit to data.
    biased = false ==> HPD noise from Ion Feedback only, unbiased
    biased = true  ==> HPD noise from Ion Feedback only, biased (rate is X times larger than nominal rate)
    */
    double getIonFeedbackNoise(HcalDetId id, double energy, double bias);
    // to be used for standalone tests (from R. Wilkinson)
    // taken from SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h
    static void initializeServices();
  protected:
    void setRandomEngine();
    void setRandomEngine(CLHEP::HepRandomEngine & engine);
  private:
    HPDNoiseData* getNoiseData(int iphi);
    // reads external file provided by the user in /data directrory
    // and fill rate for each HPD.
    void fillRates();
    // compares noise rates for each HPD with randomly thrown numbers
    // and returns the collection of Phis.
    void getNoisyPhis();
    // same as above. The only difference is that at least one phi is
    // always noisy
    void getBiasedNoisyPhis();
    // check if noise is applicable the the HPD
    bool IsNoiseApplicable(int iphi);
    
    //normal random number
    void Rannor(double &a, double &b);
    
    // clear phi vector
    void clearPhi();
    // use int iphi to create HPD names
    std::string itos(int i);    // convert int to string
    
    void shuffleData(int timeSliceId, float* &data);
  
  public: 
    //members
    std::vector<float> theDischargeNoiseRate;
    std::vector<float> theIonFeedbackFirstPeakRate;
    std::vector<float> theIonFeedbackSecondPeakRate;
    std::vector<int>   theNoisyPhi;
    CLHEP::RandFlat *     theRandFlat;
    CLHEP::RandGaussQ*    theRandGaussQ; 
    HPDNoiseReader* theReader;
    std::vector <std::string> theNames;
    std::string theHPDName;
    
};
#endif

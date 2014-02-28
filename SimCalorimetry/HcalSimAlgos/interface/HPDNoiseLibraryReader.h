
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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// Hcal Geometry
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
// HPD Noise Data Frame
#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseReader.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseData.h"
// CLHEP Random numbers
#include "TMath.h"

namespace CLHEP {
  class HepRandomEngine;
}

class HPDNoiseLibraryReader{
  
  public:
    HPDNoiseLibraryReader(const edm::ParameterSet&);
    ~HPDNoiseLibraryReader();
    // collection of noisy detIds 
    std::vector<std::pair <HcalDetId, const float* > > getNoisyHcalDetIds(CLHEP::HepRandomEngine*);
    // collection of noisy detIds. At least one HcalDetId is alwasy noiosy
    std::vector<std::pair <HcalDetId, const float* > > getBiasedNoisyHcalDetIds(CLHEP::HepRandomEngine*);


    std::vector<std::pair <HcalDetId, const float* > > getNoisyHcalDetIds(int timeSliceId, CLHEP::HepRandomEngine*);
    // collection of noisy detIds. At least one HcalDetId is alwasy noiosy
    std::vector < std::pair < HcalDetId, const float *> >getBiasedNoisyHcalDetIds(int timeSliceId, CLHEP::HepRandomEngine*);
    /** 
    HPD Ion feedback simulation based on LED data. A simple simulation
    which uses gaussian fit to data.
    biased = false ==> HPD noise from Ion Feedback only, unbiased
    biased = true  ==> HPD noise from Ion Feedback only, biased (rate is X times larger than nominal rate)
    */
    double getIonFeedbackNoise(HcalDetId id, double energy, double bias, CLHEP::HepRandomEngine*);
    // to be used for standalone tests (from R. Wilkinson)
    // taken from SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h
    static void initializeServices();
  private:
    HPDNoiseData* getNoiseData(int iphi, CLHEP::HepRandomEngine*);
    // reads external file provided by the user in /data directrory
    // and fill rate for each HPD.
    void fillRates();
    // compares noise rates for each HPD with randomly thrown numbers
    // and returns the collection of Phis.
    void getNoisyPhis(CLHEP::HepRandomEngine*);
    // same as above. The only difference is that at least one phi is
    // always noisy
    void getBiasedNoisyPhis(CLHEP::HepRandomEngine*);
    // check if noise is applicable the the HPD
    bool IsNoiseApplicable(int iphi);
    
    //normal random number
    void Rannor(double &a, double &b, CLHEP::HepRandomEngine*);
    
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
    HPDNoiseReader* theReader;
    std::vector <std::string> theNames;
    std::string theHPDName;
    
};
#endif

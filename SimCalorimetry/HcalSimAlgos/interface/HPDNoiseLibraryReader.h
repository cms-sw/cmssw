
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
// $Id: $
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
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/Exception.h"
// Hcal Geometry
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
// HPD Noise Data Frame
#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseReader.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseData.h"
// CLHEP Random numbers
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"
class HPDNoiseLibraryReader{
  
  public:
    HPDNoiseLibraryReader(const edm::ParameterSet&);
    ~HPDNoiseLibraryReader();
    // collection of noisy detIds 
    std::vector<std::pair <HcalDetId, const float* > > getNoisyHcalDetIds();
    // collection of noisy detIds. At least one HcalDetId is alwasy noiosy
    std::vector<std::pair <HcalDetId, const float* > > getBiasedNoisyHcalDetIds();
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
    void fillRate();
    // compares noise rates for each HPD with randomly thrown numbers
    // and returns the collection of Phis.
    void getNoisyPhis();
    // same as above. The only difference is that at least one phi is
    // always noisy
    void getBiasedNoisyPhis();
    // check if noise is applicable the the HPD
    bool applyNoise(int iphi);
    // clear phi vector
    void clearPhi();
    // use int iphi to create HPD names
    std::string itos(int i);    // convert int to string
  
  public: 
    HcalTopology  theTopology;
    
    //members
    std::vector<float> theNoiseRate;
    std::vector<int>   theNoisyPhi;
    CLHEP::RandFlat *     theRandFlat;
    CLHEP::RandGaussQ*    theRandGaussQ; 
    HPDNoiseReader* theReader;
    std::vector <std::string> theNames;
    std::string theHPDName;
    
};
#endif

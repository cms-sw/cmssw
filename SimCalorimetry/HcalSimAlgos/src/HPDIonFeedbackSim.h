
// --------------------------------------------------------
// A class to simulated HPD ion feedback noise.
// The deliverable of the class is the ion feedback noise
// for an HcalDetId units of fC or GeV
//
// Project: HPD ion feedback
// Author: T.Yetkin University of Iowa, Feb. 16, 2010
// $Id:$
// --------------------------------------------------------

#ifndef HcalSimAlgos_HPDIonFeedbackSim_h
#define HcalSimAlgos_HPDIonFeedbackSim_h

#include <memory>
#include <utility>
#include <iostream>
#include <vector>
#include <string>
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

// CLHEP Random numbers
#include "CLHEP/Random/RandBinomial.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoissonQ.h"

class HPDIonFeedbackSim{
  
  public:
    HPDIonFeedbackSim(const edm::ParameterSet&);
    ~HPDIonFeedbackSim();
    
    //copied from HFSimParameters.h
    void setDbService(const HcalDbService * service) {theDbService = service;}
    void setRandomEngine(CLHEP::HepRandomEngine & engine);

    // in units of fC
    double getIonFeedbackFromPE(DetId detId, double npe, bool doThermal);
    double getIonFeedback(DetId detId, double signal, double pedWidth, bool doThermal, bool isInGeV);

  private:
    double fCtoGeV(const DetId & detId) const;
    const HcalDbService * theDbService;
    
  public: 
    CLHEP::RandBinomial * theRandBinomial;
    CLHEP::RandFlat*      theRandFlat;
    CLHEP::RandGaussQ *   theRandGauss;
    CLHEP::RandPoissonQ*  theRandPoissonQ; 
};
#endif

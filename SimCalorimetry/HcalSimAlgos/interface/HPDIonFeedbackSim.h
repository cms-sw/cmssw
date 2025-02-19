
// --------------------------------------------------------
// A class to simulated HPD ion feedback noise.
// The deliverable of the class is the ion feedback noise
// for an HcalDetId units of fC or GeV
//
// Project: HPD ion feedback
// Author: T.Yetkin University of Iowa, Feb. 16, 2010
// $Id: HPDIonFeedbackSim.h,v 1.3 2011/02/23 19:51:51 rpw Exp $
// --------------------------------------------------------

#ifndef HcalSimAlgos_HPDIonFeedbackSim_h
#define HcalSimAlgos_HPDIonFeedbackSim_h

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVPECorrection.h"
// CLHEP Random numbers
#include "CLHEP/Random/RandBinomial.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoissonQ.h"
class CaloShapes;

class HPDIonFeedbackSim: public CaloVPECorrection
{
  public:
    /// need a shaper in order to set thermal noise
    HPDIonFeedbackSim(const edm::ParameterSet&, const CaloShapes * shapes);
    ~HPDIonFeedbackSim();
    
    //copied from HFSimParameters.h
    void setDbService(const HcalDbService * service) {theDbService = service;}
    /// need a shaper in order to set thermal noise
    void setRandomEngine(CLHEP::HepRandomEngine & engine);

    // in units of fC
    virtual double correctPE(const DetId & detId, double npe) const;
    double getIonFeedback(DetId detId, double signal, double pedWidth, bool doThermal, bool isInGeV);

    void addThermalNoise(CaloSamples & samples);

  private:
    double fCtoGeV(const DetId & detId) const;
    const HcalDbService * theDbService;
    const CaloShapes * theShapes;
    
    mutable CLHEP::RandBinomial * theRandBinomial;
    mutable CLHEP::RandFlat*      theRandFlat;
    mutable CLHEP::RandGaussQ *   theRandGauss;
    mutable CLHEP::RandPoissonQ*  theRandPoissonQ; 

};
#endif

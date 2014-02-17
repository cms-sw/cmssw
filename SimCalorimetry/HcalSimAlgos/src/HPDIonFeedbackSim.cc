
// --------------------------------------------------------
// A class to simulated HPD ion feedback noise.
// The deliverable of the class is the ion feedback noise
// for an HcalDetId units of fC or GeV
//
// Project: HPD ion feedback
// Author: T.Yetkin University of Iowa, Feb. 16, 2010
// $Id: HPDIonFeedbackSim.cc,v 1.6 2011/11/14 11:08:16 abdullin Exp $
// --------------------------------------------------------

#include "SimCalorimetry/HcalSimAlgos/interface/HPDIonFeedbackSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShapes.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "CondFormats/HcalObjects/interface/HcalGain.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidth.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"


using namespace edm;
using namespace std;

// constants for simulation/parameterization
double pe2Charge = 0.333333;    // fC/p.e.

HPDIonFeedbackSim::HPDIonFeedbackSim(const edm::ParameterSet & iConfig, const CaloShapes * shapes)
: theDbService(0), theShapes(shapes),
 theRandBinomial(0), theRandFlat(0), theRandGauss(0), theRandPoissonQ(0)
{
}

HPDIonFeedbackSim::~HPDIonFeedbackSim() 
{
    if(theRandBinomial) delete theRandBinomial;
    if (theRandFlat) delete theRandFlat;
    if (theRandGauss) delete theRandGauss;
    if (theRandPoissonQ) delete theRandPoissonQ;
}


void HPDIonFeedbackSim::setRandomEngine(CLHEP::HepRandomEngine & engine) 
{
    if(theRandBinomial) delete theRandBinomial;
    if (theRandFlat) delete theRandFlat;
    if (theRandGauss) delete theRandGauss;
    if (theRandPoissonQ) delete theRandPoissonQ;

    theRandBinomial = new CLHEP::RandBinomial(engine);
    theRandFlat = new CLHEP::RandFlat(engine);
    theRandGauss = new CLHEP::RandGaussQ(engine);
    theRandPoissonQ = new CLHEP::RandPoissonQ(engine);
}

double HPDIonFeedbackSim::getIonFeedback(DetId detId, double signal, double pedWidth, bool doThermal, bool isInGeV) 
{
    
  //    HcalDetId id = detId; 
    
    double GeVperfC = 1.;
    if(isInGeV) GeVperfC = 1./fCtoGeV(detId);
    
    double charge = signal / GeVperfC;
    
    double noise = 0.;          // fC
    if (charge > 3. * pedWidth) {    // 3 sigma away from pedestal mean
        int npe = int (charge / pe2Charge);
        if(doThermal) {
          double electronEmission = 0.08;
          npe += theRandPoissonQ->fire(electronEmission);
        }

        noise = correctPE(detId, npe) - npe;
    }
    return (noise * GeVperfC);

}


double HPDIonFeedbackSim::correctPE(const DetId & detId, double npe) const
{
    double rateInTail = 0.000211988;//read this from XML file
    double rateInSecondTail = 4.61579e-06;//read this from XML file

    // three gauss fit is applied to data to get ion feedback distribution
    // parameters (in fC)
    // first gaussian
    // double p0 = 9.53192e+05;
    // double p1 = -3.13653e-01;
    // double p2 = 2.78350e+00;

    // second gaussian
    // double p3 = 2.41611e+03;
    double p4 = 2.06117e+01;
    double p5 = 1.09239e+01;

    // third gaussian
    // double p6 = 3.42793e+01;
    double p7 = 5.45548e+01;
    double p8 = 1.59696e+01;

    double noise = 0.;          // fC
    int nFirst  = (int)(theRandBinomial->fire(npe, rateInTail));
    int nSecond = (int)(theRandBinomial->fire(npe, rateInSecondTail));

    for (int j = 0; j < nFirst; ++j) {
      noise += theRandGauss->fire(p4, p5);
    }
    for (int j = 0; j < nSecond; ++j) {
      noise += theRandGauss->fire(p7, p8);
    }

    return npe + std::max(noise/pe2Charge, 0.);
}


void HPDIonFeedbackSim::addThermalNoise(CaloSamples & samples)
{
  // make some chance to add a PE (with a chance of feedback)
  // for each time sample
  double meanPE = 0.02;
  DetId detId(samples.id());
  int nSamples = samples.size();
  const CaloVShape * shape = theShapes->shape(detId);
  for(int i = 0; i < nSamples; ++i) 
  {
    double npe = theRandPoissonQ->fire(meanPE);
    // TODOprobably should time-smear these
    if(npe > 0.)
    {
      // chance of feedback
      npe = correctPE(detId, npe);
      for(int j = i; j < nSamples; ++j)
      {
        double timeFromPE = (j-i) * 25.;
        samples[j] += (*shape)(timeFromPE) * npe;
      }
    }
  }
}


double HPDIonFeedbackSim::fCtoGeV(const DetId & detId) const
{
  assert(theDbService != 0);
  HcalGenericDetId hcalGenDetId(detId);
  const HcalGain* gains = theDbService->getGain(hcalGenDetId);
  const HcalGainWidth* gwidths = theDbService->getGainWidth(hcalGenDetId);
  if (!gains || !gwidths )
  {
    edm::LogError("HcalAmplifier") << "Could not fetch HCAL conditions for channel " << hcalGenDetId;
  }
  // only one gain will be recorded per channel, so just use capID 0 for now
  double result = gains->getValue(0);
  return result;
}

#ifndef CastorSim_CastorSimParameters_h
#define CastorSim_CastorSimParameters_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"

class CastorSimParameters : public CaloSimParameters
{
public:

CastorSimParameters(double simHitToPhotoelectrons, double photoelectronsToAnalog, double samplingFactor, double timePhase, bool syncPhase);
CastorSimParameters(const edm::ParameterSet & p);

  /*
  CastorSimParameters(double simHitToPhotoelectrons, double photoelectronsToAnalog,
                double samplingFactor, double timePhase,
                int readoutFrameSize, int binOfMaximum,
                bool doPhotostatistics, bool syncPhase,
                int firstRing, const std::vector<double> & samplingFactors);
 CastorSimParameters(const edm::ParameterSet & p);
  */

virtual ~CastorSimParameters() {}

void setDbService(const CastorDbService * service) {theDbService = service;}

//virtual double simHitToPhotoelectrons(const DetId & detId) const;

virtual double photoelectronsToAnalog(const DetId & detId) const;

double fCtoGeV(const DetId & detId) const;

  /// the ratio of actual incident energy to deposited energy
  /// in the SimHit
//  virtual double samplingFactor(const DetId & detId) const;


private:
const CastorDbService * theDbService;
double theSamplingFactor;
//std::vector<double> theSamplingFactors;
};

#endif
  

#ifndef EcalSimAlgos_EcalSimParameterMap_h
#define EcalSimAlgos_EcalSimParameterMap_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"


/* \class EcalSimParametersMap
 * \brief map of parameters for the ECAL (EE, EB, preshower) simulation
 *
 */
class EcalSimParameterMap : public CaloVSimParameterMap
{
public:
  /// ctor
  EcalSimParameterMap();
  EcalSimParameterMap(double simHitToPhotoelectronsBarrel, 
                      double simHitToPhotoelectronsEndcap, 
                      double photoelectronsToAnalogBarrel, 
                      double photoelectronsToAnalogEndcap, 
                      double samplingFactor, double timePhase,
                      int readoutFrameSize, int binOfMaximum,
                      bool doPhotostatistics, bool syncPhase);
  /// dtor
  virtual ~EcalSimParameterMap() {}

  /// return the sim parameters relative to the right subdet
  virtual const CaloSimParameters & simParameters(const DetId & id) const;

private:
  /// EB
  CaloSimParameters theBarrelParameters;
  /// EE
  CaloSimParameters theEndcapParameters;
  /// ES
  CaloSimParameters theESParameters;
};

#endif


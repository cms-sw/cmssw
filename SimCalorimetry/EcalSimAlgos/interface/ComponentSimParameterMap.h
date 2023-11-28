#ifndef SimCalorimetry_EcalSimAlgos_ComponentSimParameterMap_h
#define SimCalorimetry_EcalSimAlgos_ComponentSimParameterMap_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"

class ComponentSimParameterMap : public CaloVSimParameterMap {
public:
  ComponentSimParameterMap();
  ComponentSimParameterMap(bool addToBarrel,
                           bool separateDigi,
                           double simHitToPhotoelectronsBarrel,
                           double simHitToPhotoelectronsEndcap,
                           double photoelectronsToAnalogBarrel,
                           double photoelectronsToAnalogEndcap,
                           double samplingFactor,
                           double timePhase,
                           int readoutFrameSize,
                           int binOfMaximum,
                           bool doPhotostatistics,
                           bool syncPhase);
  /// dtor
  ~ComponentSimParameterMap() override {}

  /// return the sim parameters relative to the right subdet
  const CaloSimParameters& simParameters(const DetId& id) const override;
  bool addToBarrel() const { return m_addToBarrel; }
  bool separateDigi() const { return m_separateDigi; }

private:
  bool m_addToBarrel;
  bool m_separateDigi;

  /// EB
  CaloSimParameters theComponentParameters;
};

#endif

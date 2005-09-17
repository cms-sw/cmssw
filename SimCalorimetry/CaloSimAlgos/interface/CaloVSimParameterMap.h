#ifndef CaloVSimParameterMap_h
#define CaloVSimParameterMap_h


namespace cms {
  class CaloSimParameters;
  class DetId;

  class CaloVSimParameterMap
  {
  public:
    virtual const CaloSimParameters & simParameters(const DetId & id) const = 0;
  };
}

#endif


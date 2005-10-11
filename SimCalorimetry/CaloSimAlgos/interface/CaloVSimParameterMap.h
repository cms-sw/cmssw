#ifndef CaloVSimParameterMap_h
#define CaloVSimParameterMap_h

class DetId;

namespace cms {
  class CaloSimParameters;

  class CaloVSimParameterMap
  {
  public:
    virtual const CaloSimParameters & simParameters(const DetId & id) const = 0;
  };
}

#endif


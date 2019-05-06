#ifndef CaloSimAlgos_CaloVSimParameterMap_h
#define CaloSimAlgos_CaloVSimParameterMap_h

class DetId;
class CaloSimParameters;

class CaloVSimParameterMap {
public:
  virtual ~CaloVSimParameterMap() = default;
  virtual const CaloSimParameters &simParameters(const DetId &id) const = 0;
};

#endif

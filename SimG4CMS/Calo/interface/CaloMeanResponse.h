#ifndef SimG4CMS_CaloMeanResponse_h
#define SimG4CMS_CaloMeanResponse_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <vector>

class CaloMeanResponse {

public:    
  
  CaloMeanResponse(edm::ParameterSet const & p);
  virtual ~CaloMeanResponse();
  double   getWeight(int genPID, double genP);

private:

  void     readResponse (std::string fName);

  bool                            useTable;
  double                          scale;
  int                             piLast, pLast;
  std::vector<int>                pionTypes, protonTypes;
  std::vector<double>             pionMomentum, pionTable;
  std::vector<double>             protonMomentum, protonTable;

};

#endif // SimG4CMS_CaloMeanResponse_h

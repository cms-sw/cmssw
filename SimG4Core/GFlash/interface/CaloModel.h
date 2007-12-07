#ifndef SimG4Core_GFlash_CaloModel_H
#define SimG4Core_GFlash_CaloModel_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class GflashEMShowerModel;
class GflashHadronShowerModel;
class GflashHistogram;

class CaloModel 
{
public:

  CaloModel(edm::ParameterSet const &);
  ~CaloModel();

private:

  void build();  

  edm::ParameterSet m_pCaloModel;
  GflashEMShowerModel *theEMShowerModel;
  GflashHadronShowerModel *theHadronShowerModel;
  GflashHistogram* theHisto;
};

#endif

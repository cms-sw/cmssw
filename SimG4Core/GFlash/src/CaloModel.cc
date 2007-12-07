#include "G4RegionStore.hh"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimG4Core/GFlash/interface/CaloModel.h"
#include "SimG4Core/GFlash/interface/GflashEMShowerModel.h"
#include "SimG4Core/GFlash/interface/GflashHadronShowerModel.h"
#include "SimG4Core/GFlash/interface/GflashHistogram.h"

CaloModel::CaloModel(const edm::ParameterSet & p) : m_pCaloModel(p) 
{ 
  theHisto = GflashHistogram::instance();
  if(m_pCaloModel.getParameter<bool>("GflashHistogram")) {
    theHisto->setStoreFlag(true);
    theHisto->bookHistogram();
  }

  build();
}

CaloModel::~CaloModel()
{
  if(m_pCaloModel.getParameter<bool>("GflashEMShowerModel") && theEMShowerModel)
    delete theEMShowerModel;
  if(m_pCaloModel.getParameter<bool>("GflashHadronShowerModel") && theHadronShowerModel)
    delete theHadronShowerModel;
  if(theHisto) delete theHisto;
}

void CaloModel::build()
{
  //using DefaultRegionForTheWorld for an initialization
  G4Region* aRegion = G4RegionStore::GetInstance()->GetRegion("DefaultRegionForTheWorld");

  //Electromagnetic Shower Model
  if(m_pCaloModel.getParameter<bool>("GflashEMShowerModel")) {
    theEMShowerModel  = new GflashEMShowerModel("GflashEMShowerModel",aRegion);
  }    
  //Hadronic Shower Model
  if(m_pCaloModel.getParameter<bool>("GflashHadronShowerModel")) {
    theHadronShowerModel = new GflashHadronShowerModel("GflashHadronShowerModel",aRegion);
  }
} 

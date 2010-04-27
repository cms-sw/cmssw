/* SLHC Calo Trigger 
Template class for all lattice operations
Defines a general card ..
This class can take an object of TYPE T and fill
the lattice with this object

M.Bachtis,S.Dasu
University Of Wisconsin
*/
#ifndef ModuleBase_h
#define ModuleBase_h

#include <vector>
#include <map>
#include <string>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "DataFormats/Common/interface/Ref.h"


#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloRegion.h"
#include "SimDataFormats/SLHC/interface/L1CaloJet.h"


template <class T>
class ModuleBase
{

 protected:
  //Define the Map of the Lattice
  std::map<int,edm::Ref<std::vector<T> > > lattice_;
  //Setup record
  L1CaloTriggerSetup s;

 public:
  ModuleBase(){}
  ModuleBase(const L1CaloTriggerSetup& setup){
    s = setup;
  }


void reset()//Clear the Lattice
  {
    lattice_.clear();
  }

 
 bool isValid(int bin) 
   {
     bool returnVal=false;
     if(lattice_.find(bin)!=lattice_.end()) {

       if(lattice_[bin].isAvailable())
	 returnVal=true;
       else
	 printf("Ref NOT FOUND!\n");
     }
     return returnVal;
     
   }

 edm::Ref<std::vector<T> >  objectAt(int bin) {
   return lattice_[bin];
 }
};

typedef ModuleBase<l1slhc::L1CaloTower> L1CaloTowerModule;
typedef ModuleBase<l1slhc::L1CaloCluster> L1CaloClusterModule;
typedef ModuleBase<l1slhc::L1CaloRegion> L1CaloRegionModule;
typedef ModuleBase<l1slhc::L1CaloJet> L1CaloJetModule;



#endif




// -*- C++ -*-
//
// Package:    CaloRecHitMixer
// Class:      CaloRecHitMixer
//
/**\class CaloRecHitMixer CaloRecHitMixer.cc TauAnalysis/CaloRecHitMixer/src/CaloRecHitMixer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Fri Apr  9 12:15:56 CEST 2010
// $Id: CaloRecHitMixer.h,v 1.3 2012/03/01 16:22:23 fruboes Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"

#include "DataFormats/Common/interface/SortedCollection.h"

#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleanerBase.h"
#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleanerFactory.h"
//
// class decleration
//

template <typename TMyType>
class CaloRecHitMixer : public edm::EDProducer {
   public:
      explicit CaloRecHitMixer(const edm::ParameterSet&);
      ~CaloRecHitMixer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      /*
      edm::InputTag _electrons1;
      edm::InputTag _electrons2;
      */
      edm::VParameterSet psets_; 
      //typedef std::vector< TMyType > TMyTypeCol;
      typedef  edm::SortedCollection< TMyType > TMyTypeCol;
 
      TMyType merge(const TMyType & rh1, const TMyType & rh2);
      TMyType mergeHCAL(const TMyType & rh1, const TMyType & rh2);
 
      TMyType cleanRH(const TMyType & rh, float energy);
      TMyType cleanHCAL(const TMyType & rh, float energy);
    



      // ----------member data ---------------------------
      CaloCleanerBase * caloCleaner_;
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
template <typename TMyType>
CaloRecHitMixer<TMyType>::CaloRecHitMixer(const edm::ParameterSet& iConfig) :
  psets_(iConfig.getParameter< edm::VParameterSet  > ("todo")),
  caloCleaner_(0)
{

   if (psets_.size()==0) {
     throw cms::Exception("Emtpy todo list\n");
   }


   for (size_t i = 0; i < psets_.size(); ++i ){
      edm::InputTag col1 = psets_.at(i).getParameter< edm::InputTag > ("colZmumu");
      edm::InputTag col2 = psets_.at(i).getParameter< edm::InputTag > ("colTauTau");
      std::string ins1=col1.instance();
      std::string ins2=col2.instance();
      if (ins1 != ins2) {
         throw cms::Exception(std::string("Differnt instances given! ")
                               + " " + col1.encode()
                               + " " + col2.encode() + "\n" );
      }
      if (ins1==""){
        produces< TMyTypeCol >();
      } else {
        produces< TMyTypeCol >(ins1);
      }
   }


   caloCleaner_ = CaloCleanerFactory::get()->create( 
              iConfig.getParameter<std::string>("cleaningAlgo"),
              iConfig.getParameter< edm::ParameterSet >("cleaningConfig"));

   std::cout  << "Created: " << caloCleaner_->name() << std::endl;
}

template <typename TMyType>
CaloRecHitMixer<TMyType>::~CaloRecHitMixer()
{

   if (caloCleaner_) delete caloCleaner_;

}


//
// member functions
//

// ------------ method called to produce the data  ------------
template <typename TMyType>
void
CaloRecHitMixer<TMyType>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   caloCleaner_->setEvent(iEvent);

   // TODO: rename col1 col2 to colZmumu and colTauTau
   using namespace edm;

   for (size_t i = 0; i < psets_.size(); ++i ){
      edm::InputTag col1 = psets_.at(i).getParameter< edm::InputTag >  ("colZmumu");
      edm::InputTag col2 = psets_.at(i).getParameter< edm::InputTag > ("colTauTau");
      std::string ins = col1.instance();

      typedef std::map<uint32_t, TMyType> TDetToRecHitMap;
      TDetToRecHitMap myMap;

      std::vector< edm::Handle< TMyTypeCol > > cols;
      edm::Handle< TMyTypeCol  > hCol1;
      iEvent.getByLabel( col1, hCol1);

      edm::Handle< TMyTypeCol > hCol2;
      iEvent.getByLabel( col2, hCol2);

      cols.push_back(hCol1);
      cols.push_back(hCol2);


      typename std::vector< edm::Handle< TMyTypeCol > >::iterator it = cols.begin();
      int colIndex = 0;
      for(;it != cols.end(); ++it)
      {
        for ( typename TMyTypeCol::const_iterator itT = (*it)->begin() ; 
              itT != (*it)->end(); 
              ++itT)
        {
          uint32_t rawId = itT->detid().rawId();
          bool detIdPresentInMap =  (myMap.find(rawId) != myMap.end());

          if (colIndex==0) { // we shouldnt find given detId in map, this is Zmumu collection
            if ( detIdPresentInMap ) {
               throw cms::Exception("Whooops! Rechit allready in map!\n");
            }

            float correction = caloCleaner_->cleanRH(rawId, itT->energy()  );
	    if (  correction ==  0){  
               myMap[rawId]=*itT;
            }
	    if (  correction <  0)  // crossed by muon, remove completely
            {
            } 
            else 
            {
              TMyType cleanedRH = cleanRH(*itT, correction);
              if (cleanedRH.energy()>0) myMap[rawId] = cleanedRH;
            }
             
            // do cleaning here !

          // TODO: do similar xcheck to above: given rh should be present in collection only once
          } else {  // this is tautau collection
            if (!detIdPresentInMap)  {
              // potentiall problem - this cell may be switched off/bad in hw. We know nothing about it...
              myMap[rawId]=*itT; // given detId not in map, add new entry
            }
            else {    // merge two rechits
                    /*
                    edm::Provenance prov=iEvent.getProvenance(it->id());
 
		    std::cout
		     << " Me thinks tautau comes from " << prov.moduleLabel()
		     << " " <<  prov.productInstanceName()
		     << " " <<  prov.processName()
		     << std::endl;*/

                myMap[rawId] = merge(myMap[rawId], *itT );      
            }

          }
        }
        ++colIndex;
      }
      // TODO fill collection
      {
        std::auto_ptr< TMyTypeCol > finalCollection( new TMyTypeCol ) ;
        typename TDetToRecHitMap::const_iterator it =  myMap.begin(), itE = myMap.end();
        for (;it !=itE;++it){
          finalCollection->push_back(it->second);
        }
        iEvent.put(finalCollection, ins); 
      }
   }

}

// ------------ method called once each job just before starting event loop  ------------
template <typename TMyType>
void
CaloRecHitMixer<TMyType>::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
template <typename TMyType>
void
CaloRecHitMixer<TMyType>::endJob() {
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////
///////////////////////////////////////////////////



// note:: rh1 should contain Zmumu part, rh2 - tau tau 
template <typename TMyType>
TMyType CaloRecHitMixer<TMyType>::merge(const TMyType & rh1, const TMyType & rh2){
  //TODO
  if ( rh1.flags()!=0 || rh1.aux()!=0 ) std::cout << " XXX Got rechit with nonzero flags or aux " 
                                   << rh1.flags()  
                                   << " " <<rh1.aux()  
                                   << " " << typeid(rh1).name()
                                   << std::endl;
  if ( rh2.flags()!=0 || rh2.aux()!=0 ) std::cout << " XXX Got rechit with nonzero flags or aux " 
                                   << rh2.flags()  
                                   << " " << rh2.aux()  
                                   << " " << typeid(rh2).name()
                                   << std::endl;


  // is time calculation good this way?
  // xxx
  TMyType rhRet( rh1.detid(), rh1.energy()+rh2.energy(), rh2.time() );
  //std::cout << "XXX generic " << rh1.energy() << " " << rh2.energy() << std::endl;
  //TMyType rhRet(rh2);
  return rhRet;
}


template <>
EcalRecHit CaloRecHitMixer<EcalRecHit>::merge(const EcalRecHit & rhZmumu, const EcalRecHit & rhTauTau){

  /// TODO: flags, aux
  // for now - take flags from Zmumu (data), since it is more likely to show problems than MC
  // is time calculation good this way?
  // 
  //

  //*
  //EcalRecHit rhRet(rhZmumu.detid(), rhZmumu.energy()+rhTauTau.energy(), rhTauTau.time(), rhTauTau.flags(), rhTauTau.checkFlagMask(0xFFFF) );
  EcalRecHit rhRet(rhZmumu.detid(), rhZmumu.energy()+rhTauTau.energy(), rhTauTau.time(), rhTauTau.flags(), rhZmumu.checkFlagMask(0xFFFF) );
  return rhRet;
  // */
  // std::cout << "XXX ecal " << rhZmumu.energy() << " " << rhTauTau.energy() << std::endl;
  //EcalRecHit rhRet(rhTauTau);
  //return rhRet;
}


// for the moment dont care about flags, aux
template < typename TMyType >
TMyType CaloRecHitMixer<TMyType>::mergeHCAL(const TMyType & rhZmumu, const TMyType & rhTauTau){

  
  //*
  TMyType rhRet(rhZmumu.detid(), rhZmumu.energy()+rhTauTau.energy(), rhTauTau.time());

  // for now - take flags from Zmumu (data), since it is more likely to show problems than MC
  //rhRet.setFlags(rhTauTau.flags());
  rhRet.setFlags(rhZmumu.flags());

  rhRet.setAux(rhTauTau.aux()); // 4_2_6 - aux seems not to be used anywere (LXR search), 
                                // only in  Validation/HcalRecHits/src/HcalRecHitsValidation.cc
                                // same for 5_0_0
  // */
  //TMyType rhRet(rhTauTau);
  //std::cout << "XXX hcal " << rhZmumu.energy() << " " << rhTauTau.energy() << std::endl;
  return rhRet;


}

template <>
HBHERecHit CaloRecHitMixer<HBHERecHit>::merge(const HBHERecHit & rhZmumu, const HBHERecHit & rhTauTau){
  return mergeHCAL(rhZmumu, rhTauTau);
}

template <>
HORecHit CaloRecHitMixer<HORecHit>::merge(const HORecHit & rhZmumu, const HORecHit & rhTauTau){
  return mergeHCAL(rhZmumu, rhTauTau);
}


template <>
HFRecHit CaloRecHitMixer<HFRecHit>::merge(const HFRecHit & rhZmumu, const HFRecHit & rhTauTau){
  return mergeHCAL(rhZmumu, rhTauTau);
}




template <typename TMyType> TMyType CaloRecHitMixer<TMyType>::cleanRH(const TMyType & rh, float energy){

  float newEn = rh.energy()-energy;
  if (newEn<0) newEn = 0;
  TMyType rhRet( rh.detid(), newEn, rh.time() );

  return  rhRet;
}


template <>
EcalRecHit CaloRecHitMixer<EcalRecHit>::cleanRH(const EcalRecHit & rh, float energy){

  float newEn = rh.energy()-energy;
  if (newEn<0) newEn = 0;
  //*
  EcalRecHit rhRet(rh.detid(), newEn, rh.time(), rh.flags(), rh.checkFlagMask(0xFFFF) );
  return rhRet;
}



// for the moment dont care about flags, aux
template < typename TMyType >
TMyType CaloRecHitMixer<TMyType>::cleanHCAL(const TMyType & rh, float energy){


  float newEn = rh.energy()-energy;
  if (newEn<0) newEn = 0;
  TMyType rhRet(rh.detid(), newEn, rh.time());
  
  rhRet.setFlags(rh.flags());
  
  rhRet.setAux(rh.aux()); // 4_2_6 - aux seems not to be used anywere (LXR search), 
                                // only in  Validation/HcalRecHits/src/HcalRecHitsValidation.cc
                                // same for 5_0_0

  return rhRet;
  

}

template <>
HBHERecHit CaloRecHitMixer<HBHERecHit>::cleanRH(const HBHERecHit & rh, float energy){
  return cleanHCAL(rh, energy);
}

template <>
HORecHit CaloRecHitMixer<HORecHit>::cleanRH(const HORecHit & rh, float energy){
  return cleanHCAL(rh, energy);

}


template <>
HFRecHit CaloRecHitMixer<HFRecHit>::cleanRH(const HFRecHit & rh, float energy){
  return cleanHCAL(rh, energy);

}






/* L1CaloTowerProducer
Reads TPGs, fixes the energy scale compression and produces towers

M.Bachtis,S.Dasu
University of Wisconsin-Madison
*/


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
#include "SimDataFormats/SLHC/interface/L1CaloTower.h"


class L1CaloTowerProducer : public edm::EDProducer {
   public:
      explicit L1CaloTowerProducer(const edm::ParameterSet&);
      ~L1CaloTowerProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      /*INPUTS*/

      //Calorimeter Digis
      edm::InputTag ecalDigis_;
      edm::InputTag hcalDigis_;

};


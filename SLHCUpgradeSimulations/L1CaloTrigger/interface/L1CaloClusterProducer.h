/* L1CaloClusterProducer
Performs Clustering in Calorimeter
and produces a Cluster Collection

M.Bachtis,S.Dasu
University of Wisconsin-Madison
*/


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/CaloClusteringCard.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetupRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

class L1CaloClusterProducer : public edm::EDProducer {
   public:
      explicit L1CaloClusterProducer(const edm::ParameterSet&);
      ~L1CaloClusterProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      /*INPUTS*/

      //Calorimeter Digis
      edm::InputTag towers_;

      CaloClusteringCard card;



};



/* L1ExtraMaker
Creates L1 Extra Objects from Clusters and jets

M.Bachtis,S.Dasu
University of Wisconsin-Madison
*/


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterFwd.h"

#include "SimDataFormats/SLHC/interface/L1CaloJet.h"
#include "SimDataFormats/SLHC/interface/L1CaloJetFwd.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

class L1CaloGeometry;

class L1ExtraTranslator : public edm::EDProducer {
   public:
      explicit L1ExtraTranslator(const edm::ParameterSet&);
      ~L1ExtraTranslator();

   private:

      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      edm::InputTag clusters_;
      edm::InputTag jets_;
      int nParticles_; //Number of Objects to produce
      int nJets_; //Number of Objects to produce




};





L1ExtraTranslator::L1ExtraTranslator(const edm::ParameterSet& iConfig):
  clusters_(iConfig.getParameter<edm::InputTag>("Clusters")),
  jets_(iConfig.getParameter<edm::InputTag>("Jets")),
  nParticles_(iConfig.getParameter<int>("NParticles")),
  nJets_(iConfig.getParameter<int>("NJets"))
{
  //Register Product
  produces<l1extra::L1EmParticleCollection>("EGamma");
  produces<l1extra::L1EmParticleCollection>("IsoEGamma");
  produces<l1extra::L1JetParticleCollection>("Taus");
  produces<l1extra::L1JetParticleCollection>("IsoTaus");
  produces<l1extra::L1JetParticleCollection>("Jets");

}


L1ExtraTranslator::~L1ExtraTranslator()
{}


void
L1ExtraTranslator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace l1slhc;
   using namespace l1extra;


   edm::Handle<L1CaloClusterCollection> clusters;
   edm::Handle<L1CaloJetCollection> jets;

   std::auto_ptr<L1EmParticleCollection>  l1EGamma(new L1EmParticleCollection);
   std::auto_ptr<L1EmParticleCollection>  l1IsoEGamma(new L1EmParticleCollection);
   std::auto_ptr<L1JetParticleCollection>  l1Tau(new L1JetParticleCollection);
   std::auto_ptr<L1JetParticleCollection>  l1IsoTau(new L1JetParticleCollection);
   std::auto_ptr<L1JetParticleCollection>  l1Jet(new L1JetParticleCollection);




   if(iEvent.getByLabel(clusters_,clusters))
   {
     int NEGamma=0;
    int NIsoEGamma=0;
    int NIsoTau=0;
    int NTau=0;


    //sort clusters
   //Put Clusters to file
    L1CaloClusterCollection finalClusters = *clusters; 
    std::sort(finalClusters.begin(),finalClusters.end(),HigherClusterEt());
    


     for(size_t i = 0;i<finalClusters.size();++i)
       {
	 //EGamma
	 if(finalClusters.at(i).isEGamma()&&NEGamma<nParticles_)
	   {
	     printf("New L1 EGAMMA pt,eta,phi %f %f %f\n",finalClusters.at(i).p4().pt(),finalClusters.at(i).p4().eta(),finalClusters.at(i).p4().phi()); 
	     l1EGamma->push_back(L1EmParticle(finalClusters.at(i).p4()));
	     NEGamma++;
	   }

	 //Isolated EGamma
	 if(finalClusters.at(i).isIsoEGamma()&&NIsoEGamma<nParticles_)
	   {
	     l1IsoEGamma->push_back(L1EmParticle(finalClusters.at(i).p4()));
	     NIsoEGamma++;
	   }

	 //Taus
	 if(NTau<nParticles_)
	   if(abs(finalClusters.at(i).iEta())<=26&&finalClusters.at(i).isTau())
	     {
	       l1Tau->push_back(L1JetParticle(finalClusters.at(i).p4()));
	       NTau++;
	     }


	 //IsoTaus
	 if(NIsoTau<nParticles_)
	   if(abs(finalClusters.at(i).iEta())<=26&&finalClusters.at(i).isIsoTau())
	     {
	       l1IsoTau->push_back(L1JetParticle(finalClusters.at(i).p4()));
	       NIsoTau++;
	     }
	 
       }

     iEvent.put(l1EGamma,"EGamma");
     iEvent.put(l1IsoEGamma,"IsoEGamma");
     iEvent.put(l1Tau,"Taus");
     iEvent.put(l1IsoTau,"IsoTaus");
   }


   if(iEvent.getByLabel(jets_,jets))
     {
       
       size_t N=nJets_;
       if(int(jets->size())< nJets_)
       N=jets->size();
       
       for(size_t i = 0;i<N;++i)
       {
	 l1Jet->push_back(L1JetParticle((*jets)[i].p4()));
       }
       
       iEvent.put(l1Jet,"Jets");
     }
}


// ------------ method called once each job just after ending the event loop  ------------
void
L1ExtraTranslator::endJob() {
}

//#define DEFINE_ANOTHER_FWK_MODULE(type) DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<type>,#type); DEFINE_FWK_PSET_DESC_FILLER(type)
DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<L1ExtraTranslator>,"L1ExtraTranslator"); DEFINE_FWK_PSET_DESC_FILLER(L1ExtraTranslator);
//DEFINE_ANOTHER_FWK_MODULE(L1ExtraTranslator);


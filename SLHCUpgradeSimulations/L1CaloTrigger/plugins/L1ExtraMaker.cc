#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1ExtraMaker.h"

L1ExtraMaker::L1ExtraMaker(const edm::ParameterSet& iConfig):
  clusters_(iConfig.getParameter<edm::InputTag>("Clusters")),
  jets_(iConfig.getParameter<edm::InputTag>("Jets")),
  nObjects_(iConfig.getParameter<int>("NObjects"))
{
  //Register Product
  produces<l1extra::L1EmParticleCollection>("EGamma");
  produces<l1extra::L1EmParticleCollection>("IsoEGamma");
  produces<l1extra::L1JetParticleCollection>("Taus");
  produces<l1extra::L1JetParticleCollection>("IsoTaus");
  produces<l1extra::L1JetParticleCollection>("Jets");

}


L1ExtraMaker::~L1ExtraMaker()
{}


void
L1ExtraMaker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
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

     for(size_t i = 0;i<clusters->size();++i)
       {
	 //EGamma
	 if((*clusters)[i].isEGamma()&&NEGamma<nObjects_)
	   {
	     l1EGamma->push_back(L1EmParticle((*clusters)[i].p4()));
	     NEGamma++;
	   }

	 //Isolated EGamma
	 if((*clusters)[i].isIsoEGamma()&&NIsoEGamma<nObjects_)
	   {
	     l1IsoEGamma->push_back(L1EmParticle((*clusters)[i].p4()));
	     NIsoEGamma++;
	   }

	 //Taus
	 if(NTau<nObjects_)
	   if(abs((*clusters)[i].iEta())<=26)
	     {
	       l1Tau->push_back(L1JetParticle((*clusters)[i].p4()));
	       NTau++;
	     }


	 //IsoTaus
	 if((*clusters)[i].isTau()&&NIsoTau<nObjects_)
	   if(abs((*clusters)[i].iEta())<=26)
	     {
	       l1IsoTau->push_back(L1JetParticle((*clusters)[i].p4()));
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
       
       size_t N=nObjects_;
       if(int(jets->size())<nObjects_)
       N=jets->size();
       
       for(size_t i = 0;i<N;++i)
       {
	 l1Jet->push_back(L1JetParticle((*jets)[i].p4()));
       }
       
       iEvent.put(l1Jet,"Jets");
     }
}



// ------------ method called once each job just before starting event loop  ------------
void
L1ExtraMaker::beginJob(const edm::EventSetup&)
{

}


// ------------ method called once each job just after ending the event loop  ------------
void
L1ExtraMaker::endJob() {
}




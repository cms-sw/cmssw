#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"

#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"

//
// class declaration
//
class HSCParticleSelector : public edm::EDProducer {
   public:
      explicit HSCParticleSelector(const edm::ParameterSet&);
      ~HSCParticleSelector();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      edm::InputTag sourceTag_;

      float minTkdEdx;
      float maxMuBeta;
};


/////////////////////////////////////////////////////////////////////////////////////
HSCParticleSelector::HSCParticleSelector(const edm::ParameterSet& iConfig)
{
   // What is being produced
   produces<susybsm::HSCParticleCollection >();

    // Input products
   sourceTag_     = iConfig.getParameter<edm::InputTag> ("source");

    // matching criteria products
   minTkdEdx      = iConfig.getParameter<double>  ("minTkdEdx");    // 4.0; 
   maxMuBeta      = iConfig.getParameter<double>  ("maxMuBeta");    // 0.9; 
} 

/////////////////////////////////////////////////////////////////////////////////////
HSCParticleSelector::~HSCParticleSelector(){
}

/////////////////////////////////////////////////////////////////////////////////////
void HSCParticleSelector::beginJob() {
}

/////////////////////////////////////////////////////////////////////////////////////
void HSCParticleSelector::endJob(){
}

/////////////////////////////////////////////////////////////////////////////////////
void HSCParticleSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
      // Source Collection
      edm::Handle<susybsm::HSCParticleCollection > SourceHandle;
      if (!iEvent.getByLabel(sourceTag_, SourceHandle)) {
            edm::LogError("") << ">>> HSCParticleCollection does not exist !!!";
            return;
      }
      susybsm::HSCParticleCollection Source = *SourceHandle.product();


      // Output Collection
      susybsm::HSCParticleCollection* output = new susybsm::HSCParticleCollection;
      std::auto_ptr<susybsm::HSCParticleCollection> result(output);

      // cleanup the collection based on MUON AND/OR TK Beta
      for(susybsm::HSCParticleCollection::iterator hscpcandidate = Source.begin(); hscpcandidate < Source.end(); ++hscpcandidate) {
         if( hscpcandidate->Tk().dedx() < minTkdEdx)continue;
         if( hscpcandidate->Dt().invBeta>0 && 1.0/hscpcandidate->Dt().invBeta > maxMuBeta)continue;

         susybsm::HSCParticle* newhscp = new susybsm::HSCParticle(*hscpcandidate);
         output->push_back(*newhscp);
      }

      iEvent.put(result);
}

DEFINE_FWK_MODULE(HSCParticleSelector);





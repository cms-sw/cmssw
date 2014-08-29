#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"

#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"
#include "SUSYBSMAnalysis/HSCP/interface/CandidateSelector.h"

//
// class declaration
//
class HSCParticleSelector : public edm::EDFilter {
   public:
      explicit HSCParticleSelector(const edm::ParameterSet&);
      ~HSCParticleSelector();

   private:
      virtual void beginJob() override ;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;

      edm::EDGetTokenT<susybsm::HSCParticleCollection> sourceToken_;

      bool			      Filter_;
      std::vector<CandidateSelector*> Selectors;
};


/////////////////////////////////////////////////////////////////////////////////////
HSCParticleSelector::HSCParticleSelector(const edm::ParameterSet& iConfig)
{
   // What is being produced
   produces<susybsm::HSCParticleCollection >();

   // Input products
   sourceToken_     = consumes<susybsm::HSCParticleCollection>(iConfig.getParameter<edm::InputTag> ("source"));
   Filter_        = iConfig.getParameter<bool>          ("filter");

   // Load all the selections
   std::vector<edm::ParameterSet> SelectionParameters = iConfig.getParameter<std::vector<edm::ParameterSet> >("SelectionParameters");
   for(unsigned int i=0;i<SelectionParameters.size();i++){
      Selectors.push_back(new CandidateSelector(SelectionParameters[i]) );
   }
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
bool HSCParticleSelector::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
      // Source Collection
      edm::Handle<susybsm::HSCParticleCollection > SourceHandle;
      if (!iEvent.getByToken(sourceToken_, SourceHandle)) {
            edm::LogError("") << ">>> HSCParticleCollection does not exist !!!";
            return false;
      }
      susybsm::HSCParticleCollection Source = *SourceHandle.product();


      // Output Collection
      susybsm::HSCParticleCollection* output = new susybsm::HSCParticleCollection;
      std::auto_ptr<susybsm::HSCParticleCollection> result(output);

      // cleanup the collection based on the input selection
      for(susybsm::HSCParticleCollection::iterator hscpcandidate = Source.begin(); hscpcandidate < Source.end(); ++hscpcandidate){
         bool decision = false;
         for(unsigned int i=0;i<Selectors.size();i++){decision |= Selectors[i]->isSelected(*hscpcandidate);}
         if(decision){
            susybsm::HSCParticle* newhscp = new susybsm::HSCParticle(*hscpcandidate);
            output->push_back(*newhscp);
         }
      }

      bool filterResult = !Filter_ || (Filter_ && output->size()>=1);

      iEvent.put(result);

      return filterResult;
}

DEFINE_FWK_MODULE(HSCParticleSelector);





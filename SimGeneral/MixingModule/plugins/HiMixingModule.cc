// -*- C++ -*-
//
// Package:    HiMixingModule
// Class:      HiMixingModule
// 
/**\class HiMixingModule HiMixingModule.cc HeavyIonsAnalysis/HiMixingModule/src/HiMixingModule.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Tue Feb 17 17:32:06 EST 2009
//
//


// system include files
#include <vector>
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupMixingContent.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Services/src/Memory.h"

#include <string>

using namespace std;

//
// class decleration
//

namespace edm{

   class HiMixingModule;

   class HiMixingWorkerBase{
   public:
      explicit HiMixingWorkerBase(){;}
      
      HiMixingWorkerBase(std::string& object, std::vector<InputTag>& tags, std::string& label) :
      object_(object),
      tags_(tags),
      label_(label)
      {;}
      virtual ~HiMixingWorkerBase(){;}
      //   virtual void put(edm::Event &e) = 0;
      virtual void addSignals(edm::Event &e) = 0;

      std::string object_;
      std::vector<InputTag> tags_;
      std::string label_;
   };

   
   template <class T>
   class HiMixingWorker : public HiMixingWorkerBase {
   public:
      HiMixingWorker(std::string& object, std::vector<InputTag>& tags, std::string& label) : HiMixingWorkerBase(object,tags, label) {;}
      ~HiMixingWorker(){;}
      void addSignals(edm::Event &e) override {
	 std::vector<Handle<std::vector<T> > > handles;
	 bool get = true;
	 for(size_t itag = 0; itag < tags_.size(); ++itag){
	   LogInfo("HiEmbedding")<<"itag "<<itag;
	   LogInfo("HiEmbedding")<<"label "<<tags_[itag].label();
	   LogInfo("HiEmbedding")<<"instance "<<tags_[itag].instance();
	    Handle<std::vector<T> > hand;
	    handles.push_back(hand);
	    get = get && e.getByLabel(tags_[itag],handles[itag]);
	    if(!get)  LogWarning("Product inconsistency")<<"One of the sub-events is missing the product with type "
						       <<object_
						       <<", instance "
						       <<tags_[itag].instance()
						       <<" whereas the other one is fine.";
	 }
	 
         if(get){
	    std::auto_ptr<CrossingFrame<T> > crFrame(new CrossingFrame<T>() );	    
	    crFrame->addSignals(handles[0].product(),e.id());
	    for(size_t itag = 1; itag < tags_.size(); ++itag){
               std::vector<T>* product = const_cast<std::vector<T>*>(handles[itag].product());
               EncodedEventId id(0,itag);
               for(auto& item : *product) {
                 item.setEventId(id);
               }
	       crFrame->addPileups(*product);	 
	    }
	    e.put(crFrame,label_);
	 }
      }
   };
   
template <>
void HiMixingWorker<HepMCProduct>::addSignals(edm::Event &e){

   std::vector<Handle<HepMCProduct> > handles;
   bool get = true;
   for(size_t itag = 0; itag< tags_.size(); ++itag){
      Handle<HepMCProduct> hand;
      handles.push_back(hand);
      get = get && e.getByLabel(tags_[itag],handles[itag]);
      if(!get)  LogWarning("Product inconsistency")<<"One of the sub-events is missing the product with type "
						 <<object_
						 <<", instance "
						 <<tags_[itag].instance()
						 <<" whereas the other one is fine.";
   }
   
   if(get){
      std::auto_ptr<CrossingFrame<HepMCProduct> > crFrame(new CrossingFrame<HepMCProduct>() );
      crFrame->addSignals(handles[0].product(),e.id());
      for(size_t itag = 1; itag < tags_.size(); ++itag){
         HepMCProduct* product = const_cast<HepMCProduct*>(handles[itag].product());
         crFrame->addPileups(*product);	 
      }
      e.put(crFrame,label_);
   }
}

class HiMixingModule : public edm::EDProducer {
   public:
      explicit HiMixingModule(const edm::ParameterSet&);
      ~HiMixingModule();

   private:
  virtual void beginJob() override ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;
      bool verifyRegistry(std::string object, std::string subdet, InputTag &tag,std::string &label);      
      // ----------member data ---------------------------

   std::vector<HiMixingWorkerBase *> workers_;

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
HiMixingModule::HiMixingModule(const edm::ParameterSet& pset)
{

   ParameterSet ps=pset.getParameter<ParameterSet>("mixObjects");
   std::vector<std::string> names = ps.getParameterNames();
   std::vector<std::string> simtags = pset.getParameter<std::vector<std::string> >("srcSIM");
   std::vector<std::string> gentags = pset.getParameter<std::vector<std::string> >("srcGEN");

   if(simtags.size() != gentags.size()) LogError("MixingInput")<<"Generator and Simulation input lists are not matching each other"<<endl;
   
   for (std::vector<string>::iterator it=names.begin();it!= names.end();++it){
      
      ParameterSet pstag=ps.getParameter<ParameterSet>((*it));
      if (!pstag.exists("type"))  continue; //to allow replacement by empty pset
      std::string object = pstag.getParameter<std::string>("type");
      std::vector<InputTag>  tags=pstag.getParameter<std::vector<InputTag> >("input");

      std::string signal;
      for(size_t itag = 0; itag < tags.size(); ++itag){
         InputTag tag=tags[itag];
	 std::vector<InputTag> inputs;

	 for(size_t input = 0; input < simtags.size(); ++input){
	    if (object=="HepMCProduct") signal = gentags[input];
	    else signal = simtags[input];
	    inputs.push_back(InputTag(signal,tag.instance()));
	 }

	 std::string label=tag.label()+tag.instance();
	 //	 verifyRegistry(object,std::string(""),tag,label);
	    if (object=="HepMCProduct"){
	       workers_.push_back(new HiMixingWorker<HepMCProduct>(object,inputs,label));
	       produces<CrossingFrame<HepMCProduct> >(label);
	       consumes<HepMCProduct>(tag);
	    }else if (object=="SimTrack"){
	       workers_.push_back(new HiMixingWorker<SimTrack>(object,inputs,label));
	       produces<CrossingFrame<SimTrack> >(label);
	       consumes<std::vector<SimTrack> >(tag);
	    }else if (object=="SimVertex"){
	       workers_.push_back(new HiMixingWorker<SimVertex>(object,inputs,label));
	       produces<CrossingFrame<SimVertex> >(label);
	       consumes<std::vector<SimVertex> >(tag);
	    }else if (object=="PSimHit"){
	       workers_.push_back(new HiMixingWorker<PSimHit>(object,inputs,label));
	       produces<CrossingFrame<PSimHit> >(label);
	       consumes<std::vector<PSimHit> >(tag);
	    }else if (object=="PCaloHit"){
	       workers_.push_back(new HiMixingWorker<PCaloHit>(object,inputs,label));
	       produces<CrossingFrame<PCaloHit> >(label);
	       consumes<std::vector<PCaloHit> >(tag);
	    }else LogInfo("Error")<<"What the hell is this object?!";
	    
	    LogInfo("HiMixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;	 
      }
   }

   produces<PileupMixingContent>();
}
   

HiMixingModule::~HiMixingModule()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
HiMixingModule::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   for(size_t i = 0; i < workers_.size(); ++i){
      (workers_[i])->addSignals(iEvent);
   }

   std::auto_ptr< PileupMixingContent > PileupMixing_ = std::auto_ptr< PileupMixingContent >(new PileupMixingContent());
   iEvent.put(PileupMixing_);

}

// ------------ method called once each job just before starting event loop  ------------
void 
HiMixingModule::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HiMixingModule::endJob() {
}

bool HiMixingModule::verifyRegistry(std::string object, std::string subdet, InputTag &tag,std::string &label) {
   // verify that the given product exists in the product registry                                                                          
   // and create the label to be given to the CrossingFrame                                                                                 

   edm::Service<edm::ConstProductRegistry> reg;
   // Loop over provenance of products in registry.                                                                                         
   std::string lookfor;
   if (object=="HepMCProduct") lookfor="edm::"+object;//exception for HepMCProduct                                                          
   else if (object=="edm::HepMCProduct") lookfor=object;
   else  lookfor="std::vector<"+object+">";
   bool found=false;
   for (edm::ProductRegistry::ProductList::const_iterator it = reg->productList().begin();
	it != reg->productList().end(); ++it) {
      // See FWCore/Framework/interface/BranchDescription.h                                                                                  
      // BranchDescription contains all the information for the product.                                                                     
      edm::BranchDescription desc = it->second;
      if (desc.className()==lookfor && desc.moduleLabel()==tag.label() && desc.productInstanceName()==tag.instance()) {
	 label=desc.moduleLabel()+desc.productInstanceName();
	 found=true;
	 /*
	 wantedBranches_.push_back(desc.friendlyClassName() + '_' +
				   desc.moduleLabel() + '_' +
				   desc.productInstanceName());
	 */
	 break;
      }
   }
   if (!found) {
      LogWarning("MixingModule")<<"!!!!!!!!!Could not find in registry requested object: "<<object<<" with "<<tag<<".\nWill NOT be considered for mixing!!!!!!!!!";
      return false;
   }

   return true;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiMixingModule);                                                                               

}

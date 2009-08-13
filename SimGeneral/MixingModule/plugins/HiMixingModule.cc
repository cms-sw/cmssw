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
// $Id: HiMixingModule.cc,v 1.1 2009/08/03 09:41:55 yilmaz Exp $
//
//


// system include files
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
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Selector.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

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


#include <vector>
#include <string>

using namespace std;

//
// class decleration
//

namespace edm{

   class HiMixingModule;

class MixingWorkerBase{
public:
   explicit MixingWorkerBase(){;}

   MixingWorkerBase(std::string& object, InputTag &tag1,InputTag &tag2, std::string& label) :
      object_(object),
      tag1_(tag1),
      tag2_(tag2),
      label_(label)
   {;}
   virtual ~MixingWorkerBase(){;}
   //   virtual void put(edm::Event &e) = 0;
   virtual void addSignals(edm::Event &e) = 0;

   std::string object_;
   InputTag tag1_;
   InputTag tag2_;
   std::string label_;
};


   template <class T>
   class MixingWorker : public MixingWorkerBase {
   public:
      MixingWorker(std::string& object, InputTag &tag1,InputTag &tag2, std::string& label) : MixingWorkerBase(object,tag1,tag2, label) {;}
      ~MixingWorker(){;}
     void addSignals(edm::Event &e){

	 cout<<"addSignals Working"<<endl;

	 Handle<std::vector<T> > handle1;
	 Handle<std::vector<T> > handle2;
	 bool get1 = e.getByLabel(tag1_,handle1);
	 bool get2 = e.getByLabel(tag2_,handle2);

         if(get1 && get2){
	    std::auto_ptr<CrossingFrame<T> > crFrame(new CrossingFrame<T>() );	    
	    // Following should be reconsidered, what should be the bkg, is bcr useful?
	    crFrame->addSignals(handle2.product(),e.id());
	    crFrame->addPileups(0,const_cast< std::vector<T> * >(handle1.product()),e.id().event());	 
	    e.put(crFrame,label_);
	 }else if(get1 || get2){
	    LogError("Product inconsistency")<<"One of the sub-events is missing the product with type "
					     <<object_
					     <<", instance "
					     <<tag1_.instance()
					     <<" whereas the other one is fine.";
	 }
      }
   };

template <>
void MixingWorker<HepMCProduct>::addSignals(edm::Event &e){
   Handle<HepMCProduct> handle1;
   Handle<HepMCProduct> handle2;
   bool get1 = e.getByLabel(tag1_,handle1);
   bool get2 = e.getByLabel(tag2_,handle2);

   if(get1 && get2){
      std::auto_ptr<CrossingFrame<HepMCProduct> > crFrame(new CrossingFrame<HepMCProduct>() );

      // Following should be reconsidered, what should be the bkg, is bcr useful? 

      /*
	std::vector<HepMCProduct> vec1;
	std::vector<HepMCProduct> vec2;
	vec1.push_back(*(handle1.product()));
	vec2.push_back(*(handle2.product()));
	crFrame->addSignals(&vec2,e.id());
	crFrame->addPileups(0,&vec1,e.id().event());
      */

      crFrame->addSignals(handle2.product(),e.id());
      crFrame->addPileups(0, const_cast<HepMCProduct *>(handle1.product()),e.id().event());

      e.put(crFrame,label_);
   }else if(get1 || get2){
      LogError("Product inconsistency")<<"One of the sub-events is missing the product with type "
				       <<object_
				       <<", instance "
				       <<tag1_.instance()
				       <<" whereas the other one is fine.";
   }
   
}

class HiMixingModule : public edm::EDProducer {
   public:
      explicit HiMixingModule(const edm::ParameterSet&);
      ~HiMixingModule();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      bool verifyRegistry(std::string object, std::string subdet, InputTag &tag,std::string &label);      
      // ----------member data ---------------------------

   std::vector<MixingWorkerBase *> workers_;

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
   std::vector<std::string> signals = pset.getParameter<std::vector<std::string> >("signalTag");
   
   for (std::vector<string>::iterator it=names.begin();it!= names.end();++it){
      
      ParameterSet pstag=ps.getParameter<ParameterSet>((*it));
      if (!pstag.exists("type"))  continue; //to allow replacement by empty pset
      std::string object = pstag.getParameter<std::string>("type");
      std::vector<InputTag>  tags=pstag.getParameter<std::vector<InputTag> >("input");

      std::string signal;
      if (object=="HepMCProduct") signal = signals[0];
      else signal = signals[1];

      for(size_t itag = 0; itag < tags.size(); ++itag){
	 InputTag tag=tags[itag];
	 InputTag tag2 = InputTag(signal,tag.instance());
	 
	 std::string label;
	 if (verifyRegistry(object,std::string(""),tag,label)){
	    if (object=="HepMCProduct"){
	       workers_.push_back(new MixingWorker<HepMCProduct>(object,tag,tag2,label));
	       produces<CrossingFrame<HepMCProduct> >(label);
	    }else if (object=="SimTrack"){
	       workers_.push_back(new MixingWorker<SimTrack>(object,tag,tag2,label));
	       produces<CrossingFrame<SimTrack> >(label);
	    }else if (object=="SimVertex"){
	       workers_.push_back(new MixingWorker<SimVertex>(object,tag,tag2,label));
	       produces<CrossingFrame<SimVertex> >(label);
	    }else if (object=="PSimHit"){
	       workers_.push_back(new MixingWorker<PSimHit>(object,tag,tag2,label));
	       produces<CrossingFrame<PSimHit> >(label);
	    }else if (object=="PCaloHit"){
	       workers_.push_back(new MixingWorker<PCaloHit>(object,tag,tag2,label));
	       produces<CrossingFrame<PCaloHit> >(label);
	    }else LogInfo("Error")<<"What the hell is this object?!";
	    
	    LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;	 
	    cout<<"The COUT : "<<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label<<endl;	 
	 }	 
      }
   }  
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
}

// ------------ method called once each job just before starting event loop  ------------
void 
HiMixingModule::beginJob(const edm::EventSetup&)
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
DEFINE_ANOTHER_FWK_MODULE(HiMixingModule);                                                                               

}

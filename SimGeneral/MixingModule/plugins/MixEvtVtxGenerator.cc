#ifndef HI_MixEvtVtxGenerator_H
#define HI_MixEvtVtxGenerator_H
/*
*   $Date: 2009/08/03 09:41:55 $
*   $Revision: 1.1 $
*/
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Provenance/interface/Provenance.h"

#include "TMatrixD.h"

using namespace edm;
using namespace std;


namespace HepMC {
   class FourVector ;
}


class MixEvtVtxGenerator : public edm::EDProducer
{
   public:
      
   // ctor & dtor
   explicit MixEvtVtxGenerator( const edm::ParameterSet& );
   virtual ~MixEvtVtxGenerator();
      
   virtual void produce( edm::Event&, const edm::EventSetup& );
      
   virtual HepMC::FourVector* getVertex(edm::Event&);
   
   protected:

   HepMC::FourVector*       fVertex ;
   TMatrixD *boost_;
   
   private :

   edm::InputTag            signalLabel;
  edm::InputTag            hiLabel;

};

MixEvtVtxGenerator::MixEvtVtxGenerator( const ParameterSet& pset ) 
	: fVertex(0), boost_(0),
	  signalLabel(pset.getParameter<edm::InputTag>("signalLabel")),
          hiLabel(pset.getParameter<edm::InputTag>("heavyIonLabel"))
{   
   produces<bool>("matchedVertex"); 
}

MixEvtVtxGenerator::~MixEvtVtxGenerator() 
{
   delete fVertex ;
   if (boost_ != 0 ) delete boost_;
   // no need since now it's done in HepMCProduct
   // delete fEvt ;
}

HepMC::FourVector* MixEvtVtxGenerator::getVertex( Event& evt){

  Handle<HepMCProduct> input;
  evt.getByLabel(hiLabel,input);

  const HepMC::GenEvent* inev = input->GetEvent();
  HepMC::GenVertex* genvtx = inev->signal_process_vertex();
  if(!genvtx){
    cout<<"No Signal Process Vertex!"<<endl;
    HepMC::GenEvent::particle_const_iterator pt=inev->particles_begin();
    HepMC::GenEvent::particle_const_iterator ptend=inev->particles_end();
    while(!genvtx || ( genvtx->particles_in_size() == 1 && pt != ptend ) ){
      if(!genvtx) cout<<"No Gen Vertex!"<<endl;
      ++pt;
      if(pt == ptend) cout<<"End reached!"<<endl;
      genvtx = (*pt)->production_vertex();
    }
  }
  double aX,aY,aZ,aT;

  aX = genvtx->position().x();
  aY = genvtx->position().y();
  aZ = genvtx->position().z();
  aT = genvtx->position().t();
  
  if(!fVertex) fVertex = new HepMC::FourVector();
  fVertex->set(aX,aY,aZ,aT);
  
  return fVertex;

}


void MixEvtVtxGenerator::produce( Event& evt, const EventSetup& )
{
   
   
   Handle<HepMCProduct> HepMCEvt ;
   
   evt.getByLabel( signalLabel, HepMCEvt ) ;
   
   // generate new vertex & apply the shift 
   //
   HepMCEvt->applyVtxGen( getVertex(evt) ) ;

   //   HepMCEvt->boostToLab( GetInvLorentzBoost(), "vertex" );
   //   HepMCEvt->boostToLab( GetInvLorentzBoost(), "momentum" );
   
   // OK, create a (pseudo)product and put in into edm::Event
   //
   auto_ptr<bool> NewProduct(new bool(true)) ;      
   evt.put( NewProduct ,"matchedVertex") ;
      
   return ;

}

DEFINE_ANOTHER_FWK_MODULE(MixEvtVtxGenerator);

#endif

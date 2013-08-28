#include "Validation/EventGenerator/interface/TTbar_P4Violation.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TTbar_P4Violation::TTbar_P4Violation(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

}


TTbar_P4Violation::~TTbar_P4Violation()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
TTbar_P4Violation::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // --- get TopQuarkAnalysis TtGenEvent
   Handle<TtGenEvent> genEvt;
   iEvent.getByLabel("genEvt", genEvt);

   if(!genEvt.isValid()) return false;

   const reco::GenParticle*        top = 0;
   const reco::GenParticle*    antitop = 0;
   const reco::GenParticle*     bottom = 0;
   const reco::GenParticle* antibottom = 0;
   const reco::GenParticle*      Wplus = 0;
   const reco::GenParticle*      Wmin  = 0;

   top        = genEvt->top();
   antitop    = genEvt->topBar();
   bottom     = genEvt->b();
   antibottom = genEvt->bBar();
   Wplus      = genEvt->wPlus();
   Wmin       = genEvt->wMinus();
   

   if(top && antitop && bottom && antibottom && Wplus && Wmin){
     const reco::Particle::LorentzVector     topP4 =     bottom->p4() + Wplus->p4() ;
     const reco::Particle::LorentzVector antitopP4 = antibottom->p4() + Wmin ->p4() ;
     
     double tolerance = 0.1 ;
     
     bool     topViolated = false ;
     bool antitopViolated = false ;
     
     if ( (top->p4().px() - topP4.px() > tolerance) || 
	  (top->p4().py() - topP4.py() > tolerance) ||
	  (top->p4().pz() - topP4.pz() > tolerance) ||
	  (top->p4().e () - topP4.e () > tolerance) )  {
       
       topViolated = true ;
       
       //printf( "momentum not conserved for top:\n" ) ;
       //printf( " %5.5f\t %5.5f \t %5.5f \t %5.5f \n",      top->p4().px(),     top->p4().py(),     top->p4().pz(),     top->p4().e() ) ;
       //printf( " %5.5f\t %5.5f \t %5.5f \t %5.5f \n",          topP4.px(),         topP4.py(),         topP4.pz(),         topP4.e() ) ;
     }
     
     if ( (antitop->p4().px() - antitopP4.px() > tolerance) || 
	  (antitop->p4().py() - antitopP4.py() > tolerance) ||
	  (antitop->p4().pz() - antitopP4.pz() > tolerance) ||
	  (antitop->p4().e () - antitopP4.e () > tolerance) )  {
       
       antitopViolated = true ;
       
       //printf( "momentum not conserved for anti-top:\n" ) ;
       //printf( " %5.5f\t %5.5f \t %5.5f \t %5.5f \n ", antitop->p4().px(), antitop->p4().py(), antitop->p4().pz(), antitop->p4().e() ) ;
       //printf( " %5.5f\t %5.5f \t %5.5f \t %5.5f \n ",     antitopP4.px(),     antitopP4.py(),     antitopP4.pz(),     antitopP4.e() ) ;
     }
     
     return (topViolated || antitopViolated);
     
     
     // GOSSIE temp
     bool     bottomOK = true ;
     bool antibottomOK = true ;
     if ( fabs(    bottom->p4().pz()) < 1. )     bottomOK = false ;
     if ( fabs(antibottom->p4().pz()) < 1. ) antibottomOK = false ;
     return (bottomOK && antibottomOK);
   }
   return false;
}

// ------------ method called once each job just before starting event loop  ------------
void 
TTbar_P4Violation::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TTbar_P4Violation::endJob() {
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
TTbar_P4Violation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}


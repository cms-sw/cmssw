#include "SimG4Core/Generators/interface/Generator.h"
#include "SimG4Core/Generators/interface/HepMCParticle.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4HEPEvtParticle.hh"
#include "G4ParticleDefinition.hh"
#include "G4UnitsTable.hh"

using namespace edm;
using std::cout;
using std::endl;
using std::string;

Generator::Generator(const ParameterSet & p) : 
    fPtCuts(p.getParameter<bool>("ApplyPtCuts")),
    fEtaCuts(p.getParameter<bool>("ApplyEtaCuts")), 
    fPhiCuts(p.getParameter<bool>("ApplyPhiCuts")),
    //theMinPhiCut(p.getParameter<double>("MinPhiCut")*deg),
    //theMaxPhiCut(p.getParameter<double>("MaxPhiCut")*deg),
    theMinPhiCut(p.getParameter<double>("MinPhiCut")),   // now operates in radians (CMS standard)
    theMaxPhiCut(p.getParameter<double>("MaxPhiCut")),
    theMinEtaCut(p.getParameter<double>("MinEtaCut")),
    theMaxEtaCut(p.getParameter<double>("MaxEtaCut")),
    //theMinPtCut(p.getParameter<double>("MinPtCut")*MeV),
    //theMaxPtCut(p.getParameter<double>("MaxPtCut")*MeV),   
    theMinPtCut(p.getParameter<double>("MinPtCut")),    // now operates in GeV (CMS standard)
    theMaxPtCut(p.getParameter<double>("MaxPtCut")),   
    verbose(p.getUntrackedParameter<int>("Verbosity",0)),
    evt_(0),
    vtx_(0) ,
    weight_(0)
{
    std::cout << " Generator constructed " << std::endl;
}

Generator::~Generator() 
{ 
}

void Generator::HepMC2G4(const HepMC::GenEvent * evt, G4Event * g4evt)
{

   if (vtx_ != 0) delete vtx_;
   vtx_ = new HepLorentzVector((*(evt->vertices_begin()))->position());
    
   if (verbose > 0)
   {
       evt->print();
       cout << " " << endl;
       cout << " Prim.Vtx : " << vtx_->x() << " " 
                              << vtx_->y() << " "
			      << vtx_->z() << endl;
   }

   for(HepMC::GenEvent::vertex_const_iterator vitr= evt->vertices_begin();
   vitr != evt->vertices_end(); ++vitr ) 
   { // loop for vertex ...

      // real vertex?
      G4bool qvtx=false;
      for (HepMC::GenVertex::particle_iterator 
	   pitr= (*vitr)->particles_begin(HepMC::children);
	 pitr != (*vitr)->particles_end(HepMC::children); ++pitr) 
      {

         if (!(*pitr)->end_vertex() && (*pitr)->status()==1) 
	 {
	    qvtx=true;
	    break;
         }
      }
      if (!qvtx) 
      {
	 continue;
      }
      
      // std::cout << " Found good vertex " << std::endl ;

      // check world boundary
      //G4LorentzVector xvtx= (*vitr)-> position();
      HepLorentzVector xvtx = (*vitr)->position() ;
      // fix later
      //if (! CheckVertexInsideWorld(xvtx.vect()*mm)) continue;

      // create G4PrimaryVertex and associated G4PrimaryParticles
      G4PrimaryVertex* g4vtx= 
         new G4PrimaryVertex(xvtx.x()*mm, xvtx.y()*mm, xvtx.z()*mm, 
			     xvtx.t()*mm/c_light);

      for (HepMC::GenVertex::particle_iterator 
	   vpitr= (*vitr)->particles_begin(HepMC::children);
	   vpitr != (*vitr)->particles_end(HepMC::children); ++vpitr) 
      {

         if( (*vpitr)->status() != 1 ) continue;

         //G4LorentzVector p= (*vpitr)-> momentum();
	 HepLorentzVector p = (*vpitr)->momentum() ;
	 
	 if ( !particlePassesPrimaryCuts( p ) ) 
	 {
	    // std::cout << " Particle does NOT pass cuts " << std::endl ;
	    continue ;
	 }

         G4int pdgcode= (*vpitr)-> pdg_id();
	 
         G4PrimaryParticle* g4prim= 
	    new G4PrimaryParticle(pdgcode, p.x()*GeV, p.y()*GeV, p.z()*GeV);

         if ( g4prim->GetG4code() != 0 )
	 { 
            g4prim->SetMass( g4prim->GetG4code()->GetPDGMass() ) ;
	    g4prim->SetCharge( g4prim->GetG4code()->GetPDGCharge() ) ;	 
	 }
	 g4prim->SetWeight( 10000*(*vpitr)->barcode() ) ;
	 setGenId( g4prim, (*vpitr)->barcode() ) ;
	 
	 g4vtx-> SetPrimary(g4prim);
      }
      g4evt-> AddPrimaryVertex(g4vtx);
   } 

   return ;

}

bool Generator::particlePassesPrimaryCuts( const HepLorentzVector& mom ) const 
{

   double phi = mom.phi() ;   
   double pt  = sqrt( mom.x()*mom.x() + mom.y()*mom.y() ) ;
   double eta = -log( tan(mom.theta()/2.) ) ;
      
   if ( (fPtCuts)  && (pt  < theMinPtCut  || pt  > theMaxPtCut) )  return false ;
   if ( (fEtaCuts) && (eta < theMinEtaCut || eta > theMaxEtaCut) ) return false ;
   if ( (fPhiCuts) && (phi < theMinPhiCut || phi > theMaxPhiCut) ) return false ;
   
   return true;   
}

bool Generator::particlePassesPrimaryCuts(const G4PrimaryParticle * p) const
{
    G4ThreeVector mom = p->GetMomentum();
    double        phi = mom.phi() ;
    double        pt  = sqrt(p->GetPx()*p->GetPx() + p->GetPy()*p->GetPy());
    pt /= GeV ;  // need to convert, since Geant4 operates in MeV
    double        eta = -log(tan(mom.theta()/2));
    if (((fPtCuts)  && (pt  < theMinPtCut  || pt  > theMaxPtCut))           ||
	((fEtaCuts) && (eta < theMinEtaCut || eta > theMaxEtaCut))          ||
	((fPhiCuts) && (phi < theMinPhiCut || phi > theMaxPhiCut)))
	return false;
    else return true;
}

void Generator::nonBeamEvent2G4(const HepMC::GenEvent * evt, G4Event * g4evt)
{
    int i = 0; 
    for(HepMC::GenEvent::particle_const_iterator it = evt->particles_begin(); 
	it != evt->particles_end(); ++it )
    {
	i++;
	HepMC::GenParticle * g = (*it);	
	int g_status = g->status();
	// storing only particle with status == 1 	
	if (g_status == 1)
	{
	    HepLorentzVector mom  = g->momentum();
	    int g_id = g->pdg_id();	    
	    G4PrimaryParticle * g4p = 
		new G4PrimaryParticle(g_id,mom.x()*GeV,mom.y()*GeV,mom.z()*GeV);
	    if (g4p->GetG4code() != 0)
	    { 
		g4p->SetMass(g4p->GetG4code()->GetPDGMass());
		g4p->SetCharge(g4p->GetG4code()->GetPDGCharge()) ;
	    }
	    g4p->SetWeight(i*10000);
	    setGenId(g4p,i);
            if (particlePassesPrimaryCuts(g4p))
            {
		HepLorentzVector vtx = g->production_vertex()->position();
                G4PrimaryVertex * v = new 
		     G4PrimaryVertex(vtx.x()*mm,vtx.y()*mm,vtx.z()*mm,vtx.t()*mm/c_light);
                v->SetPrimary(g4p);
                g4evt->AddPrimaryVertex(v);
            }
	}
    } // end loop on HepMC particles
}

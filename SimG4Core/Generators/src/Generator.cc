#include "SimG4Core/Generators/interface/Generator.h"
#include "SimG4Core/Generators/interface/HepMCParticle.h"

// #include "FWCore/Utilities/interface/Exception.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
  theDecLenCut(p.getParameter<double>("DecLenCut")*cm),
  verbose(p.getUntrackedParameter<int>("Verbosity",0)),
  evt_(0),
  vtx_(0) ,
  weight_(0)
{
  edm::LogInfo("SimG4CoreGenerator") << " Generator constructed " ;
}

Generator::~Generator() 
{ 
}

void Generator::HepMC2G4(const HepMC::GenEvent * evt_orig, G4Event * g4evt)
{

  //protection against empty events
  //if ( *(evt_orig->vertices_begin()) == 0 ) 
  //  throw cms::Exception("EventCorruption") << "Input GenEvent with no vertex \n" ;

  if ( *(evt_orig->vertices_begin()) == 0 )
  {
     throw SimG4Exception( "SimG4CoreGenerator: Corrupted Event - GenEvent with no vertex" ) ;
  }
  
  
  HepMC::GenEvent* evt = new HepMC::GenEvent(*evt_orig) ;
    
  //M. Vander Donckt : modified to take the generator event weight  
  if ( evt->weights().size() > 0 )
    {
      weight_ = evt->weights()[0] ;
      for ( int iw=1; iw<evt->weights().size(); iw++ )
	{
	  // terminate if the versot of weights contains a zero-weight
	  if ( evt->weights()[iw] <= 0 ) break;
	  weight_ *= evt->weights()[iw] ;
	}     
    }
  // end modification
  
  // in the future, we probably want to skip events of zero-weight
  // but at this point, it's zero in most cases anyway... 
  // just a note for future... (JY)  


  if (vtx_ != 0) delete vtx_;
  vtx_ = new HepLorentzVector((*(evt->vertices_begin()))->position().x(),
			      (*(evt->vertices_begin()))->position().y(),
			      (*(evt->vertices_begin()))->position().z(),
			      (*(evt->vertices_begin()))->position().t());
  
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
	    }  else if ( (*pitr)->status()== 2 ) {
	      if ( (*pitr)->end_vertex() != 0  ) { 
		HepLorentzVector xvtx = HepLorentzVector((*vitr)->position().x(),
							 (*vitr)->position().y(),
							 (*vitr)->position().z(),
							 (*vitr)->position().t()) ;
		HepLorentzVector dvtx= HepLorentzVector((*pitr)->end_vertex()->position().x(),
							(*pitr)->end_vertex()->position().y(),
							(*pitr)->end_vertex()->position().z(),
							(*pitr)->end_vertex()->position().t());
		double dd=(xvtx-dvtx).rho();
		if (dd>theDecLenCut){
		  qvtx=true;
		  break;
		}
	      }
	    }
	}
      if (!qvtx) 
	{
	  continue;
	}
	
      // check world boundary
      //G4LorentzVector xvtx= (*vitr)-> position();
      HepLorentzVector xvtx = HepLorentzVector((*vitr)->position().x(),
					       (*vitr)->position().y(),
					       (*vitr)->position().z(),
					       (*vitr)->position().t());
      //fix later
      //if (! CheckVertexInsideWorld(xvtx.vect()*mm)) continue;
	
      // create G4PrimaryVertex and associated G4PrimaryParticles
      G4PrimaryVertex* g4vtx= 
	new G4PrimaryVertex(xvtx.x()*mm, xvtx.y()*mm, xvtx.z()*mm, 
			    xvtx.t()*mm/c_light);
		
      for (HepMC::GenVertex::particle_iterator 
	     vpitr= (*vitr)->particles_begin(HepMC::children);
	   vpitr != (*vitr)->particles_end(HepMC::children); ++vpitr) 
	{
	
	  // M. Vander Donckt modification: to take also decay mother
	  // in case decay length is large; decay procuts get setup
	  // as daughters of G4Particle in this case, through the method
	  // particleAssignDaughters, and they get marked 1000+status 
	  // in the generator product (this seem to "violate" the idea 
	  // that a product can't be modified one it's in edm::Event... 
	  // but it seems to fly...)
	  double decay_length=-1;
	  if ( (*vpitr)->status() == 2 ) 
	    {
	      // this particle has decayed
	      if ( (*vpitr)->end_vertex() != 0 ) // needed some particles have status 2 and no end_vertex 
		{
		  HepLorentzVector dvtx=HepLorentzVector((*vpitr)->end_vertex()->position().x(),
							 (*vpitr)->end_vertex()->position().y(),
							 (*vpitr)->end_vertex()->position().z(),
							 (*vpitr)->end_vertex()->position().t());
		  decay_length=(dvtx-xvtx).rho();
		}
	    }             
	  // end modification
		
	  if( (*vpitr)->status() == 1 || ((*vpitr)->status() == 2 && decay_length > theDecLenCut ) ) {
	  
	    //G4LorentzVector p= (*vpitr)->momentum();
	    HepLorentzVector p = HepLorentzVector((*vpitr)->momentum().px(),
						  (*vpitr)->momentum().py(),
						  (*vpitr)->momentum().pz(),
						  (*vpitr)->momentum().e());
	 
	    
	    if ( !particlePassesPrimaryCuts( p ) ) 
	      {
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
	    if ( (*vpitr)->status() == 2) particleAssignDaughters(g4prim,(HepMC::GenParticle *) *vpitr, decay_length);
	    g4vtx->SetPrimary(g4prim);
	  }
	}

      g4evt->AddPrimaryVertex(g4vtx);
    }
    
  delete evt ;  
    
  return ;
}

void Generator::particleAssignDaughters( G4PrimaryParticle* g4p, HepMC::GenParticle* vp, double decaylength)
{
 
  if ( !(vp->end_vertex())  ) return ;
   
  edm::LogInfo("SimG4CoreGenerator") << "Special case of long decay length" ;
  edm::LogInfo("SimG4CoreGenerator") << "Assign daughters with to mother with decaylength=" << decaylength << "mm";
  HepLorentzVector p(vp->momentum().px(), vp->momentum().py(), vp->momentum().pz(), vp->momentum().e());
  LogDebug("SimG4CoreGenerator") <<" px="<<vp->momentum().px()<<" py="<<vp->momentum().py()<<" pz="<<vp->momentum().pz()<<" e="<<vp->momentum().e();
  LogDebug("SimG4CoreGenerator") <<" p.mag2="<<p.mag2()<<" (p.ee)**2="<<p.e()*p.e()<<" ratio:"<<p.mag2()/p.e()/p.e();
  LogDebug("SimG4CoreGenerator") <<" mass="<<vp->generatedMass()<<" rho="<<p.rho()<<" mag="<<p.mag();
  Hep3Vector cmboost=p.findBoostToCM();
  double proper_time=decaylength/(p.beta()*p.gamma()*c_light);
  LogDebug("SimG4CoreGenerator") <<" beta="<<p.beta()<<" gamma="<<p.gamma()<<" Proper time="
				     <<proper_time<<" ns" ;
  g4p->SetProperTime(proper_time*ns); // the particle will decay after the same length if it has not interacted before
  HepLorentzVector xvtx=HepLorentzVector(vp->end_vertex()->position().x(),
					 vp->end_vertex()->position().y(),
					 vp->end_vertex()->position().z(),
					 vp->end_vertex()->position().t());
  for (HepMC::GenVertex::particle_iterator 
	 vpdec= vp->end_vertex()->particles_begin(HepMC::children);
       vpdec != vp->end_vertex()->particles_end(HepMC::children); ++vpdec) {

    //transform decay products such that in the rest frame of mother
    HepLorentzVector pdec = (HepLorentzVector((*vpdec)->momentum().px(),
					      (*vpdec)->momentum().py(),
					      (*vpdec)->momentum().pz(),
					      (*vpdec)->momentum().e())).boost(cmboost) ;
    // children should only be taken into account once
    G4PrimaryParticle * g4daught= 
      new G4PrimaryParticle((*vpdec)->pdg_id(), pdec.x()*GeV, pdec.y()*GeV, pdec.z()*GeV);
    if ( g4daught->GetG4code() != 0 )
      { 
	g4daught->SetMass( g4daught->GetG4code()->GetPDGMass() ) ;
	g4daught->SetCharge( g4daught->GetG4code()->GetPDGCharge() ) ;  
      }
    g4daught->SetWeight( 10000*(*vpdec)->barcode() ) ;
    setGenId( g4daught, (*vpdec)->barcode() ) ;
    edm::LogInfo("SimG4CoreGenerator") <<" Assigning a "<<(*vpdec)->pdg_id()<<" as daughter of a "
				       <<vp->pdg_id() ;
    if ( (*vpdec)->status() == 2 && (*vpdec)->end_vertex() != 0 ) 
      {
	HepLorentzVector dvtx=HepLorentzVector((*vpdec)->end_vertex()->position().x(),
					       (*vpdec)->end_vertex()->position().y(),
					       (*vpdec)->end_vertex()->position().z(),
					       (*vpdec)->end_vertex()->position().t());
	double dd=(dvtx-xvtx).rho();
	particleAssignDaughters(g4daught,*vpdec,dd);
      }
    (*vpdec)->set_status(1000+(*vpdec)->status()); 
    g4p->SetDaughter(g4daught);
  }
  return;
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
	      HepLorentzVector mom  = HepLorentzVector(g->momentum().px(),
						       g->momentum().py(),
						       g->momentum().pz(),
						       g->momentum().e());
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
		  HepLorentzVector vtx = HepLorentzVector(g->production_vertex()->position().x(),
							  g->production_vertex()->position().y(),
							  g->production_vertex()->position().z(),
							  g->production_vertex()->position().t());
		  G4PrimaryVertex * v = new 
		    G4PrimaryVertex(vtx.x()*mm,vtx.y()*mm,vtx.z()*mm,vtx.t()*mm/c_light);
		  v->SetPrimary(g4p);
		  g4evt->AddPrimaryVertex(v);
		}
	    }
	} // end loop on HepMC particles
    }


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
  fPCuts(p.getParameter<bool>("ApplyPCuts")),
  fEtaCuts(p.getParameter<bool>("ApplyEtaCuts")), 
  fPhiCuts(p.getParameter<bool>("ApplyPhiCuts")),
  theMinPhiCut(p.getParameter<double>("MinPhiCut")),   // now operates in radians (CMS standard)
  theMaxPhiCut(p.getParameter<double>("MaxPhiCut")),
  theMinEtaCut(p.getParameter<double>("MinEtaCut")),
  theMaxEtaCut(p.getParameter<double>("MaxEtaCut")),
  theMinPCut(p.getParameter<double>("MinPCut")),    // now operates in GeV (CMS standard)
  theMaxPCut(p.getParameter<double>("MaxPCut")),   
  theRDecLenCut(p.getParameter<double>("RDecLenCut")*cm),
  theEtaCutForHector(p.getParameter<double>("EtaCutForHector")),
  verbose(p.getUntrackedParameter<int>("Verbosity",0)),
  evt_(0),
  vtx_(0),
  weight_(0),
  Z_lmin(0),
  Z_lmax(0),
  Z_hector(0)
{
  edm::LogInfo("SimG4CoreGenerator") << " Generator constructed " ;

  if(fEtaCuts){
    Z_lmax = theRDecLenCut*( ( 1 - exp(-2*theMaxEtaCut) ) / ( 2*exp(-theMaxEtaCut) ) );
    Z_lmin = theRDecLenCut*( ( 1 - exp(-2*theMinEtaCut) ) / ( 2*exp(-theMinEtaCut) ) );
  }

  Z_hector = theRDecLenCut*( ( 1 - exp(-2*theEtaCutForHector) ) / ( 2*exp(-theEtaCutForHector) ) );

  //  std::cout << "Z_min = " << Z_lmin << " Z_max = " << Z_lmax << " Z_hector = " << Z_hector << std::endl;

}

Generator::~Generator() 
{ 
}

void Generator::HepMC2G4(const HepMC::GenEvent * evt_orig, G4Event * g4evt)
{

  if ( *(evt_orig->vertices_begin()) == 0 )
    {
      throw SimG4Exception( "SimG4CoreGenerator: Corrupted Event - GenEvent with no vertex" ) ;
    }
  
  
  HepMC::GenEvent* evt = new HepMC::GenEvent(*evt_orig) ;
  
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
  
  if (vtx_ != 0) delete vtx_;
  vtx_ = new math::XYZTLorentzVector((*(evt->vertices_begin()))->position().x(),
                                     (*(evt->vertices_begin()))->position().y(),
                                     (*(evt->vertices_begin()))->position().z(),
                                     (*(evt->vertices_begin()))->position().t());
  
  if(verbose >0){
    evt->print();
    cout << " " << endl;
    cout << " Prim.Vtx : " << vtx_->x() << " " 
         << vtx_->y() << " "
         << vtx_->z() << endl;
  }
  
  double x0 = vtx_->x();
  double y0 = vtx_->y();
  
  for(HepMC::GenEvent::vertex_const_iterator vitr= evt->vertices_begin();
      vitr != evt->vertices_end(); ++vitr ) { 
    // loop for vertex ...
    // real vertex?
    G4bool qvtx=false;
    
    for (HepMC::GenVertex::particle_iterator pitr= (*vitr)->particles_begin(HepMC::children);
         pitr != (*vitr)->particles_end(HepMC::children); ++pitr) {
      // Admit also status=1 && end_vertex for long vertex special decay treatment 
      if ((*pitr)->status()==1) {
        qvtx=true;
        break;
      }  
      // The selection is made considering if the partcile with status = 2 have the end_vertex
      // with a radius (R) greater then the theRDecLenCut that means: the end_vertex is outside
      // the beampipe cilinder (no requirement on the Z of the vertex is applyed).
      else if ( (*pitr)->status()== 2 ) {
        if ( (*pitr)->end_vertex() != 0  ) { 
          //double xx = x0-(*pitr)->end_vertex()->position().x();
          //double yy = y0-(*pitr)->end_vertex()->position().y();
          double xx = (*pitr)->end_vertex()->position().x();
          double yy = (*pitr)->end_vertex()->position().y();
          double r_dd=std::sqrt(xx*xx+yy*yy);
          if (r_dd>theRDecLenCut){
            qvtx=true;
            break;
          }
        }
      }
    }


    if (!qvtx) {
      continue;
    }
    
    double x1 = (*vitr)->position().x();
    double y1 = (*vitr)->position().y();
    double z1 = (*vitr)->position().z();
    double t1 = (*vitr)->position().t();	
    G4PrimaryVertex* g4vtx= 
      new G4PrimaryVertex(x1*mm, y1*mm, z1*mm, t1*mm/c_light);
    
    for (HepMC::GenVertex::particle_iterator vpitr= (*vitr)->particles_begin(HepMC::children);
         vpitr != (*vitr)->particles_end(HepMC::children); ++vpitr){

      // Special cases:
      // 1) import in Geant4 a full decay chain (e.g. also particles with status == 2) 
      //    from the generator in case the decay radius is larger than theRDecLenCut
      //    In this case no cuts will be applied to select the particles hat has to be 
      //    processed by geant
      // 2) import in Geant4 particles with status == 1 but a final end vertex. 
      //    The time of the vertex is used as the time of flight to be forced for the particle 
      
      double r_decay_length=-1;
      double decay_length=-1;

      if ( (*vpitr)->status() == 1 || (*vpitr)->status() == 2 ) {
        // this particle has decayed
        if ( (*vpitr)->end_vertex() != 0 ) { 
          // needed some particles have status 2 and no end_vertex 
          // Which are the particles with status 2 and not end_vertex, what are suppose to di such kind of particles
          // when propagated to geant?
          
          double x2 = (*vpitr)->end_vertex()->position().x();
          double y2 = (*vpitr)->end_vertex()->position().y();
          double z2 = (*vpitr)->end_vertex()->position().z();
          r_decay_length=std::sqrt(x2*x2+y2*y2);
          decay_length=std::sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2));

        }
      } 
      // end modification

      bool toBeAdded = false;
      math::XYZTLorentzVector p((*vpitr)->momentum().px(),
                                (*vpitr)->momentum().py(),
                                (*vpitr)->momentum().pz(),
                                (*vpitr)->momentum().e());

      double zimpact = (theRDecLenCut-sqrt(x1*x1+y1*y1))*(1/tan(p.Theta()))+z1;

      // Standard case: particles not decayed by the generator
      if( (*vpitr)->status() == 1 && fabs(zimpact) < Z_hector ) {
        if ( !particlePassesPrimaryCuts( p, zimpact ) ) {
          continue ;
        }
        toBeAdded = true;
      }
      
      // Decay chain entering exiting the fiducial cylinder defined by theRDecLenCut
      else if((*vpitr)->status() == 2 && r_decay_length > theRDecLenCut  && 
         fabs(zimpact) < Z_hector ) {
        toBeAdded=true;
      }
      
      // Particles trasnported along the beam pipe for forward detectors (HECTOR)
      // Always pass to Geant4 without cuts (to be checked)
      else if( (*vpitr)->status() == 1 && fabs(zimpact) >= Z_hector && fabs(z1) >= Z_hector) {
        toBeAdded = true;
      }

      if(toBeAdded){
        
        G4int pdgcode= (*vpitr)-> pdg_id();
        G4PrimaryParticle* g4prim= 
          new G4PrimaryParticle(pdgcode, p.Px()*GeV, p.Py()*GeV, p.Pz()*GeV);
        
        if ( g4prim->GetG4code() != 0 ){ 
          g4prim->SetMass( g4prim->GetG4code()->GetPDGMass() ) ;
          g4prim->SetCharge( g4prim->GetG4code()->GetPDGCharge() ) ;  
        }
        
        g4prim->SetWeight( 10000*(*vpitr)->barcode() ) ;
        setGenId( g4prim, (*vpitr)->barcode() ) ;

        if ( (*vpitr)->status() == 2 ) 
          particleAssignDaughters(g4prim,(HepMC::GenParticle *) *vpitr, decay_length);
        if ( verbose > 1 ) g4prim->Print();
        g4vtx->SetPrimary(g4prim);
        // impose also proper time for status=1 and available end_vertex
        if ( (*vpitr)->status()==1 && (*vpitr)->end_vertex()!=0) {
          double proper_time=decay_length/(p.Beta()*p.Gamma()*c_light);
          //LogDebug("SimG4CoreGenerator") <<" beta="<<p.beta()<<" gamma="<<p.gamma()<<" Proper time=" <<proper_time<<" ns" ;
          g4prim->SetProperTime(proper_time*ns);
        }
      }
    }

    if (verbose > 1 ) g4vtx->Print();
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
  math::XYZTLorentzVector p(vp->momentum().px(), vp->momentum().py(), vp->momentum().pz(), vp->momentum().e());
  LogDebug("SimG4CoreGenerator") <<" px="<<vp->momentum().px()<<" py="<<vp->momentum().py()<<" pz="<<vp->momentum().pz()<<" e="<<vp->momentum().e();
  LogDebug("SimG4CoreGenerator") <<" p.mag2="<<p.mag2()<<" (p.ee)**2="<<p.e()*p.e()<<" ratio:"<<p.mag2()/p.e()/p.e();
  LogDebug("SimG4CoreGenerator") <<" mass="<<vp->generatedMass()<<" rho="<<p.P()<<" mag="<<p.mag();
  double proper_time=decaylength/(p.Beta()*p.Gamma()*c_light);
  LogDebug("SimG4CoreGenerator") <<" beta="<<p.Beta()<<" gamma="<<p.Gamma()<<" Proper time="
                                 <<proper_time<<" ns" ;
  g4p->SetProperTime(proper_time*ns); // the particle will decay after the same length if it has not interacted before
  double x1 = vp->end_vertex()->position().x();
  double y1 = vp->end_vertex()->position().y();
  double z1 = vp->end_vertex()->position().z();
  for (HepMC::GenVertex::particle_iterator 
         vpdec= vp->end_vertex()->particles_begin(HepMC::children);
       vpdec != vp->end_vertex()->particles_end(HepMC::children); ++vpdec) {
    
    //transform decay products such that in the rest frame of mother
    math::XYZTLorentzVector pdec((*vpdec)->momentum().px(),
                                 (*vpdec)->momentum().py(),
                                 (*vpdec)->momentum().pz(),
                                 (*vpdec)->momentum().e());
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
        double x2 = (*vpdec)->end_vertex()->position().x();
        double y2 = (*vpdec)->end_vertex()->position().y();
        double z2 = (*vpdec)->end_vertex()->position().z();
        double dd = std::sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2));
        particleAssignDaughters(g4daught,*vpdec,dd);
      }
    (*vpdec)->set_status(1000+(*vpdec)->status()); 
    g4p->SetDaughter(g4daught);
  }
  return;
}

bool Generator::particlePassesPrimaryCuts( const math::XYZTLorentzVector& mom, const double zimp ) const 
{
  
  double phi = mom.Phi() ;   
  double nrg  = mom.P() ;

  if ( (fEtaCuts) && (zimp < Z_lmin       || zimp > Z_lmax)       ) return false ;
  if ( (fPCuts)  && (nrg   < theMinPCut   || nrg  > theMaxPCut)   ) return false ;
  if ( (fPhiCuts) && (phi  < theMinPhiCut || phi  > theMaxPhiCut) ) return false ;

  //  std::cout << "Passed p=" << mom.P() << " p_t=" << mom.Pt() << " z_imp=" << zimp << " eta=" << mom.Eta() << " theta=" << mom.Theta() << std::endl;
  
  return true;   
}

bool Generator::particlePassesPrimaryCuts(const G4PrimaryParticle * p) const
{

  G4ThreeVector mom = p->GetMomentum();
  double        phi = mom.phi() ;
  double        nrg  = sqrt(p->GetPx()*p->GetPx() + p->GetPy()*p->GetPy() + p->GetPz()*p->GetPz());
  nrg /= GeV ;  // need to convert, since Geant4 operates in MeV
  double        eta = -log(tan(mom.theta()/2));

  if ( (fEtaCuts) && (eta < theMinEtaCut || eta > theMaxEtaCut) ) return false ;
  if ( (fPCuts)  &&  (nrg  < theMinPCut  || nrg > theMaxPCut)   ) return false ;
  if ( (fPhiCuts) && (phi < theMinPhiCut || phi > theMaxPhiCut) ) return false ;

  return true;

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
	  int g_id = g->pdg_id();	    
	  G4PrimaryParticle * g4p = 
	    new G4PrimaryParticle(g_id,g->momentum().px()*GeV,g->momentum().py()*GeV,g->momentum().pz()*GeV);
	  if (g4p->GetG4code() != 0)
	    { 
	      g4p->SetMass(g4p->GetG4code()->GetPDGMass());
	      g4p->SetCharge(g4p->GetG4code()->GetPDGCharge()) ;
	    }
	  g4p->SetWeight(i*10000);
	  setGenId(g4p,i);
	  if (particlePassesPrimaryCuts(g4p))
	    {
	      G4PrimaryVertex * v = new 
		G4PrimaryVertex(g->production_vertex()->position().x()*mm,	     
				g->production_vertex()->position().y()*mm,	     
				g->production_vertex()->position().z()*mm,	     
				g->production_vertex()->position().t()*mm/c_light);
	      v->SetPrimary(g4p);
	      g4evt->AddPrimaryVertex(v);
	    }
	}
    } // end loop on HepMC particles
}

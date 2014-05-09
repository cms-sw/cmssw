#include "SimG4Core/Generators/interface/Generator.h"
#include "SimG4Core/Generators/interface/HepMCParticle.h"

#include "SimG4Core/Notification/interface/SimG4Exception.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Event.hh"

#include "G4HEPEvtParticle.hh"
#include "G4ParticleDefinition.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

using namespace edm;
//using std::cout;
//using std::endl;

Generator::Generator(const ParameterSet & p) : 
  fPCuts(p.getParameter<bool>("ApplyPCuts")),
  fPtransCut(p.getParameter<bool>("ApplyPtransCut")),
  fEtaCuts(p.getParameter<bool>("ApplyEtaCuts")), 
  fPhiCuts(p.getParameter<bool>("ApplyPhiCuts")),
  theMinPhiCut(p.getParameter<double>("MinPhiCut")), // in radians (CMS standard)
  theMaxPhiCut(p.getParameter<double>("MaxPhiCut")),
  theMinEtaCut(p.getParameter<double>("MinEtaCut")),
  theMaxEtaCut(p.getParameter<double>("MaxEtaCut")),
  theMinPCut(p.getParameter<double>("MinPCut")),    // in GeV (CMS standard)
  theMaxPCut(p.getParameter<double>("MaxPCut")),   
  theEtaCutForHector(p.getParameter<double>("EtaCutForHector")),
  verbose(p.getUntrackedParameter<int>("Verbosity",0)),
  evt_(0),
  vtx_(0),
  weight_(0),
  Z_lmin(0),
  Z_lmax(0),
  Z_hector(0),
  pdgFilterSel(false) 
{
  double theRDecLenCut = p.getParameter<double>("RDecLenCut")*cm;
  theRDecLenCut2 = theRDecLenCut*theRDecLenCut;
  theMinPtCut2 = theMinPCut*theMinPCut;

  pdgFilter.resize(0);
  if ( p.exists("PDGselection") ) {

    pdgFilterSel = 
      (p.getParameter<edm::ParameterSet>("PDGselection")).getParameter<bool>("PDGfilterSel");
    pdgFilter = 
      (p.getParameter<edm::ParameterSet>("PDGselection")).getParameter<std::vector< int > >("PDGfilter");
    for ( unsigned int ii = 0; ii < pdgFilter.size(); ++ii) {
      if (pdgFilterSel) {
        edm::LogWarning("SimG4CoreGenerator") << " *** Selecting only PDG ID = " 
					      << pdgFilter[ii];
      } else {
        edm::LogWarning("SimG4CoreGenerator") << " *** Filtering out PDG ID = " 
					      << pdgFilter[ii];
      }
    }
  }

  if(fEtaCuts){
    Z_lmax = theRDecLenCut*((1 - exp(-2*theMaxEtaCut) )/( 2*exp(-theMaxEtaCut)));
    Z_lmin = theRDecLenCut*((1 - exp(-2*theMinEtaCut) )/( 2*exp(-theMinEtaCut)));
  }

  Z_hector = theRDecLenCut*((1 - exp(-2*theEtaCutForHector)) 
			    / ( 2*exp(-theEtaCutForHector) ) );

  edm::LogInfo("SimG4CoreGenerator") 
    << "Z_min = " << Z_lmin << " Z_max = " << Z_lmax << " Z_hector = " << Z_hector;
  if(0 < verbose) LogDebug("SimG4CoreGenerator")
    << "Z_min = " << Z_lmin << " Z_max = " << Z_lmax << " Z_hector = " << Z_hector;
}

Generator::~Generator() 
{}

void Generator::HepMC2G4(const HepMC::GenEvent * evt_orig, G4Event * g4evt)
{

  if ( *(evt_orig->vertices_begin()) == 0 ) {
    throw SimG4Exception("SimG4CoreGenerator: Corrupted Event - GenEvent with no vertex");
  }  
  
  HepMC::GenEvent* evt = new HepMC::GenEvent(*evt_orig);
  
  if (evt->weights().size() > 0) {

    weight_ = evt->weights()[0] ;
    for (unsigned int iw=1; iw<evt->weights().size(); ++iw) {

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
  
  if(verbose > 0) {
    evt->print();
    LogDebug("SimG4CoreGenerator") << "Primary Vertex = (" 
				   << vtx_->x() << "," 
				   << vtx_->y() << ","
				   << vtx_->z() << ")";
  }
    
  unsigned int ng4vtx = 0;

  for(HepMC::GenEvent::vertex_const_iterator vitr= evt->vertices_begin();
      vitr != evt->vertices_end(); ++vitr ) { 

    // loop for vertex ...
    // real vertex?
    G4bool qvtx=false;
    HepMC::GenVertex::particle_iterator pitr;
    for (pitr= (*vitr)->particles_begin(HepMC::children);
         pitr != (*vitr)->particles_end(HepMC::children); ++pitr) {

      // Admit also status=1 && end_vertex for long vertex special decay treatment 
      if (1 == (*pitr)->status()) {

        qvtx = true;
        if (verbose > 1) LogDebug("SimG4CoreGenerator") 
	  << "GenVertex barcode = " << (*vitr)->barcode() 
	  << " selected for GenParticle barcode = " << (*pitr)->barcode();
        break;
      }  
      // The selection is made considering if the partcile with status = 2 
      // have the end_vertex with a radius (R) greater then the theRDecLenCut 
      // that means: the end_vertex is outside the beampipe cilinder 
      // (no requirement on the Z of the vertex is applyed).
      else if (2 == (*pitr)->status()) {

        if ( (*pitr)->end_vertex() != 0  ) { 
          double xx = (*pitr)->end_vertex()->position().x();
          double yy = (*pitr)->end_vertex()->position().y();
          double r_dd = xx*xx+yy*yy;
          if (r_dd > theRDecLenCut2){
            qvtx = true;
            if (verbose > 1) LogDebug("SimG4CoreGenerator") 
	      << "GenVertex barcode = " << (*vitr)->barcode() 
	      << " selected for GenParticle barcode = " 
	      << (*pitr)->barcode() << " radius = " << std::sqrt(r_dd);
            break;
          }
        } else {
	  // particles with stus 2 without end_vertex are equivalent to stable
	  qvtx = true;
	}
      }
    }

    // if this vertex has no long-lived secondary the vertex is not saved 
    if (!qvtx) {
      continue;
    }
    
    double x1 = (*vitr)->position().x()*mm;
    double y1 = (*vitr)->position().y()*mm;
    double z1 = (*vitr)->position().z()*mm;
    double t1 = (*vitr)->position().t()*mm/c_light;

    G4PrimaryVertex* g4vtx = new G4PrimaryVertex(x1, y1, z1, t1);
    
    for (pitr= (*vitr)->particles_begin(HepMC::children);
         pitr != (*vitr)->particles_end(HepMC::children); ++pitr){

      // Filter on allowed particle species if required      
      if ( pdgFilter.size() > 0 ) {
        std::vector<int>::iterator it = find(pdgFilter.begin(),pdgFilter.end(),
					     (*pitr)->pdg_id()); 
        if ( (it != pdgFilter.end() && !pdgFilterSel) 
	     || ( it == pdgFilter.end() && pdgFilterSel ) ) {
          if (verbose > 1) LogDebug("SimG4CoreGenerator") 
	    << "Skip GenParticle barcode = " << (*pitr)->barcode() 
	    << " PDG Id = " << (*pitr)->pdg_id();
          continue;
        }
      }

      // Special cases:
      // 1) import in Geant4 a full decay chain (e.g. also particles with status == 2) 
      //    from the generator in case the decay radius is larger than theRDecLenCut
      //    In this case no cuts will be applied to select the particles hat has to be 
      //    processed by geant
      // 2) import in Geant4 particles with status == 1 but a final end vertex. 
      //    The time of the vertex is used as the time of flight to be forced for the particle 
      
      double x2 = 0.0;
      double y2 = 0.0;
      double z2 = 0.0;
      double decay_length = 0.0;
      int status = (*pitr)->status();

      if (1 == status || 2 == status) {

        // this particle has decayed but have no vertex
	// it is an exotic case
        if ( !(*pitr)->end_vertex() ) { 
          status = 1; 
	} else {
          x2 = (*pitr)->end_vertex()->position().x();
          y2 = (*pitr)->end_vertex()->position().y();
          z2 = (*pitr)->end_vertex()->position().z();
	  decay_length = 
	      std::sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
	}
      } 

      bool toBeAdded = false;
      double px = (*pitr)->momentum().px();
      double py = (*pitr)->momentum().py();
      double pz = (*pitr)->momentum().pz();
      double ptot = std::sqrt(px*px + py*py + pz*pz);
      math::XYZTLorentzVector p(px, py, pz, (*pitr)->momentum().e());
      
      double ximpact = x1;
      double yimpact = y1;
      double zimpact = z1;

      // protection against numerical problems for extremely low momenta
      const double minTan = 1.e-20;
      if(fabs(z1) < Z_hector && fabs(pz) >= minTan*ptot) {
        if(pz > 0.0) { zimpact =  Z_hector; }
	else         { zimpact = -Z_hector; }
        double del = (zimpact - z1)/pz;
        ximpact += del*px;
        yimpact += del*py;
      }
      double rimpact2 = ximpact*ximpact + yimpact*yimpact;
      
      if (verbose > 1) LogDebug("SimG4CoreGenerator") 
	<< "Processing GenParticle barcode= " << (*pitr)->barcode() 
	<< " status= " << (*pitr)->status() 
	<< " st= " << status 
	<< " rimpact(cm)= " << std::sqrt(rimpact2)/cm
	<< " zimpact(cm)= " << zimpact/cm
	<< " ptot(GeV)= " << ptot;

      // Particles trasnported along the beam pipe for forward detectors (HECTOR)
      // Always pass to Geant4 without cuts (to be checked)
      if( 1 == status && 
	  fabs(zimpact) >= Z_hector && rimpact2 <= theRDecLenCut2) {
        toBeAdded = true;
        if ( verbose > 1 ) LogDebug("SimG4CoreGenerator") 
	  << "GenParticle barcode = " << (*pitr)->barcode() 
	  << " passed case 3";
      } else {

        double decay_length = 0.0;
	// Standard case: particles not decayed by the generator
	// or particle decying without end vertex
	if(1 == status && 
	   (fabs(zimpact) < Z_hector || rimpact2 > theRDecLenCut2)) {

	  // Ptot cut (was assumed to be Pt?)
	  if (fPCuts && (ptot < theMinPCut || ptot > theMaxPCut)) {
            continue;
	  }
          // phi cut
	  if (fPhiCuts) {
            double phi = p.phi();
	    if(phi < theMinPhiCut || phi  > theMaxPhiCut) {
	      continue;
	    }
	  }

	  // eta cut
	  double xi = x1;
	  double yi = y1;
	  double zi = z1;

	  // can be propagated along Z
	  if(fabs(pz) >= minTan*ptot) {
	    if((zi >= Z_lmax) & (pz < 0.0)) {
              zi = Z_lmax;
	    } else if((zi <= Z_lmin) & (pz > 0.0)) {
              zi = Z_lmin;
	    } else {
              if(pz > 0) { zi = Z_lmax; }
              else { zi = Z_lmin; }
	    }
	    double del = (zi - z1)/pz;
	    xi += del*px;
	    yi += del*py;
	  }
	  // check eta cut
	  if((zi >= Z_lmin) & (zi <= Z_lmax) 
	     & (xi*xi + yi*yi < theRDecLenCut2)) {
	    continue;
	  }
	  toBeAdded = true;
	  if ( verbose > 1 ) LogDebug("SimG4CoreGenerator") 
	    << "GenParticle barcode = " << (*pitr)->barcode() 
	    << " passed case 1";
	} else if(2 == status && 
		  x2*x2 + y2*y2 >= theRDecLenCut2 && fabs(z2) < Z_hector){

	  // Decay chain entering exiting the fiducial cylinder 
	  // defined by theRDecLenCut
	  toBeAdded=true;

	  if ( verbose > 1 ) LogDebug("SimG4CoreGenerator") 
	    << "GenParticle barcode = " << (*pitr)->barcode() 
	    << " passed case 2" 
	    << " decay_length(cm)= " << decay_length/cm;
	}
      }
      if(toBeAdded){
        
        G4int pdgcode= (*pitr)-> pdg_id();
        G4PrimaryParticle* g4prim= 
          new G4PrimaryParticle(pdgcode, px*GeV, py*GeV, pz*GeV);
        
        if ( g4prim->GetG4code() != 0 ){ 
          g4prim->SetMass( g4prim->GetG4code()->GetPDGMass() );
          double charge = g4prim->GetG4code()->GetPDGCharge();

	  // apply Pt cut
	  if (fPtransCut && 
	      0.0 != charge && px*px + py*py < theMinPtCut2) {
            delete g4prim;
            continue;
	  }
          g4prim->SetCharge(charge);  
        }

	// V.I. do not use SetWeight but the same code
        // value of the code compute inside TrackWithHistory        
        //g4prim->SetWeight( 10000*(*vpitr)->barcode() ) ;
        setGenId( g4prim, (*pitr)->barcode() );

        if (2 == status) {
          particleAssignDaughters(g4prim,
				  (HepMC::GenParticle *) *pitr, decay_length);
	}
        if ( verbose > 1 ) g4prim->Print();
        g4vtx->SetPrimary(g4prim);

        // impose also proper time for status=1 and available end_vertex
        if ( 1 == status && decay_length > 0.0) {
          double proper_time = decay_length/(p.Beta()*p.Gamma()*c_light);
          if ( verbose > 1 ) LogDebug("SimG4CoreGenerator") 
	    <<"Setting proper time for beta="<<p.Beta()<<" gamma="
	    <<p.Gamma()<<" Proper time=" <<proper_time/ns<<" ns" ;
          g4prim->SetProperTime(proper_time);
        }
      }
    }

    if (verbose > 1 ) g4vtx->Print();
    g4evt->AddPrimaryVertex(g4vtx);
    ++ng4vtx;
  }
  
  // Add a protection for completely empty events (produced by LHCTransport): 
  // add a dummy vertex with no particle attached to it
  if ( ng4vtx == 0 ) {
    G4PrimaryVertex* g4vtx = new G4PrimaryVertex(0.0, 0.0, 0.0, 0.0);
    if (verbose > 1 ) g4vtx->Print();
    g4evt->AddPrimaryVertex(g4vtx);
  }
  
  delete evt;
}

void Generator::particleAssignDaughters( G4PrimaryParticle* g4p, 
					 HepMC::GenParticle* vp, double decaylength)
{
  // not needed anymore
  //if (!(vp->end_vertex())) return;
   
  if ( verbose > 1 ) 
    LogDebug("SimG4CoreGenerator") 
      << "Special case of long decay length \n" 
      << "Assign daughters with to mother with decaylength=" 
      << decaylength/cm << " cm";
  math::XYZTLorentzVector p(vp->momentum().px(), vp->momentum().py(), 
			    vp->momentum().pz(), vp->momentum().e());
  double proper_time = decaylength/(p.Beta()*p.Gamma()*c_light);
  if( verbose > 1 ) {
    LogDebug("SimG4CoreGenerator") <<" px= "<< p.px()
				   <<" py= "<< p.py()
				   <<" pz= "<< p.pz()
				   <<" e= " <<  p.e()
				   <<" beta= "<< p.Beta()
				   <<" gamma= " << p.Gamma()
				   <<" Proper time= " <<proper_time/ns <<" ns";
  }
  g4p->SetProperTime(proper_time); 
  // the particle will decay after the same length if it 
  // has not interacted before
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
    // V.I. do not use SetWeight but the same code
    // value of the code compute inside TrackWithHistory        
    //g4daught->SetWeight( 10000*(*vpdec)->barcode() ) ;
    setGenId( g4daught, (*vpdec)->barcode() );

    if ( verbose > 1 ) LogDebug("SimG4CoreGenerator") 
      <<"Assigning a "<<(*vpdec)->pdg_id()
      <<" as daughter of a " <<vp->pdg_id();
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

bool Generator::particlePassesPrimaryCuts(const G4PrimaryParticle * p) const 
{

  G4ThreeVector mom = p->GetMomentum();
  double        phi = mom.phi();
  double        nrg = mom.mag()/GeV;

  double        eta = -log(0.5*tan(mom.theta()));
  bool   flag = true;

  if ( (fEtaCuts) && (eta < theMinEtaCut || eta > theMaxEtaCut) ) {
    flag = false;
  } else if ( (fPCuts)  &&  (nrg  < theMinPCut  || nrg > theMaxPCut)   ) {
    flag = false;
  } else if ( (fPhiCuts) && (phi < theMinPhiCut || phi > theMaxPhiCut) ) {
    flag = false;
  }
  
  if ( verbose > 1 ) LogDebug("SimG4CoreGenerator") 
    << "Generator p = " << nrg  << " eta = " << eta 
    << " theta = " << mom.theta() << " phi = " << phi << " Flag = " << flag;

  return flag;
}

void Generator::nonBeamEvent2G4(const HepMC::GenEvent * evt, G4Event * g4evt) 
{
  int i = 0; 
  for(HepMC::GenEvent::particle_const_iterator it = evt->particles_begin(); 
      it != evt->particles_end(); ++it ) {
    ++i;
    HepMC::GenParticle * g = (*it);	
    int g_status = g->status();
    // storing only particle with status == 1 	
    if (g_status == 1) {
      int g_id = g->pdg_id();	    
      G4PrimaryParticle * g4p = 
	new G4PrimaryParticle(g_id,g->momentum().px()*GeV,
			      g->momentum().py()*GeV,g->momentum().pz()*GeV);
      if (g4p->GetG4code() != 0) { 
	g4p->SetMass(g4p->GetG4code()->GetPDGMass());
	g4p->SetCharge(g4p->GetG4code()->GetPDGCharge()) ;
      }
      // V.I. do not use SetWeight but the same code
      // value of the code compute inside TrackWithHistory        
      //g4p->SetWeight(i*10000);
      setGenId(g4p,i);
      if (particlePassesPrimaryCuts(g4p)) {
	G4PrimaryVertex * v = 
	  new G4PrimaryVertex(g->production_vertex()->position().x()*mm,
			      g->production_vertex()->position().y()*mm,
			      g->production_vertex()->position().z()*mm,
			      g->production_vertex()->position().t()*mm/c_light);
	v->SetPrimary(g4p);
	g4evt->AddPrimaryVertex(v);
	if(verbose >0) {
	  v->Print();
	}
      }
    }
  } // end loop on HepMC particles
}

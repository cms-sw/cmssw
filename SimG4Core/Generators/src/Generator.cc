#include "SimG4Core/Generators/interface/Generator.h"
#include "SimG4Core/Generators/interface/HepMCParticle.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4HEPEvtParticle.hh"
#include "G4ParticleDefinition.hh"
#include "G4UnitsTable.hh"

#include "CLHEP/HepMC/ReadHepMC.h"
 
#include <fstream> 

using namespace edm;
using std::cout;
using std::endl;
using std::string;

std::ifstream * inputFile;

Generator::Generator(const ParameterSet & p) : 
    fPtCuts(p.getParameter<bool>("ApplyPtCuts")),
    fEtaCuts(p.getParameter<bool>("ApplyEtaCuts")), 
    fPhiCuts(p.getParameter<bool>("ApplyPhiCuts")),
    theMinPhiCut(p.getParameter<double>("MinPhiCut")*deg),
    theMaxPhiCut(p.getParameter<double>("MaxPhiCut")*deg),
    theMinEtaCut(p.getParameter<double>("MinEtaCut")),
    theMaxEtaCut(p.getParameter<double>("MaxEtaCut")),
    theMinPtCut(p.getParameter<double>("MinPtCut")*MeV),
    theMaxPtCut(p.getParameter<double>("MaxPtCut")*MeV),
    inputFileName(p.getParameter<std::string>("InputFileName")),
    verbose(p.getParameter<int>("Verbosity")),
    evt_(0),vtx_(HepLorentzVector(0.,0.,0.,0.)),weight_(0),runNumber_(0)
{
    if ( inputFileName == "Internal" )
    {
       inputFile = 0 ;
    }
    else
    {
       inputFile = new std::ifstream(inputFileName.c_str(),std::ios::in);
       if(!*inputFile) {
	  delete inputFile;
	  throw cms::Exception("FailedFileOpen")<<" Unable to open the generator file "<<inputFileName<<".\n  Please check to see if file name correct for parameter 'InputFileName'";
       }
    }
    std::cout << " Generator constructed " << std::endl;
}

Generator::~Generator() 
{ 
   if ( inputFile != 0 ) delete inputFile; 
}

const HepMC::GenEvent * Generator::generateEvent()
{
    if ( inputFile != 0 )
    {
       evt_ = HepMC::readGenEvent(*inputFile);
    }
    // vtx_ = HepLorentzVector(0.,0.,0.,0.);
    if (verbose>0) evt_->print();
    return evt_;
}

void Generator::HepMC2G4(const HepMC::GenEvent * evt, G4Event * g4evt)
{	
    int i = 0; 
    for(HepMC::GenEvent::particle_const_iterator it = evt->particles_begin(); 
	it != evt->particles_end(); ++it )
    {
	i++;
	HepMC::GenParticle * g = (*it);	
	int g_status = g->status();
	// storing only particle with status == 1 or status ==2 	
	if (g_status == 1 || g_status == 2)
	{
	    HepLorentzVector g_momentum  = g->momentum();
	    int g_id = g->pdg_id();	    
	    G4PrimaryParticle * g4p = 
		new G4PrimaryParticle(g_id,g_momentum.x()*GeV,g_momentum.y()*GeV,g_momentum.z()*GeV);
	    if (g4p->GetG4code() != 0) g4p->SetMass(g4p->GetG4code()->GetPDGMass());
	    g4p->SetWeight(i*10000);
	    setGenId(g4p,i);

	    // for decays, assign the pre-defined decay time
	    // derive the decay proper time from the decay length
	    // tau = L/(gamma*beta*c) 
	    if (g->production_vertex() != 0 && g->end_vertex() != 0 && g_status == 2)
	    {
		double xm = g->production_vertex()->position().x();
		double ym = g->production_vertex()->position().y();
		double zm = g->production_vertex()->position().z();
		double xd = g->end_vertex()->position().x();
		double yd = g->end_vertex()->position().y();
		double zd = g->end_vertex()->position().z();
		double decl = sqrt((xd-xm)*(xd-xm)+(yd-ym)*(yd-ym)+(zd-zm)*(zd-zm));
		double labTime = decl/c_light;
		// convert lab time to proper time
		double properTime = labTime/g_momentum.rho()*(g4p->GetMass()/1000);
		// set the proper time in nanoseconds
		g4p->SetProperTime(properTime);
	    } 
	    // this is for bookeeping of the particle status, which is not in g4p 
	    HepMCParticle * hepmcp = new HepMCParticle(g4p,g_status);
	    pmap[g] = hepmcp;
					
	}
    } // end loop on HepMC particles
	
    if (pmap.size() == 0) return; 
		
    // set status to -1 for stable particles failing primary cuts
    // => they will not be passed to G4
    for (PMT it = pmap.begin();  it != pmap.end(); ++it)
    {
        if ((*it).first->status() == 1 && !(particlePassesPrimaryCuts((*it).second->getTheParticle())))
            (*it).second->done();
        // we're done with that guy, remember this in hepmcparticle
    }
         
    // make G4 connection between daughter particles from same mother
    for (PMT it = pmap.begin(); it != pmap.end(); ++it)
    {
	HepMC::GenParticle * g = (*it).first;
	G4PrimaryParticle  * p = (*it).second->getTheParticle();
	if (g->hasChildren())
	{
	    for (HepMC::GenVertex::particle_iterator 
		     ic = g->end_vertex()->particles_begin(HepMC::children);
		 ic != g->end_vertex()->particles_end(HepMC::children); ++ic) 
	    { 
		if ((*it).second->getStatus() >0)
		{
		    PMT id = pmap.find((*ic));
		    if (id != pmap.end())
		    {	
			G4PrimaryParticle * d = (*id).second->getTheParticle();
			p->SetDaughter(d);
			// we have the daughtes of that guy =>
			// don't do it a second time, therefore 
			// flag in map to remember this
			(*it).second->done();			      
		    }
		}
	    }
	}	
    } // end PMT it = pmap.begin();
	
    // create G4PrimaryVertex and associated G4PrimaryParticles
    G4PrimaryVertex * g4vtx = new G4PrimaryVertex(vtx_.x()*mm,vtx_.y()*mm,vtx_.z()*mm,vtx_.t()*mm/c_light);
    // assign vertex particles
    for (PMT it = pmap.begin(); it != pmap.end(); ++it) 
    {
	// negative status for daughters and primaries not passing primary cuts
	if ((*it).second->getStatus()>0) 
	{
	    g4vtx->SetPrimary((*it).second->getTheParticle());        
	}
    }
    g4evt->AddPrimaryVertex(g4vtx);
	
    pmap.clear();
}

bool Generator::particlePassesPrimaryCuts(const G4PrimaryParticle * p) const
{
    G4ThreeVector mom = p->GetMomentum();
    double        phi = mom.phi()+180.*deg;
    double        pt  = sqrt(p->GetPx()*p->GetPx() + p->GetPy()*p->GetPy());
    double        eta = -log(tan(mom.theta()/2));
    if (((fPtCuts)  && (pt  < theMinPtCut  || pt  > theMaxPtCut))           ||
	((fEtaCuts) && (eta < theMinEtaCut || eta > theMaxEtaCut))          ||
	((fPhiCuts) && (phi < theMinPhiCut || phi > theMaxPhiCut)))
	return false;
    else return true;
}

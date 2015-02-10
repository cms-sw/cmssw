/*class BasicHepMCHeavyIonValidation
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *  Quan Wang - 04/2013
 */

#include "Validation/EventGenerator/interface/BasicHepMCHeavyIonValidation.h"

#include "CLHEP/Units/defs.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "Validation/EventGenerator/interface/DQMHelper.h"

using namespace edm;

BasicHepMCHeavyIonValidation::BasicHepMCHeavyIonValidation(const edm::ParameterSet& iPSet): 
  wmanager_(iPSet,consumesCollector()),
	hepmcCollection_(iPSet.getParameter<edm::InputTag>("hepmcCollection"))
{    
	QWdebug_ = iPSet.getUntrackedParameter<bool>("QWdebug",false);

	hepmcCollectionToken_=consumes<HepMCProduct>(hepmcCollection_);
}

BasicHepMCHeavyIonValidation::~BasicHepMCHeavyIonValidation() {}

void BasicHepMCHeavyIonValidation::bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &){

  ///Setting the DQM top directories
  DQMHelper dqm(&i); i.setCurrentFolder("Generator/HeavyIon");
  
  // Number of analyzed events
  nEvt = dqm.book1dHisto("nEvt", "n analyzed Events", 1, 0., 1.);
  
  ///Booking the ME's
  Ncoll_hard = dqm.book1dHisto("Ncoll_hard", "Ncoll_hard", 700, 0, 700);
  Npart_proj = dqm.book1dHisto("Npart_proj", "Npart_proj", 250, 0, 250);
  Npart_targ = dqm.book1dHisto("Npart_targ", "Npart_targ", 250, 0, 250);
  Ncoll = dqm.book1dHisto("Ncoll", "Ncoll", 700, 0, 700);
  N_Nwounded_collisions = dqm.book1dHisto("N_Nwounded_collisions", "N_Nwounded_collisions", 250, 0, 250);
  Nwounded_N_collisions = dqm.book1dHisto("Nwounded_N_collisions", "Nwounded_N_collisions", 250, 0, 250);
  Nwounded_Nwounded_collisions = dqm.book1dHisto("Nwounded_Nwounded_collisions", "Nwounded_Nwounded_collisions", 250, 0, 250);
  spectator_neutrons = dqm.book1dHisto("spectator_neutrons", "spectator_neutrons", 250, 0, 250);
  spectator_protons = dqm.book1dHisto("spectator_protons", "spectator_protons", 250, 0, 250);
  impact_parameter = dqm.book1dHisto("impact_parameter", "impact_parameter", 50, 0, 50);
  event_plane_angle = dqm.book1dHisto("event_plane_angle", "event_plane_angle", 200, -CLHEP::pi, CLHEP::pi);
  eccentricity = dqm.book1dHisto("eccentricity", "eccentricity", 200, 0, 1.0);
  sigma_inel_NN = dqm.book1dHisto("sigma_inel_NN", "sigma_inel_NN", 200, 0, 10.0);
  
  return;
}

void BasicHepMCHeavyIonValidation::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup){ 
	///counters to zero for every event

	///Gathering the HepMCProduct information
	edm::Handle<HepMCProduct> evt;
	iEvent.getByToken(hepmcCollectionToken_, evt);

	//Get EVENT
	//HepMC::GenEvent *myGenEvent = new HepMC::GenEvent(*(evt->GetEvent()));



	const HepMC::HeavyIon* ion = evt->GetEvent()->heavy_ion();

	if (!ion) {
		if ( QWdebug_ ) std::cout << "!!QW!! HeavyIon == null" << std::endl;
		return;
	}

	double weight = wmanager_.weight(iEvent);
	nEvt->Fill(0.5,weight);

	Ncoll_hard->Fill(ion->Ncoll_hard(), weight);
	Npart_proj->Fill(ion->Npart_proj(), weight);
	Npart_targ->Fill(ion->Npart_targ(), weight);
	Ncoll->Fill(ion->Ncoll(), weight);
	N_Nwounded_collisions->Fill(ion->N_Nwounded_collisions(), weight);
	Nwounded_N_collisions->Fill(ion->Nwounded_N_collisions(), weight);
	Nwounded_Nwounded_collisions->Fill(ion->Nwounded_Nwounded_collisions(), weight);
	spectator_neutrons->Fill(ion->spectator_neutrons(), weight);
	spectator_protons->Fill(ion->spectator_protons(), weight);
	impact_parameter->Fill(ion->impact_parameter(), weight);
	event_plane_angle->Fill(ion->event_plane_angle(), weight);
	eccentricity->Fill(ion->eccentricity(), weight);
	sigma_inel_NN->Fill(ion->sigma_inel_NN(), weight);


	//delete myGenEvent;
}//analyze

#ifndef BASICHEPMCHEAVYIONVALIDATION_H
#define BASICHEPMCHEAVYIONVALIDATION_H

/*class BasicHepMCHeavyIonValidation
 *  
 *  Class to fill Event Generator dqm monitor elements; works on HepMCProduct HepMC::HeavyIon
 *  Quan Wang - 04/2013
 *
 */

// framework & common header files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"


#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "Validation/EventGenerator/interface/WeightManager.h"

class BasicHepMCHeavyIonValidation : public DQMEDAnalyzer {
	public:
		explicit BasicHepMCHeavyIonValidation(const edm::ParameterSet&);
		~BasicHepMCHeavyIonValidation() override;

		void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;
		void analyze(edm::Event const&, edm::EventSetup const&) override;

	private:
		WeightManager wmanager_;
		edm::InputTag hepmcCollection_;
		bool QWdebug_;

		/// PDT table
		//edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;

		MonitorElement* nEvt;

		// Additional information stored in HeavyIon structure
		MonitorElement* Ncoll_hard;		// Number of hard scatterings
		MonitorElement* Npart_proj;		// Number of projectile participants
		MonitorElement* Npart_targ;		// Number of target participants
		MonitorElement* Ncoll;			// Number of NN (nucleon-nucleon) collisions
		MonitorElement* N_Nwounded_collisions;		// Number of N-Nwounded collisions
		MonitorElement* Nwounded_N_collisions;		// Number of Nwounded-N collisons
		MonitorElement* Nwounded_Nwounded_collisions;	// Number of Nwounded-Nwounded collisions
		MonitorElement* spectator_neutrons;		// Number of spectator neutrons
		MonitorElement* spectator_protons;		// Number of spectator protons
		MonitorElement* impact_parameter;		// Impact Parameter(fm) of collision
		MonitorElement* event_plane_angle;		// Azimuthal angle of event plane
		MonitorElement* eccentricity;		// eccentricity of participating nucleons
							// in the transverse plane 
							// (as in phobos nucl-ex/0510031) 
		MonitorElement* sigma_inel_NN;		// nucleon-nucleon inelastic 
							// (including diffractive) cross-section

		edm::EDGetTokenT<edm::HepMCProduct> hepmcCollectionToken_;
};

#endif

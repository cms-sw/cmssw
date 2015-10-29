#include "Validation/CaloTowers/interface/CaloTowersValidation.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

CaloTowersValidation::CaloTowersValidation(edm::ParameterSet const& conf)
{

  tok_calo_ = consumes<CaloTowerCollection>(conf.getUntrackedParameter<edm::InputTag>("CaloTowerCollectionLabel"));
    tok_evt_ = consumes<edm::HepMCProduct>(edm::InputTag("generatorSmeared"));

  // DQM ROOT output
  outputFile_ = conf.getUntrackedParameter<std::string>("outputFile", "myfile.root");

  hcalselector_ = conf.getUntrackedParameter<std::string>("hcalselector", "all");

  mc_           = conf.getUntrackedParameter<std::string>("mc", "yes");
  useAllHistos_ = conf.getUntrackedParameter<bool>("useAllHistos", false);

  etaMin[0] = 0.;
  etaMax[0] = 1.4;
  etaMin[1] = 1.4;
  etaMax[1] = 2.9;
  etaMin[2] = 2.9;
  etaMax[2] = 5.2;

  isub = 0;
  if(hcalselector_ == "HB") isub = 1;
  if(hcalselector_ == "HE") isub = 2;
  if(hcalselector_ == "HF") isub = 3;

  imc = 1;
  if(mc_ == "no") imc = 0;
  
  if ( outputFile_.size() != 0 ) {
    edm::LogInfo("OutputInfo") << " Hcal RecHit Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("OutputInfo") << " Hcal RecHit Task histograms will NOT be saved";
  }

  nevent = 0;
	// const char * sub = hcalselector_.c_str();

}


void CaloTowersValidation::beginRun() {}

void CaloTowersValidation::endRun() {}

CaloTowersValidation::~CaloTowersValidation() {

}


void CaloTowersValidation::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & irun, edm::EventSetup const & isetup)
{
	Char_t histo[100];

	ibooker.setCurrentFolder("CaloTowersV/CaloTowersTask");

	//These two histos are not drawn by our macros, but they are used
	//in the EndJob for norms and such so I am leaving them alone for now
	//-------------------------------------------------------------------------------------------
	sprintf  (histo, "Ntowers_per_event_vs_ieta" );
	Ntowers_vs_ieta = ibooker.book1D(histo, histo, 82, -41., 41.);

	sprintf  (histo, "CaloTowersTask_map_Nentries" );
	mapEnergy_N = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
	//-------------------------------------------------------------------------------------------

	//These the single pion scan histos
	//-------------------------------------------------------------------------------------------
	//The first three are not used
	if (useAllHistos_){
		sprintf  (histo, "emean_vs_ieta_E" );
		emean_vs_ieta_E = ibooker.bookProfile(histo, histo, 82, -41., 41., 2100, -100., 2000., "s");
		sprintf  (histo, "emean_vs_ieta_H" );
		emean_vs_ieta_H = ibooker.bookProfile(histo, histo, 82, -41., 41., 2100, -100., 2000., "s");
		sprintf  (histo, "emean_vs_ieta_EH" );
		emean_vs_ieta_EH = ibooker.bookProfile(histo, histo, 82, -41., 41., 2100, -100., 2000., "s");
	}
	//These are drawn
	sprintf  (histo, "emean_vs_ieta_E1" );
	emean_vs_ieta_E1 = ibooker.bookProfile(histo, histo, 82, -41., 41., 2100, -100., 2000., "s");
	sprintf  (histo, "emean_vs_ieta_H1" );
	emean_vs_ieta_H1 = ibooker.bookProfile(histo, histo, 82, -41., 41., 2100, -100., 2000., "s");
	sprintf  (histo, "emean_vs_ieta_EH1" );
	emean_vs_ieta_EH1 = ibooker.bookProfile(histo, histo, 82, -41., 41., 2100, -100., 2000., "s");
	//-------------------------------------------------------------------------------------------

	//Map energy histos are not drawn
	if (useAllHistos_){
		sprintf  (histo, "CaloTowersTask_map_energy_E" );
		mapEnergy_E = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
		sprintf  (histo, "CaloTowersTask_map_energy_H");
		mapEnergy_H = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
		sprintf  (histo, "CaloTowersTask_map_energy_EH" );
		mapEnergy_EH = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
	}

	//All ECAL cell histos are used
	// XXX: ECAL 0-25 [0-26, 26 bins]   HCAL 0-4 [0-5, 5 bins]
	sprintf  (histo, "number_of_bad_cells_Ecal_EB");
	numBadCellsEcal_EB = ibooker.book1D(histo, histo, 26, 0, 26);
	sprintf  (histo, "number_of_bad_cells_Ecal_EE");
	numBadCellsEcal_EE = ibooker.book1D(histo, histo, 26, 0, 26);
	sprintf  (histo, "number_of_recovered_cells_Ecal_EB");
	numRcvCellsEcal_EB = ibooker.book1D(histo, histo, 26, 0, 26);
	sprintf  (histo, "number_of_recovered_cells_Ecal_EE");
	numRcvCellsEcal_EE = ibooker.book1D(histo, histo, 26, 0, 26);
	sprintf  (histo, "number_of_problematic_cells_Ecal_EB");
	numPrbCellsEcal_EB = ibooker.book1D(histo, histo, 26, 0, 26);
	sprintf  (histo, "number_of_problematic_cells_Ecal_EE");
	numPrbCellsEcal_EE = ibooker.book1D(histo, histo, 26, 0, 26); 

	//Occupancy vs. ieta is drawn, occupancy map is needed to draw it 
	sprintf  (histo, "CaloTowersTask_map_occupancy" );
	occupancy_map = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);

	sprintf  (histo, "CaloTowersTask_occupancy_vs_ieta" );
	occupancy_vs_ieta = ibooker.book1D(histo, histo, 82, -41, 41);

	if( isub == 1 || isub == 0) {
		//All cell histos are used
		sprintf  (histo, "number_of_bad_cells_Hcal_HB");
		numBadCellsHcal_HB = ibooker.book1D(histo, histo, 5, 0, 5);
		sprintf  (histo, "number_of_recovered_cells_Hcal_HB");
		numRcvCellsHcal_HB = ibooker.book1D(histo, histo, 5, 0, 5);
		sprintf  (histo, "number_of_problematic_cells_Hcal_HB");
		numPrbCellsHcal_HB = ibooker.book1D(histo, histo, 5, 0, 5); 

		//These are the five oldest CaloTower histos used: NTowers, E in HCAL/ECAL, MET and SET
		//-------------------------------------------------------------------------------------------
		sprintf (histo, "CaloTowersTask_energy_HCAL_HB" ) ;
		meEnergyHcal_HB    = ibooker.book1D(histo, histo, 4100, -200, 8000);  

		sprintf (histo, "CaloTowersTask_energy_ECAL_HB" ) ;
		meEnergyEcal_HB    = ibooker.book1D(histo, histo, 3100, -200, 6000); 

		sprintf (histo, "CaloTowersTask_number_of_fired_towers_HB" ) ;
		meNumFiredTowers_HB = ibooker.book1D(histo, histo, 1000, 0, 2000); 

		sprintf  (histo, "CaloTowersTask_MET_HB" ) ;
		MET_HB = ibooker.book1D(histo, histo, 3000, 0. , 3000. ) ;

		sprintf  (histo, "CaloTowersTask_SET_HB" ) ;
		SET_HB = ibooker.book1D(histo, histo, 8000, 0. , 8000. ) ;
		//-------------------------------------------------------------------------------------------

		//Timing histos and profiles -- all six are necessary
		//-------------------------------------------------------------------------------------------
		sprintf  (histo, "CaloTowersTask_EM_Timing_HB" ) ;
		emTiming_HB = ibooker.book1D(histo, histo, 110, -120., 100. ) ;

		sprintf  (histo, "CaloTowersTask_HAD_Timing_HB" ) ;
		hadTiming_HB = ibooker.book1D(histo, histo, 70, -48., 92. ) ;

		//Energy-Timing histos are divided into low, medium and high to reduce memory usage
		//EM
		sprintf  (histo, "CaloTowersTask_EM_Energy_Timing_Low_HB" ) ;
		emEnergyTiming_Low_HB = ibooker.book2D(histo, histo, 40, 0. , 40., 110, -120., 100.  ) ;

		sprintf  (histo, "CaloTowersTask_EM_Energy_Timing_HB" ) ;
		emEnergyTiming_HB = ibooker.book2D(histo, histo, 200, 0. , 400., 110, -120., 100.  ) ;

		sprintf  (histo, "CaloTowersTask_EM_Energy_Timing_High_HB" ) ;
		emEnergyTiming_High_HB = ibooker.book2D(histo, histo, 200, 0. , 3000., 110, -120., 100.  ) ;

		sprintf  (histo, "CaloTowersTask_EM_Energy_Timing_profile_Low_HB" ) ;
		emEnergyTiming_profile_Low_HB = ibooker.bookProfile(histo, histo, 40, 0. , 40., 110, -120., 100.  ) ;

		sprintf  (histo, "CaloTowersTask_EM_Energy_Timing_profile_HB" ) ;
		emEnergyTiming_profile_HB = ibooker.bookProfile(histo, histo, 200, 0. , 400., 110, -120., 100.  ) ;

		sprintf  (histo, "CaloTowersTask_EM_Energy_Timing_profile_High_HB" ) ;
		emEnergyTiming_profile_High_HB = ibooker.bookProfile(histo, histo, 200, 0. , 3000., 110, -120., 100.  ) ;

		//HAD
		sprintf  (histo, "CaloTowersTask_HAD_Energy_Timing_Low_HB" ) ;
		hadEnergyTiming_Low_HB = ibooker.book2D(histo, histo, 40, 0. , 40., 70, -48., 92. ) ;

		sprintf  (histo, "CaloTowersTask_HAD_Energy_Timing_HB" ) ;
		hadEnergyTiming_HB = ibooker.book2D(histo, histo, 100, 0. , 200., 70, -48., 92. ) ;

		sprintf  (histo, "CaloTowersTask_HAD_Energy_Timing_High_HB" ) ;
		hadEnergyTiming_High_HB = ibooker.book2D(histo, histo, 300, 0. , 3000., 70, -48., 92. ) ;

		sprintf  (histo, "CaloTowersTask_HAD_Energy_Timing_profile_Low_HB" ) ;
		hadEnergyTiming_profile_Low_HB = ibooker.bookProfile(histo, histo, 40, 0. , 40., 70, -48., 92. ) ;

		sprintf  (histo, "CaloTowersTask_HAD_Energy_Timing_profile_HB" ) ;
		hadEnergyTiming_profile_HB = ibooker.bookProfile(histo, histo, 100, 0. , 200., 70, -48., 92. ) ;

		sprintf  (histo, "CaloTowersTask_HAD_Energy_Timing_profile_High_HB" ) ;
		hadEnergyTiming_profile_High_HB = ibooker.bookProfile(histo, histo, 300, 0. , 3000., 70, -48., 92. ) ;
		//-------------------------------------------------------------------------------------------

		//Everything else is not drawn
		if (useAllHistos_){
			sprintf (histo, "CaloTowersTask_sum_of_energy_HCAL_vs_ECAL_HB") ;
			meEnergyHcalvsEcal_HB    = ibooker.book2D(histo, histo, 500, 0., 500., 500, 0., 500.);

			sprintf (histo, "CaloTowersTask_energy_OUTER_HB" ) ;
			meEnergyHO_HB    = ibooker.book1D(histo, histo, 1640, -200, 8000);   

			sprintf (histo, "CaloTowersTask_energy_of_ECAL_component_of_tower_HB" ) ;
			meEnergyEcalTower_HB = ibooker.book1D(histo, histo, 440, -200, 2000); 

			sprintf (histo, "CaloTowersTask_energy_of_HCAL_component_of_tower_HB" ) ;
			meEnergyHcalTower_HB = ibooker.book1D(histo, histo, 440 , -200 , 2000); 

			sprintf  (histo, "CaloTowersTask_energy_HcalPlusEcalPlusHO_HB" ) ;
			meTotEnergy_HB = ibooker.book1D(histo, histo,400, 0., 2000.) ;

			sprintf  (histo, "CaloTowersTask_map_energy_HB" );
			mapEnergy_HB = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
			sprintf  (histo, "CaloTowersTask_map_energy_HCAL_HB");
			mapEnergyHcal_HB = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
			sprintf  (histo, "CaloTowersTask_map_energy_ECAL_HB" );
			mapEnergyEcal_HB = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);

			sprintf  (histo, "CaloTowersTask_phi_MET_HB" ) ;
			phiMET_HB = ibooker.book1D(histo, histo, 72, -3.1415926535898, 3.1415926535898 ) ;
		}
	} 

	if( isub == 2 || isub == 0) {
		//All cell histos are used
		sprintf  (histo, "number_of_bad_cells_Hcal_HE");
		numBadCellsHcal_HE = ibooker.book1D(histo, histo, 5, 0, 5);
		sprintf  (histo, "number_of_recovered_cells_Hcal_HE");
		numRcvCellsHcal_HE = ibooker.book1D(histo, histo, 5, 0, 5);
		sprintf  (histo, "number_of_problematic_cells_Hcal_HE");
		numPrbCellsHcal_HE = ibooker.book1D(histo, histo, 5, 0, 5); 

		//These are the five oldest CaloTower histos used: NTowers, E in HCAL/ECAL, MET and SET
		//-------------------------------------------------------------------------------------------
		sprintf (histo, "CaloTowersTask_energy_HCAL_HE" ) ;
		meEnergyHcal_HE    = ibooker.book1D(histo, histo, 1240, -200, 6000); 

		sprintf (histo, "CaloTowersTask_energy_ECAL_HE" ) ;
		meEnergyEcal_HE    = ibooker.book1D(histo, histo, 1240, -200, 6000);

		sprintf (histo, "CaloTowersTask_number_of_fired_towers_HE" ) ;
		meNumFiredTowers_HE = ibooker.book1D(histo, histo, 1000, 0, 2000); 

		sprintf  (histo, "CaloTowersTask_MET_HE" ) ;
		MET_HE = ibooker.book1D(histo, histo, 1000, 0. , 1000. ) ;

		sprintf  (histo, "CaloTowersTask_SET_HE" ) ;
		SET_HE = ibooker.book1D(histo, histo, 2000, 0. , 2000. ) ;
		//-------------------------------------------------------------------------------------------

		//Timing histos and profiles -- all six are necessary
		//-------------------------------------------------------------------------------------------
		sprintf  (histo, "CaloTowersTask_EM_Timing_HE" ) ;
		emTiming_HE = ibooker.book1D(histo, histo, 110, -120., 100. ) ;

		sprintf  (histo, "CaloTowersTask_HAD_Timing_HE" ) ;
		hadTiming_HE = ibooker.book1D(histo, histo, 70, -48., 92. ) ;

		//Energy-Timing histos are divided into low and normal to reduce memory usage
		//EM
		sprintf  (histo, "CaloTowersTask_EM_Energy_Timing_Low_HE" ) ;
		emEnergyTiming_Low_HE = ibooker.book2D(histo, histo, 160, 0. , 160., 110, -120., 100.  ) ;

		sprintf  (histo, "CaloTowersTask_EM_Energy_Timing_HE" ) ;
		emEnergyTiming_HE = ibooker.book2D(histo, histo, 200, 0. , 800., 110, -120., 100.  ) ;

		sprintf  (histo, "CaloTowersTask_EM_Energy_Timing_profile_Low_HE" ) ;
		emEnergyTiming_profile_Low_HE = ibooker.bookProfile(histo, histo, 160, 0. , 160., 110, -120., 100.  ) ;

		sprintf  (histo, "CaloTowersTask_EM_Energy_Timing_profile_HE" ) ;
		emEnergyTiming_profile_HE = ibooker.bookProfile(histo, histo, 200, 0. , 800., 110, -120., 100.  ) ;

		//HAD
		sprintf  (histo, "CaloTowersTask_HAD_Energy_Timing_Low_HE" ) ;
		hadEnergyTiming_Low_HE = ibooker.book2D(histo, histo, 160, 0. , 160., 70, -48., 92. ) ;

		sprintf  (histo, "CaloTowersTask_HAD_Energy_Timing_HE" ) ;
		hadEnergyTiming_HE = ibooker.book2D(histo, histo, 200, 0. , 800., 70, -48., 92. ) ;

		sprintf  (histo, "CaloTowersTask_HAD_Energy_Timing_profile_Low_HE" ) ;
		hadEnergyTiming_profile_Low_HE = ibooker.bookProfile(histo, histo, 160, 0. , 160., 70, -48., 92. ) ;

		sprintf  (histo, "CaloTowersTask_HAD_Energy_Timing_profile_HE" ) ;
		hadEnergyTiming_profile_HE = ibooker.bookProfile(histo, histo, 200, 0. , 800., 70, -48., 92. ) ;
		//-------------------------------------------------------------------------------------------

		//Everything else is not drawn
		if (useAllHistos_){
			sprintf (histo, "CaloTowersTask_sum_of_energy_HCAL_vs_ECAL_HE") ;
			meEnergyHcalvsEcal_HE    = ibooker.book2D(histo, histo, 500, 0., 500., 500, 0., 500.);

			sprintf (histo, "CaloTowersTask_energy_OUTER_HE" ) ;
			meEnergyHO_HE    = ibooker.book1D(histo, histo, 440, -200, 2000);   

			sprintf (histo, "CaloTowersTask_energy_of_ECAL_component_of_tower_HE" ) ;
			meEnergyEcalTower_HE = ibooker.book1D(histo, histo, 1100, -200, 2000); 

			sprintf (histo, "CaloTowersTask_energy_of_HCAL_component_of_tower_HE" ) ;
			meEnergyHcalTower_HE = ibooker.book1D(histo, histo, 1100 , -200 , 2000); 

			sprintf  (histo, "CaloTowersTask_energy_HcalPlusEcalPlusHO_HE" ) ;
			meTotEnergy_HE = ibooker.book1D(histo, histo,400, 0., 2000.) ;

			sprintf  (histo, "CaloTowersTask_map_energy_HE" );
			mapEnergy_HE = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
			sprintf  (histo, "CaloTowersTask_map_energy_HCAL_HE");
			mapEnergyHcal_HE = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
			sprintf  (histo, "CaloTowersTask_map_energy_ECAL_HE" );
			mapEnergyEcal_HE = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);

			sprintf  (histo, "CaloTowersTask_phi_MET_HE" ) ;
			phiMET_HE = ibooker.book1D(histo, histo, 72, -3.1415926535898, 3.1415926535898 ) ;
		}
	}


	if( isub == 3 || isub == 0) {
		//All cell histos are used
		sprintf  (histo, "number_of_bad_cells_Hcal_HF");
		numBadCellsHcal_HF = ibooker.book1D(histo, histo, 5, 0, 5);
		sprintf  (histo, "number_of_recovered_cells_Hcal_HF");
		numRcvCellsHcal_HF = ibooker.book1D(histo, histo, 5, 0, 5);
		sprintf  (histo, "number_of_problematic_cells_Hcal_HF");
		numPrbCellsHcal_HF = ibooker.book1D(histo, histo, 5, 0, 5); 

		//These are the five oldest CaloTower histos used: NTowers, E in HCAL/ECAL, MET and SET
		//-------------------------------------------------------------------------------------------
		sprintf (histo, "CaloTowersTask_energy_HCAL_HF" ) ;
		meEnergyHcal_HF    = ibooker.book1D(histo, histo, 4040, -200, 20000);

		sprintf (histo, "CaloTowersTask_energy_ECAL_HF" ) ;
		meEnergyEcal_HF    = ibooker.book1D(histo, histo, 2440, -200, 12000);

		sprintf (histo, "CaloTowersTask_number_of_fired_towers_HF" ) ;
		meNumFiredTowers_HF = ibooker.book1D(histo, histo, 1000, 0, 2000); 

		sprintf  (histo, "CaloTowersTask_MET_HF" ) ;
		MET_HF = ibooker.book1D(histo, histo, 500, 0. , 500. ) ;

		sprintf  (histo, "CaloTowersTask_SET_HF" ) ;
		SET_HF = ibooker.book1D(histo, histo, 2000, 0. , 2000. ) ;
		//-------------------------------------------------------------------------------------------

		//Timing histos and profiles -- all six are necessary
		//-------------------------------------------------------------------------------------------
		sprintf  (histo, "CaloTowersTask_EM_Timing_HF" ) ;
		emTiming_HF = ibooker.book1D(histo, histo, 110, -120., 100. ) ;

		sprintf  (histo, "CaloTowersTask_HAD_Timing_HF" ) ;
		hadTiming_HF = ibooker.book1D(histo, histo, 70, -48., 92. ) ;

		//EM
		sprintf  (histo, "CaloTowersTask_EM_Energy_Timing_HF" ) ;
		emEnergyTiming_HF = ibooker.book2D(histo, histo, 150, 0. , 300., 110, -120., 100.  ) ;

		sprintf  (histo, "CaloTowersTask_EM_Energy_Timing_profile_HF" ) ;
		emEnergyTiming_profile_HF = ibooker.bookProfile(histo, histo, 150, 0. , 300., 110, -120., 100.  ) ;

		//HAD (requires two different sets of histograms to lower RAM usage)
		sprintf  (histo, "CaloTowersTask_HAD_Energy_Timing_Low_HF" ) ;
		hadEnergyTiming_Low_HF = ibooker.book2D(histo, histo, 40, 0. , 40., 70, -48., 92. ) ;

		sprintf  (histo, "CaloTowersTask_HAD_Energy_Timing_HF" ) ;
		hadEnergyTiming_HF = ibooker.book2D(histo, histo, 200, 0. , 600., 70, -48., 92. ) ;

		sprintf  (histo, "CaloTowersTask_HAD_Energy_Timing_profile_Low_HF" ) ;
		hadEnergyTiming_profile_Low_HF = ibooker.bookProfile(histo, histo, 40, 0. , 40., 70, -48., 92. ) ;

		sprintf  (histo, "CaloTowersTask_HAD_Energy_Timing_profile_HF" ) ;
		hadEnergyTiming_profile_HF = ibooker.bookProfile(histo, histo, 200, 0. , 600., 70, -48., 92. ) ;
		//-------------------------------------------------------------------------------------------

		//Everything else is not drawn
		if (useAllHistos_){
			sprintf (histo, "CaloTowersTask_sum_of_energy_HCAL_vs_ECAL_HF") ;
			meEnergyHcalvsEcal_HF    = ibooker.book2D(histo, histo, 500, 0., 500., 500, 0., 500.);

			sprintf (histo, "CaloTowersTask_energy_OUTER_HF" ) ;
			meEnergyHO_HF    = ibooker.book1D(histo, histo, 440, -200, 2000);   

			sprintf (histo, "CaloTowersTask_energy_of_ECAL_component_of_tower_HF" ) ;
			meEnergyEcalTower_HF = ibooker.book1D(histo, histo, 440, -200, 2000); 

			sprintf (histo, "CaloTowersTask_energy_of_HCAL_component_of_tower_HF" ) ;
			meEnergyHcalTower_HF = ibooker.book1D(histo, histo, 440 , -200 , 2000); 

			sprintf  (histo, "CaloTowersTask_energy_HcalPlusEcalPlusHO_HF" ) ;
			meTotEnergy_HF = ibooker.book1D(histo, histo, 400, 0., 2000.) ;

			sprintf  (histo, "CaloTowersTask_map_energy_HF" );
			mapEnergy_HF = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
			sprintf  (histo, "CaloTowersTask_map_energy_HCAL_HF");
			mapEnergyHcal_HF = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
			sprintf  (histo, "CaloTowersTask_map_energy_ECAL_HF" );
			mapEnergyEcal_HF = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);

			sprintf  (histo, "CaloTowersTask_phi_MET_HF" ) ;
			phiMET_HF = ibooker.book1D(histo, histo, 72, -3.1415926535898, 3.1415926535898 ) ;
		}
	}



}



void CaloTowersValidation::analyze(edm::Event const& event, edm::EventSetup const& c) {

	nevent++;

	// bool     MC = false; // UNUSED
	double   phi_MC = 9999.;
	double   eta_MC = 9999.;

	if (imc != 0){
		edm::Handle<edm::HepMCProduct> evtMC;
		event.getByToken(tok_evt_,evtMC);  // generator in late 310_preX
		if (!evtMC.isValid()) {
			std::cout << "no HepMCProduct found" << std::endl;    
		} else {
			// MC=true; // UNUSED
			//    std::cout << "*** source HepMCProduct found"<< std::endl;
		}  

		// MC particle with highest pt is taken as a direction reference  
		double maxPt = -99999.;
		int npart    = 0;
		const HepMC::GenEvent * myGenEvent = evtMC->GetEvent();
		for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
				p != myGenEvent->particles_end(); ++p ) {
			double phip = (*p)->momentum().phi();
			double etap = (*p)->momentum().eta();
			//    phi_MC = phip;
			//    eta_MC = etap;
			double pt  = (*p)->momentum().perp();
			if(pt > maxPt) { npart++; maxPt = pt; phi_MC = phip; eta_MC = etap; }
		}
		//  std::cout << "*** Max pT = " << maxPt <<  std::endl;  
	}    

	edm::Handle<CaloTowerCollection> towers;
	event.getByToken(tok_calo_,towers);
	CaloTowerCollection::const_iterator cal;

	double met;
	double phimet;

	// ieta scan 
	double partR  = 0.3;
	double Rmin   = 9999.;
	double Econe  = 0.;
	double Hcone  = 0.;
	double Ee1    = 0.;
	double Eh1    = 0.;
	double ieta_MC = 9999;
	double iphi_MC = 9999;
	//  double  etaM   = 9999.;


	// HB   
	double sumEnergyHcal_HB = 0.;
	double sumEnergyEcal_HB = 0.;
	double sumEnergyHO_HB   = 0.;
	Int_t numFiredTowers_HB = 0;
	double metx_HB   =  0.;
	double mety_HB   =  0.;
	double metz_HB   =  0.;
	double sEt_HB    =  0.;
	// HE   
	double sumEnergyHcal_HE = 0.;
	double sumEnergyEcal_HE = 0.;
	double sumEnergyHO_HE   = 0.;
	Int_t numFiredTowers_HE = 0;
	double metx_HE   =  0.;
	double mety_HE   =  0.;
	double metz_HE   =  0.;
	double sEt_HE    =  0.;
	// HF   
	double sumEnergyHcal_HF = 0.;
	double sumEnergyEcal_HF = 0.;
	double sumEnergyHO_HF   = 0.;
	Int_t numFiredTowers_HF = 0;
	double metx_HF   =  0.;
	double mety_HF   =  0.;
	double metz_HF   =  0.;
	double sEt_HF    =  0.;

	for ( cal = towers->begin(); cal != towers->end(); ++cal ) {

		double eE     = cal->emEnergy();
		double eH     = cal->hadEnergy();
		double eHO    = cal->outerEnergy();
		double etaT   = cal->eta();
		double phiT   = cal->phi();
		double en     = cal->energy();
		double etT    = cal->et();
		double had_tm = cal->hcalTime();
		double em_tm  = cal->ecalTime();

		int numBadEcalCells = cal->numBadEcalCells();
		int numRcvEcalCells = cal->numRecoveredEcalCells();
		int numPrbEcalCells = cal->numProblematicEcalCells();

		int numBadHcalCells = cal->numBadHcalCells();
		int numRcvHcalCells = cal->numRecoveredHcalCells();
		int numPrbHcalCells = cal->numProblematicHcalCells(); 

		math::RhoEtaPhiVector mom(cal->et(), cal->eta(), cal->phi());
		//  Vector mom  = cal->momentum(); 

		// cell properties    
		CaloTowerDetId idT = cal->id();
		int ieta = idT.ieta();
		if(ieta > 0) ieta -= 1;
		int iphi = idT.iphi();

		// ecal:  0 EcalBarrel  1 EcalEndcap
		// hcal:  0 hcalBarrel  1 HcalEndcap  2 HcalForward
		std::vector<int> inEcals(2), inHcals(3);
		unsigned int constitSize = cal->constituentsSize();
		for(unsigned int ic=0; ic<constitSize; ic++){
			DetId detId = cal->constituent(ic);
			if(detId.det() == DetId::Ecal){
				if( detId.subdetId()==EcalBarrel ) inEcals[0] =1;
				else if( detId.subdetId()==EcalEndcap ) inEcals[1] =1;
			}
			if(detId.det() == DetId::Hcal){
				if( HcalDetId(detId).subdet()==HcalBarrel ) inHcals[0] =1;
				else if( HcalDetId(detId).subdet()==HcalEndcap ) inHcals[1] =1;
				else if( HcalDetId(detId).subdet()==HcalForward ) inHcals[2] =1;
			}
		}
		//All cell histos are used
		if( inEcals[0] ) {
			numBadCellsEcal_EB->Fill(numBadEcalCells);
			numRcvCellsEcal_EB->Fill(numRcvEcalCells);
			numPrbCellsEcal_EB->Fill(numPrbEcalCells);
		}
		if( inEcals[1] ) {
			numBadCellsEcal_EE->Fill(numBadEcalCells);
			numRcvCellsEcal_EE->Fill(numRcvEcalCells);
			numPrbCellsEcal_EE->Fill(numPrbEcalCells);
		} 

		if (imc != 0){
			double r    = dR(eta_MC, phi_MC, etaT, phiT);

			if( r < partR ){
				Econe += eE; 
				Hcone += eH; 

				// closest to MC
				if(r < Rmin) { 
					if( fabs(eta_MC) < 3.0 && (ieta > 29 || ieta < -29)) {;}
					else {    
						Rmin = r;
						ieta_MC = ieta; 
						iphi_MC = iphi; 
						Ee1     = eE;
						Eh1     = eH;
					}
				}
			}
		}

		//Ntowers is used in EndJob, occupancy_map is used for occupancy vs ieta
		Ntowers_vs_ieta -> Fill(double(ieta),1.);
		occupancy_map -> Fill(double(ieta),double(iphi));


		if((isub == 0 || isub == 1) 
				&& (fabs(etaT) <  etaMax[0] && fabs(etaT) >= etaMin[0] )) {

			//All cell histos are used
			numBadCellsHcal_HB->Fill(numBadHcalCells);
			numRcvCellsHcal_HB->Fill(numRcvHcalCells);
			numPrbCellsHcal_HB->Fill(numPrbHcalCells);

			//Map energy histos are not used
			if (useAllHistos_){
				mapEnergy_HB     -> Fill(double(ieta), double(iphi), en); 
				mapEnergyHcal_HB -> Fill(double(ieta), double(iphi), eH); 
				mapEnergyEcal_HB -> Fill(double(ieta), double(iphi), eE); 
			}
			//      std::cout << " e_ecal = " << eE << std::endl;

			//  simple sums
			sumEnergyHcal_HB += eH;
			sumEnergyEcal_HB += eE;
			sumEnergyHO_HB   += eHO;

			numFiredTowers_HB++;

			//Not used
			if (useAllHistos_){
				meEnergyEcalTower_HB->Fill(eE);
				meEnergyHcalTower_HB->Fill(eH);    
			}

			// MET, SET & phimet
			//  double  etT = cal->et();
			metx_HB += mom.x();   
			mety_HB += mom.y();  //etT * sin(phiT);          
			sEt_HB  += etT;

			//Timing (all histos are used)
			emTiming_HB->Fill(em_tm);
			hadTiming_HB->Fill(had_tm);

			emEnergyTiming_Low_HB->Fill(eE, em_tm);
			emEnergyTiming_HB->Fill(eE, em_tm);
			emEnergyTiming_High_HB->Fill(eE, em_tm);
			emEnergyTiming_profile_Low_HB->Fill(eE, em_tm);
			emEnergyTiming_profile_HB->Fill(eE, em_tm);
			emEnergyTiming_profile_High_HB->Fill(eE, em_tm);

			hadEnergyTiming_Low_HB->Fill(eH, had_tm); 
			hadEnergyTiming_HB->Fill(eH, had_tm); 
			hadEnergyTiming_High_HB->Fill(eH, had_tm); 
			hadEnergyTiming_profile_Low_HB->Fill(eH, had_tm);     
			hadEnergyTiming_profile_HB->Fill(eH, had_tm);     
			hadEnergyTiming_profile_High_HB->Fill(eH, had_tm);     
		}

		if((isub == 0 || isub == 2) 
				&& (fabs(etaT) <  etaMax[1] && fabs(etaT) >= etaMin[1] )) {

			//All cell histos are used
			numBadCellsHcal_HE->Fill(numBadHcalCells);
			numRcvCellsHcal_HE->Fill(numRcvHcalCells);
			numPrbCellsHcal_HE->Fill(numPrbHcalCells);

			//Map energy histos are not used
			if (useAllHistos_){
				mapEnergy_HE     -> Fill(double(ieta), double(iphi), en); 
				mapEnergyHcal_HE -> Fill(double(ieta), double(iphi), eH); 
				mapEnergyEcal_HE -> Fill(double(ieta), double(iphi), eE); 
			}
			//      std::cout << " e_ecal = " << eE << std::endl;

			//  simple sums
			sumEnergyHcal_HE += eH;
			sumEnergyEcal_HE += eE;
			sumEnergyHO_HE   += eHO;

			numFiredTowers_HE++;

			//Not used
			if (useAllHistos_){
				meEnergyEcalTower_HE->Fill(eE);
				meEnergyHcalTower_HE->Fill(eH);    
			}
			// MET, SET & phimet
			//  double  etT = cal->et();
			metx_HE += mom.x();   
			mety_HE += mom.y();  //etT * sin(phiT);          
			sEt_HE  += etT;    

			//Timing (all histos are used)
			emTiming_HE->Fill(em_tm);
			hadTiming_HE->Fill(had_tm);

			emEnergyTiming_Low_HE->Fill(eE, em_tm);
			emEnergyTiming_HE->Fill(eE, em_tm);
			emEnergyTiming_profile_Low_HE->Fill(eE, em_tm);
			emEnergyTiming_profile_HE->Fill(eE, em_tm);

			hadEnergyTiming_Low_HE->Fill(eH, had_tm);     
			hadEnergyTiming_HE->Fill(eH, had_tm);   
			hadEnergyTiming_profile_Low_HE->Fill(eH, had_tm);     
			hadEnergyTiming_profile_HE->Fill(eH, had_tm);     
		}

		if((isub == 0 || isub == 3) 
				&& (fabs(etaT) <  etaMax[2] && fabs(etaT) >= etaMin[2] )) {

			//All cell histos are used
			numBadCellsHcal_HF->Fill(numBadHcalCells);
			numRcvCellsHcal_HF->Fill(numRcvHcalCells);
			numPrbCellsHcal_HF->Fill(numPrbHcalCells);

			//Map energy histos are not used
			if (useAllHistos_){
				mapEnergy_HF     -> Fill(double(ieta), double(iphi), en); 
				mapEnergyHcal_HF -> Fill(double(ieta), double(iphi), eH); 
				mapEnergyEcal_HF -> Fill(double(ieta), double(iphi), eE); 
			}
			//      std::cout << " e_ecal = " << eE << std::endl;

			//  simple sums
			sumEnergyHcal_HF += eH;
			sumEnergyEcal_HF += eE;
			sumEnergyHO_HF   += eHO;

			numFiredTowers_HF++;

			//Not used
			if (useAllHistos_){
				meEnergyEcalTower_HF->Fill(eE);
				meEnergyHcalTower_HF->Fill(eH);    
			}
			// MET, SET & phimet
			//  double  etT = cal->et();
			metx_HF += mom.x();   
			mety_HF += mom.y();  //etT * sin(phiT);          
			sEt_HF  += etT;    

			//Timing (all histos are used)
			emTiming_HF->Fill(em_tm);
			hadTiming_HF->Fill(had_tm);
			emEnergyTiming_HF->Fill(eE, em_tm);
			emEnergyTiming_profile_HF->Fill(eE, em_tm);

			hadEnergyTiming_Low_HF->Fill(eH, had_tm);     
			hadEnergyTiming_HF->Fill(eH, had_tm);     
			hadEnergyTiming_profile_Low_HF->Fill(eH, had_tm);     
			hadEnergyTiming_profile_HF->Fill(eH, had_tm);     
		}



	} // end of Towers cycle 

	//These are the six single pion histos; only the second set is used
	if (useAllHistos_){
		emean_vs_ieta_E  -> Fill(double(ieta_MC), Econe); 
		emean_vs_ieta_H  -> Fill(double(ieta_MC), Hcone); 
		emean_vs_ieta_EH -> Fill(double(ieta_MC), Econe+Hcone); 
	}
	emean_vs_ieta_E1  -> Fill(double(ieta_MC), Ee1); 
	emean_vs_ieta_H1  -> Fill(double(ieta_MC), Eh1); 
	emean_vs_ieta_EH1 -> Fill(double(ieta_MC), Ee1+Eh1); 

	//Map histos are not used except the last one in EndJob
	if (useAllHistos_){
		mapEnergy_E -> Fill(double(ieta_MC), double(iphi_MC), Ee1); 
		mapEnergy_H -> Fill(double(ieta_MC), double(iphi_MC), Eh1); 
		mapEnergy_EH -> Fill(double(ieta_MC), double(iphi_MC), Ee1+Eh1);
	} 
	mapEnergy_N  -> Fill(double(ieta_MC), double(iphi_MC), 1.); 


	if(isub == 0 || isub == 1) {
		met    = sqrt(metx_HB*metx_HB + mety_HB*mety_HB);
		Vector metv(metx_HB,mety_HB,metz_HB);
		phimet = metv.phi();

		//Five oldest drawn histos first; the rest are not used
		meEnergyHcal_HB->      Fill(sumEnergyHcal_HB);
		meEnergyEcal_HB->      Fill(sumEnergyEcal_HB);
		meNumFiredTowers_HB->  Fill(numFiredTowers_HB);
		MET_HB    -> Fill (met); 
		SET_HB    -> Fill (sEt_HB); 

		if (useAllHistos_){
			meEnergyHcalvsEcal_HB->Fill(sumEnergyEcal_HB, sumEnergyHcal_HB);
			meEnergyHO_HB->        Fill(sumEnergyHO_HB);
			meTotEnergy_HB->       Fill(sumEnergyHcal_HB+sumEnergyEcal_HB
					+sumEnergyHO_HB);    
			phiMET_HB -> Fill (phimet);
		} 
	}    


	if(isub == 0 || isub == 2) {
		met    = sqrt(metx_HE*metx_HE + mety_HE*mety_HE);
		Vector metv(metx_HE,mety_HE,metz_HE);
		phimet = metv.phi();

		//Five oldest drawn histos first; the rest are not used
		meEnergyHcal_HE->      Fill(sumEnergyHcal_HE);
		meEnergyEcal_HE->      Fill(sumEnergyEcal_HE);
		meNumFiredTowers_HE->  Fill(numFiredTowers_HE);
		MET_HE    -> Fill (met); 
		SET_HE    -> Fill (sEt_HE); 

		if (useAllHistos_){
			meEnergyHcalvsEcal_HE->Fill(sumEnergyEcal_HE, sumEnergyHcal_HE);
			meEnergyHO_HE->        Fill(sumEnergyHO_HE);
			meTotEnergy_HE->       Fill(sumEnergyHcal_HE+sumEnergyEcal_HE
					+sumEnergyHO_HE);    
			phiMET_HE -> Fill (phimet); 
		}
	}

	if(isub == 0 || isub == 3) {
		met    = sqrt(metx_HF*metx_HF + mety_HF*mety_HF);
		Vector metv(metx_HF,mety_HF,metz_HF);
		phimet = metv.phi();

		//Five oldest drawn histos first; the rest are not used
		meEnergyHcal_HF->      Fill(sumEnergyHcal_HF);
		meEnergyEcal_HF->      Fill(sumEnergyEcal_HF);
		meNumFiredTowers_HF->  Fill(numFiredTowers_HF);
		MET_HF    -> Fill (met); 
		SET_HF    -> Fill (sEt_HF); 

		if (useAllHistos_){
			meEnergyHcalvsEcal_HF->Fill(sumEnergyEcal_HF, sumEnergyHcal_HF);
			meEnergyHO_HF->        Fill(sumEnergyHO_HF);
			meTotEnergy_HF->       Fill(sumEnergyHcal_HF+sumEnergyEcal_HF
					+sumEnergyHO_HF);    
			phiMET_HF -> Fill (phimet);
		}
	}

}

double CaloTowersValidation::dR(double eta1, double phi1, double eta2, double phi2) { 
	double PI = 3.1415926535898;
	double deltaphi= phi1 - phi2;
	if( phi2 > phi1 ) { deltaphi= phi2 - phi1;}
	if(deltaphi > PI) { deltaphi = 2.*PI - deltaphi;}
	double deltaeta = eta2 - eta1;
	double tmp = sqrt(deltaeta* deltaeta + deltaphi*deltaphi);
	return tmp;
}

DEFINE_FWK_MODULE(CaloTowersValidation);


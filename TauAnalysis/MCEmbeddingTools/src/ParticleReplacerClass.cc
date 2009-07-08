#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerClass.h"

#ifndef TXGIVE
#define TXGIVE txgive_
extern "C" {
  void TXGIVE(const char*,int length);
}
#endif

#ifndef TXGIVE_INIT
#define TXGIVE_INIT txgive_init_
extern "C" {
  void TXGIVE_INIT();
}
#endif

ParticleReplacerClass::ParticleReplacerClass(const edm::ParameterSet& pset)
{
// 	using namespace reco;
	using namespace edm;
	using namespace std;

	// this module creates a edm::HepMCProduct
	produces<edm::HepMCProduct>();
	
	HepMC::HEPEVT_Wrapper::set_max_number_entries(4000);
	HepMC::HEPEVT_Wrapper::set_sizeof_real(8);

	// replacementMode =
	//	0 - remove Myons from existing HepMCProduct and implant taus (+decay products)
	//	1 - build new HepMCProduct only with taus (+decay products)
	replacementMode_ = pset.getUntrackedParameter<int>("replacementMode",1);

/*
	// sourceMode =
	//	0 - use HepMCProduct-source
	//			=> transformMode_ is set to 1 automatically
	//	1 - use two reconstructed muons
	//			=> replacementMode_ is set to 1 automatically
	sourceMode_ = pset.getUntrackedParameter<int>("sourceMode",0);

	// transformationMode =
	//	0 - the stored particles are already transformed
	//	1 - transform the two muons into two taus
	//			=> replacementMode_ is set to 1 automatically
	transformationMode_ = pset.getUntrackedParameter<int>("transformationMode",1);
*/
	// generatorMode =
	//	0 - use Pythia
	//	1 - use Tauola
	generatorMode_ = pset.getUntrackedParameter<int>("generatorMode",0);

	// If one wants to use two instances of this module in one
	// configuration file, there might occur some segmentation
	// faults due to the second initialisation of Tauola. This
	// can be prevented by setting noInitialisation to false.
	//          Caution: This option is not tested!
	noInitialisation_ = pset.getUntrackedParameter<bool>("noInitialisation",false);

	printEvent_ = pset.getUntrackedParameter<bool>("printEvent",false);	// normally tau-mass

	motherParticleID_ = pset.getUntrackedParameter<int>("motherParticleID",23);

	selectedParticles_ = pset.getUntrackedParameter<string>("selectedParticles","selectMuons");

	HepMCSource_ = pset.getUntrackedParameter<string>("HepMCSource","source");

/*
	if (sourceMode_==1 && replacementMode_!=1)
	{
		cout << "replacementMode_ is forced to be '1'!!!\n";
		replacementMode_=1;
	}
	if (sourceMode_==0 && transformationMode_!=0)
	{
		cout << "transformMode is forced to be '1'!!!\n";
		transformationMode_=1;
	}
*/
	if (!noInitialisation_)
		cout << "starting init process... " << endl;
	else
		cout << "skip init process..." << endl;

	if (generatorMode_==0 && !noInitialisation_) //pythia		
	{
		std::cout << "initialize Pythia only...\n";
		initPythia(pset);
	}
	else if (generatorMode_==1 && !noInitialisation_) //tauola
	{
		std::cout << "initialize Pythia and Tauola...\n";
		initPythiaTauola(pset);
	}

	std::cout << "*** generatorMode      "<< generatorMode_<< "\n";
	std::cout << "*** replacementMode    "<< replacementMode_<< "\n";

	// setting up the random numbers
	srand(time(NULL));
	uint32_t seed = rand();
	ostringstream sRandomSet;
	sRandomSet <<"MRPY(1)="<<seed;
	call_pygive(sRandomSet.str());

	return;
}

void ParticleReplacerClass::initPythia(const edm::ParameterSet& pset)
{
	////////////////////////
	// Set PYTHIA parameters in a single ParameterSet
	
	ParameterSet pythia_params = 
	pset.getParameter<ParameterSet>("PythiaParameters") ;
	
	// The parameter sets to be read (default, min bias, user ...) in the
	// proper order.
	vector<string> setNames = pythia_params.getParameter<vector<string> >("parameterSets");

	// Loop over the sets
	
	for ( unsigned i=0; i<setNames.size(); ++i )
	{
		string mySet = setNames[i];
	
		// Read the PYTHIA parameters for each set of parameters
		vector<string> pars = pythia_params.getParameter<vector<string> >(mySet);
		
		if (mySet != "CSAParameters")
		{
			cout << "----------------------------------------------" << endl;
			cout << "Read PYTHIA parameter set " << mySet << endl;
			cout << "----------------------------------------------" << endl;
		
			// Loop over all parameters and stop in case of mistake
			for( vector<string>::const_iterator itPar = pars.begin(); itPar != pars.end(); ++itPar ) 
			{
			  cout << (*itPar) << "\n";
				static string sRandomValueSetting("MRPY(1)");
				if( 0 == itPar->compare(0,sRandomValueSetting.size(),sRandomValueSetting) )
				{
					throw edm::Exception(edm::errors::Configuration,"PythiaError") <<" attempted to set random number using pythia command 'MRPY(1)' this is not allowed.\n  Please use the RandomNumberGeneratorService to set the random number seed.";
				}
				if( ! call_pygive(*itPar) ) 
				{
					throw edm::Exception(edm::errors::Configuration,"PythiaError") <<" pythia did not accept the following \""<<*itPar<<"\"";
				}
			}
		}
	}
}

// the following code has been taken from GeneratorInterface/Pythia6Interface
void ParticleReplacerClass::initPythiaTauola(const edm::ParameterSet& pset)
{
	// Set PYTHIA parameters in a single ParameterSet
	ParameterSet pythia_params = pset.getParameter<ParameterSet>("PythiaParameters") ;
	
	// The parameter sets to be read (default, min bias, user ...) in the
	// proper order.
	vector<string> setNames = pythia_params.getParameter<vector<string> >("parameterSets");
	
	// Loop over the sets
	for ( unsigned i=0; i<setNames.size(); ++i )
	{
		
		string mySet = setNames[i];
		std::cout << mySet << " -----------------\n";
		// Read the PYTHIA parameters for each set of parameters
		vector<string> pars = pythia_params.getParameter<vector<string> >(mySet);
		
		if (mySet != "SLHAParameters" && mySet != "CSAParameters")
		{
		
			// Loop over all parameters and stop in case of mistake
			for( vector<string>::const_iterator itPar = pars.begin(); itPar != pars.end(); ++itPar ) 
			{
				static string sRandomValueSetting("MRPY(1)");
				if( 0 == itPar->compare(0,sRandomValueSetting.size(),sRandomValueSetting) )
				{
					throw edm::Exception(edm::errors::Configuration,"PythiaError") <<" attempted to set random number using pythia command 'MRPY(1)' this is not allowed.\n  Please use the RandomNumberGeneratorService to set the random number seed.";
				}
				if( ! call_pygive(*itPar) ) 
				{
					throw edm::Exception(edm::errors::Configuration,"PythiaError")<<" pythia did not accept the following \""<<*itPar<<"\"";
				}
			}
		}
		else if(mySet == "CSAParameters")
		{
			// Read CSA parameter
			pars = pythia_params.getParameter<vector<string> >("CSAParameters");
			
			call_txgive_init();
			// Loop over all parameters and stop in case of a mistake
			for (vector<string>::const_iterator itPar = pars.begin(); itPar != pars.end(); ++itPar)
			{
				call_txgive(*itPar); 
			} 
		} 
// 		else if(mySet == "SLHAParameters")
// 		{
// 			// Read SLHA parameter
// 			
// 			pars = pythia_params.getParameter<vector<string> >("SLHAParameters");
// 			
// 			// Loop over all parameters and stop in case of a mistake
// 			for (vector<string>::const_iterator itPar = pars.begin(); itPar != pars.end(); ++itPar) 
// 			{
// 				call_slhagive(*itPar); 
// 			} 
// 			
// 			call_slha_init(); 
// 		}
  }

	// TAUOLA, etc.
	//
	useExternalGenerators_ = pset.getUntrackedParameter<bool>("UseExternalGenerators",false);

	if ( useExternalGenerators_ )
	{
		// read External Generator parameters
		ParameterSet ext_gen_params = pset.getParameter<ParameterSet>("ExternalGenerators") ;
		vector<string> extGenNames = ext_gen_params.getParameter< vector<string> >("parameterSets");
		for (unsigned int ip=0; ip<extGenNames.size(); ++ip )
		{
			string curSet = extGenNames[ip];
			ParameterSet gen_par_set =
			ext_gen_params.getUntrackedParameter< ParameterSet >(curSet);
			/*
			cout << "----------------------------------------------" << endl;
			cout << "Read External Generator parameter set "  << endl;
			cout << "----------------------------------------------" << endl;
			*/
			if ( curSet == "Tauola" )
			{
				useTauola_ = true;
				if ( useTauola_ )
					cout << "--> use TAUOLA" << endl;
				useTauolaPolarization_ = gen_par_set.getParameter<bool>("UseTauolaPolarization");
				if ( useTauolaPolarization_ ) 
				{
					cout << "(Polarization effects enabled)" << endl;
					tauola_.enablePolarizationEffects();
				} 
				else 
				{
					cout << "(Polarization effects disabled)" << endl;
					tauola_.disablePolarizationEffects();
				}
				vector<string> cards = gen_par_set.getParameter< vector<string> >("InputCards");
				cout << "----------------------------------------------" << endl;
				cout << "Initializing Tauola" << endl;
				for( vector<string>::const_iterator itPar = cards.begin(); itPar != cards.end(); ++itPar )
				{
					call_txgive(*itPar);
				}
				tauola_.initialize();
			}
		}
	}

}

ParticleReplacerClass::~ParticleReplacerClass()
{
	// do anything here that needs to be done at desctruction time
	// (e.g. close files, deallocate resources etc.)
}

// ------------ method called to produce the data  ------------
void
ParticleReplacerClass::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	using namespace edm;
	using namespace std;
	using namespace HepMC;

	HepMC::GenEvent * evt=0;

	GenVertex * zvtx = new GenVertex();
	// things that are now obsolete:	
	//	sourceMode_
	//	transformationMode_
	
	reco::GenParticle * part1=0;
	reco::GenParticle * part2=0;

	/// 1) access the particles to be used	
	Handle<std::vector<reco::Particle> > dataHandle;
	if (!iEvent.getByLabel(selectedParticles_,dataHandle))
	{
		std::cout << "Stored Particles not found:\n"<< selectedParticles_ << "\n";
		return;
	}
	const std::vector<reco::Particle> particles = *( dataHandle.product() );

	if (particles.size()==0)
	{
		std::cout << "NO PARTICLES FOUND!";
		return;
	}
	else
	{
		std::cout << particles.size() << " particles found, continue processing\n";
	}

	/// 3) prepare the event
	if (replacementMode_==0)
	{
			Handle<edm::HepMCProduct> HepMCHandle;
			iEvent.getByLabel(HepMCSource_,HepMCHandle);
		
			evt = new HepMC::GenEvent(*(HepMCHandle->GetEvent()));
	
			for ( GenEvent::vertex_iterator p = evt->vertices_begin(); p != evt->vertices_end(); p++ ) 
			{
				GenVertex * vtx=(*p);
				
				if ( vtx->particles_out_size()<=0 || vtx->particles_in_size()<=0)
					continue;
				
				bool temp_muon1=false,temp_muon2=false,temp_z=false;
				for (GenVertex::particles_out_const_iterator it = vtx->particles_out_const_begin(); it!=vtx->particles_out_const_end(); it++)
				{
					if ((*it)->pdg_id()==13) temp_muon1=true;
					if ((*it)->pdg_id()==-13) temp_muon2=true;
					if ((*it)->pdg_id()==23) temp_z=true;
				}
	
				int mother_pdg=(*vtx->particles_in_const_begin())->pdg_id();
	
				if ((vtx->particles_out_size()==2 && vtx->particles_in_size()>0 
					&& mother_pdg == 23  
					&& temp_muon1
					&& temp_muon2)
				|| (vtx->particles_out_size()>2 && vtx->particles_in_size()>0 
					&& mother_pdg == 23
					&& temp_muon1 && temp_muon2 && temp_z ))
				{
					zvtx=*p;
				}
			}
/*
*/

			cleanEvent(evt, zvtx);

			// prevent a decay of existing particles
			// this is due to a bug in the PythiaInterface that should be fixed in newer versions
			for (GenEvent::particle_iterator it=evt->particles_begin();it!=evt->particles_end();it++)
				(*it)->set_status(0);

			for (std::vector<reco::Particle>::const_iterator it=particles.begin();it!=particles.end();it++)
			{
				zvtx->add_particle_out(new HepMC::GenParticle((FourVector)it->p4(), it->pdgId(), 1, Flow(), Polarization(0,0)));
			}
	}

	// new product with tau decays
	if (replacementMode_==1)
	{
		reco::Particle::LorentzVector mother_particle_p4;
		for (std::vector<reco::Particle>::const_iterator it=particles.begin();it!=particles.end();it++)
			mother_particle_p4+=it->p4();

		reco::Particle::Point production_point = particles.begin()->vertex();
		GenVertex * decayvtx = new GenVertex(FourVector(production_point.x(),production_point.y(),production_point.z(),0));

		HepMC::GenParticle * mother_particle = new HepMC::GenParticle((FourVector)mother_particle_p4, motherParticleID_, (generatorMode_==0 ? 3 : 2), Flow(), Polarization(0,0));

		decayvtx->add_particle_in(mother_particle);
		
		evt = new HepMC::GenEvent();

		for (std::vector<reco::Particle>::const_iterator it=particles.begin();it!=particles.end();it++)
		{
			decayvtx->add_particle_out(new HepMC::GenParticle((FourVector)it->p4(), it->pdgId(), 1, Flow(), Polarization(0,0)));
		}
		evt->add_vertex(decayvtx);
	}
	repairBarcodes(evt);

	HepMC::GenEvent * retevt = 0;

	/// 3) process the event
	if (generatorMode_==0)	// Pythia
		retevt=processEventWithPythia(evt);

	if (generatorMode_==1)	// TAUOLA
		retevt=processEventWithTauola(evt);

	// recover the status codes
	if (replacementMode_==0)
	{
		for (GenEvent::particle_iterator it=retevt->particles_begin();it!=retevt->particles_end();it++)
		{
			if ((*it)->end_vertex())
				(*it)->set_status(2);
			else
				(*it)->set_status(1);
		}
	}

	auto_ptr<HepMCProduct> bare_product(new HepMCProduct()); 
	if (printEvent_)
		retevt->print(std::cout);

	bare_product->addHepMCData(retevt);
	iEvent.put(bare_product);

	delete part1;
	delete part2;
	delete zvtx;
	delete evt;
	return;
}

// ------------ method called once each job just before starting event loop  ------------
void 
ParticleReplacerClass::beginJob(const edm::EventSetup&)
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
ParticleReplacerClass::endJob()
{

}

HepMC::GenEvent *  ParticleReplacerClass::processEventWithTauola(HepMC::GenEvent * evt)
{
	using namespace HepMC;

	// convert the event from HepMC to HEPEVT
	HepMC::IO_HEPEVT conv;
	conv.write_event(evt);

	// HEPEVT to PYJETS (or so)
	call_pyhepc(2);

	// call tauola
	tauola_.processEvent();

	// PYJETS to HEPEVT (or so)
	call_pyhepc(1);

	// event to be returned
	return conv.read_next_event();
}

HepMC::GenEvent *  ParticleReplacerClass::processEventWithPythia(HepMC::GenEvent * evt)
{
	using namespace HepMC;

	// convert the event from HepMC to HEPEVT
	HepMC::IO_HEPEVT conv;
	conv.write_event(evt);

	// HEPEVT to PYJETS (or so)
	call_pyhepc(2);

	// call Pythia
	call_pyexec();

	// PYJETS to HEPEVT (or so)
	call_pyhepc(1);

	// event to be returned
	return conv.read_next_event();
}

void ParticleReplacerClass::cleanEvent(HepMC::GenEvent * evt, HepMC::GenVertex * vtx)
{
	using namespace HepMC;
	using namespace std;
	using namespace edm;
	using namespace reco;

	stack<HepMC::GenParticle *> deleteParticle;
	
	stack<GenVertex *> deleteVertex;
	stack<GenVertex *> queueVertex;

	if (vtx->particles_out_size()>0)
	{
		for (GenVertex::particles_out_const_iterator it=vtx->particles_out_const_begin();it!=vtx->particles_out_const_end();it++)
		{
			deleteParticle.push(*it);
			if ((*it)->end_vertex())
				queueVertex.push((*it)->end_vertex());
		}
	}

	while (!queueVertex.empty())
	{
		GenVertex * temp_vtx=queueVertex.top();
		if (temp_vtx->particles_out_size()>0)
		{
			for (GenVertex::particles_out_const_iterator it=temp_vtx->particles_out_const_begin();it!=temp_vtx->particles_out_const_end();it++)
			{
				if ((*it)->end_vertex())
					queueVertex.push((*it)->end_vertex());
			}
			delete temp_vtx;
		}
		deleteVertex.push(queueVertex.top());
		queueVertex.pop();
	}
	
	while (!deleteVertex.empty())
	{
  		evt->remove_vertex(deleteVertex.top());
		deleteVertex.pop();
	}

	while (!deleteParticle.empty())
	{
		delete vtx->remove_particle(deleteParticle.top());
		deleteParticle.pop();
	}

	while (!deleteVertex.empty())
		deleteVertex.pop();
	while (!queueVertex.empty())
		queueVertex.pop();

	repairBarcodes(evt);
}

void ParticleReplacerClass::repairBarcodes(HepMC::GenEvent * evt)
{
	using namespace HepMC;

	// repair the barcodes
	int max_barc=0;
	for (GenEvent::vertex_iterator it=evt->vertices_begin();it!=evt->vertices_end();it++)
	{
		while (!(*it)->suggest_barcode(-1*(++max_barc)))
			;
	}

	max_barc=0;
	for (GenEvent::particle_iterator it=evt->particles_begin();it!=evt->particles_end();it++)
	{
		while (!(*it)->suggest_barcode(++max_barc))
			;
	}
}

bool ParticleReplacerClass::call_pygive(const std::string& iParm ) 
{
	int numWarn = pydat1.mstu[26]; //# warnings
	int numErr = pydat1.mstu[22];// # errors
	// call the fortran routine pygive with a fortran string
	PYGIVE( iParm.c_str(), iParm.length() );  
	//if an error or warning happens it is problem
	return pydat1.mstu[26] == numWarn && pydat1.mstu[22] == numErr;   
}

// tested:
bool ParticleReplacerClass::call_txgive(const std::string& iParm )
{
	TXGIVE( iParm.c_str(), iParm.length() );
	cout << "     " <<  iParm.c_str() << endl; 
	return 1;  
}

bool ParticleReplacerClass::call_txgive_init()
{
	TXGIVE_INIT();
	cout << "  Setting CSA reweighting parameters.   "   << endl;
	return 1;
}

//define this as a plug-in
//DEFINE_FWK_MODULE(Replacer);

#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerClass.h"

#include "GeneratorInterface/ExternalDecays/interface/DecayRandomEngine.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "HepMC/IO_HEPEVT.h"

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

ParticleReplacerClass::ParticleReplacerClass(const edm::ParameterSet& pset, bool verbose):
  ParticleReplacerBase(pset),
  generatorMode_(pset.getParameter<std::string>("generatorMode")),
  tauola_(gen::TauolaInterface::getInstance()),
  printEvent_(verbose),
  outTree(0),
  maxNumberOfAttempts_(pset.getUntrackedParameter<int>("maxNumberOfAttempts", 1000))
{
	tauola_->setPSet(pset.getParameter< edm::ParameterSet>("TauolaOptions"));
// 	using namespace reco;
	using namespace edm;
	using namespace std;

	//HepMC::HEPEVT_Wrapper::set_max_number_entries(4000);
	//HepMC::HEPEVT_Wrapper::set_sizeof_real(8);

	// transformationMode =
	//  0 - no transformation
	//  1 - mumu -> tautau
	transformationMode_ = pset.getUntrackedParameter<int>("transformationMode",1);
	switch (transformationMode_)
	{
		case 0:
		{	
			LogInfo("Replacer") << "won't do any transformation with the given mumu";
			break;
		}
		case 1:
		{
			LogInfo("Replacer") << "will transform mumu into tautau";
			break;
		}
		case 2:
    {
      LogInfo("Replacer") << "will transform mumu into taunu (as coming from a W boson)";
      break;
    }    
		case 3:
		{
			LogInfo("Replacer") << "Will transform  mu-nu into tau-nu. No mass correction will be made.";
			break;
		}
		default:
		{
			throw cms::Exception("ParticleReplacerClass")  << "Unknown transformation mode!\n";
			break;
		}
            
	}
	
	// If one wants to use two instances of this module in one
	// configuration file, there might occur some segmentation
	// faults due to the second initialisation of Tauola. This
	// can be prevented by setting noInitialisation to false.
	//          Caution: This option is not tested!
	noInitialisation_ = pset.getUntrackedParameter<bool>("noInitialisation",false);

	motherParticleID_ = pset.getUntrackedParameter<int>("motherParticleID",23);

	// requires the visible decay products of a tau to have a sum transverse momentum
	std::string minVisibleTransverseMomentumLine = pset.getUntrackedParameter<std::string>("minVisibleTransverseMomentum","");

  // fallback for backwards compatibility: If it's a single number then use this as a threshold for both particles
  const char* startptr = minVisibleTransverseMomentumLine.c_str();
  char* endptr;
  double d = strtod(startptr, &endptr);
  if(*endptr == '\0' && endptr != startptr)
  {
		MinVisPtCut cuts[2];
		cuts[0].type_ = cuts[1].type_ = MinVisPtCut::TAU;
    cuts[0].pt_ = cuts[1].pt_ = d;
		cuts[0].index_ = 0; cuts[1].index_ = 1;
		minVisPtCuts_.push_back(std::vector<MinVisPtCut>(cuts, cuts+2));
  }
  else
  {
	  // string has new format: parse the minvistransversemomentum string
		for(std::string::size_type prev = 0, pos = 0; prev < minVisibleTransverseMomentumLine.length(); prev = pos + 1)
		{
			pos = minVisibleTransverseMomentumLine.find(';', prev);
			if(pos == std::string::npos) pos = minVisibleTransverseMomentumLine.length();

			std::string sub = minVisibleTransverseMomentumLine.substr(prev, pos - prev);
			std::vector<MinVisPtCut> cuts;
			const char* sub_c = sub.c_str();
			while(*sub_c != '\0')
			{
				const char* sep = std::strchr(sub_c, '_');
				if(sep == NULL) throw cms::Exception("Configuration") << "Minimum transverse parameter string must contain an underscore to separate type from pt threshold" << std::endl;
				std::string type(sub_c, sep);

				MinVisPtCut cut;
				if(type == "elec1") { cut.type_ = MinVisPtCut::ELEC; cut.index_ = 0; }
				else if(type == "mu1") { cut.type_ = MinVisPtCut::MU; cut.index_ = 0; }
				else if(type == "had1") { cut.type_ = MinVisPtCut::HAD; cut.index_ = 0; }
				else if(type == "tau1") { cut.type_ = MinVisPtCut::TAU; cut.index_ = 0; }
				else if(type == "elec2") { cut.type_ = MinVisPtCut::ELEC; cut.index_ = 1; }
				else if(type == "mu2") { cut.type_ = MinVisPtCut::MU; cut.index_ = 1; }
				else if(type == "had2") { cut.type_ = MinVisPtCut::HAD; cut.index_ = 1; }
				else if(type == "tau2") { cut.type_ = MinVisPtCut::TAU; cut.index_ = 1; }
				else throw cms::Exception("Configuration") << "'" << type << "' is not a valid type. Allowed values are elec1,mu1,had1,tau1,elec2,mu2,had2,tau2" << std::endl;

				char* endptr;
				cut.pt_ = strtod(sep + 1, &endptr);
				if(endptr == sep + 1) throw cms::Exception("Configuration") << "No pt threshold given" << std::endl;

				cuts.push_back(cut);
				sub_c = endptr;
			}
		minVisPtCuts_.push_back(cuts);
		}
  }

	edm::Service<TFileService> fileService_;
        if(fileService_.isAvailable()) {
          outTree = fileService_->make<TTree>( "event_generation","This tree stores information about the event generation");
          outTree->Branch("attempts",&attempts,"attempts/I");
        }

        edm::Service<RandomNumberGenerator> rng;
        if(!rng.isAvailable()) {
          throw cms::Exception("Configuration")
            << "The RandomNumberProducer module requires the RandomNumberGeneratorService\n"
            "which appears to be absent.  Please add that service to your configuration\n"
            "or remove the modules that require it." << std::endl;
        } 
        // this is a global variable defined in GeneratorInterface/ExternalDecays/src/ExternalDecayDriver.cc
        decayRandomEngine = &rng->getEngine();

	edm::LogInfo("Replacer") << "generatorMode = "<< generatorMode_<< "\n";

	return;
}

ParticleReplacerClass::~ParticleReplacerClass()
{
	// do anything here that needs to be done at desctruction time
	// (e.g. close files, deallocate resources etc.)
}

// ------------ method called to produce the data  ------------
std::auto_ptr<HepMC::GenEvent> ParticleReplacerClass::produce(const reco::MuonCollection& muons, const reco::Vertex *pvtx, const HepMC::GenEvent *genEvt)
{
	using namespace edm;
	using namespace std;
	using namespace HepMC;

        if(pvtx != 0)
          throw cms::Exception("Configuration") << "ParticleReplacerClass does NOT support using primary vertex as the origin for taus" << std::endl;

	HepMC::GenEvent * evt=0;

	GenVertex * zvtx = new GenVertex();

	reco::GenParticle * part1=0;
	reco::GenParticle * part2=0;

	/// 2) transform the muons to the desired particles
	std::vector<reco::Particle> particles;	
	switch (transformationMode_)
	{
		case 0:	// mumu->mumu
		{
			if (muons.size()!=2)
			{
				LogError("Replacer") << "the decay mode Z->mumu requires exactly two muons, aborting processing";
				return std::auto_ptr<HepMC::GenEvent>(0);
			}
	
			targetParticleMass_  = 0.105658369;
			targetParticlePdgID_ = 13;
				
			reco::Muon muon1 = muons.at(0);
			reco::Muon muon2 = muons.at(1);
			reco::Particle tau1(muon1.charge(), muon1.p4(), muon1.vertex(), muon1.pdgId(), 0, true);
			reco::Particle tau2(muon2.charge(), muon2.p4(), muon2.vertex(), muon2.pdgId(), 0, true);
			particles.push_back(tau1);
			particles.push_back(tau2);
			break;
		} 
		case 1:	// mumu->tautau
		{
			if (muons.size()!=2)
			{
				LogError("Replacer") << "the decay mode Z->tautau requires exactly two muons, aborting processing";
				return std::auto_ptr<HepMC::GenEvent>(0);
			}

			targetParticleMass_  = 1.77690;
			targetParticlePdgID_ = 15;
			
			reco::Muon muon1 = muons.at(0);
			reco::Muon muon2 = muons.at(1);
			reco::Particle tau1(muon1.charge(), muon1.p4(), muon1.vertex(), muon1.pdgId(), 0, true);
			reco::Particle tau2(muon2.charge(), muon2.p4(), muon2.vertex(), muon2.pdgId(), 0, true);
			transformMuMu2TauTau(&tau1, &tau2);
			particles.push_back(tau1);
			particles.push_back(tau2);			
			break;
		}
    case 2: // mumu->taunu (W boson)
    {
      if (muons.size()!=2)
      {
        LogError("Replacer") << "the decay mode Z->tautau requires exactly two muons, aborting processing";
        return std::auto_ptr<HepMC::GenEvent>(0);
      }

      targetParticleMass_  = 1.77690;
      targetParticlePdgID_ = 15;
      
      reco::Muon muon1 = muons.at(0);
      reco::Muon muon2 = muons.at(1);
      reco::Particle tau1(muon1.charge(), muon1.p4(), muon1.vertex(), muon1.pdgId(), 0, true);
      reco::Particle tau2(muon2.charge(), muon2.p4(), muon2.vertex(), muon2.pdgId(), 0, true);
      transformMuMu2TauNu(&tau1, &tau2);
      particles.push_back(tau1);
      particles.push_back(tau2);                      
      break;
    }  
    case 3: // mu-nu->tau-nu
    {
      if (muons.size()!=2)
      {
        LogError("Replacer") << "transformation mode mu-nu ->tau-nu - wrong input";
        return std::auto_ptr<HepMC::GenEvent>(0);
      }

      targetParticleMass_  = 1.77690;
      targetParticlePdgID_ = 15;
      int targetParticlePdgIDNu_ = 16;
      
      reco::Muon muon1 = muons.at(0);
      reco::Muon::LorentzVector l(muon1.px(), muon1.py(), muon1.pz(), 
                                sqrt(
                                muon1.px()*muon1.px()+
                                muon1.py()*muon1.py()+
                                muon1.pz()*muon1.pz()+targetParticleMass_*targetParticleMass_));

      reco::Particle tau1(muon1.charge(), l, muon1.vertex(), targetParticlePdgID_*std::abs(muon1.pdgId())/muon1.pdgId() 
                                , 0, true
                         );
      tau1.setStatus(1);
      particles.push_back(tau1);

      reco::Muon nu    = muons.at(1);
      reco::Particle nutau( 0, nu.p4(), nu.vertex(), -targetParticlePdgIDNu_*std::abs(muon1.pdgId())/muon1.pdgId(), 0, true);
      nutau.setStatus(1);
      particles.push_back(nutau);
 
      break;
    }  

	}
	
	if (particles.size()==0)
	{
		LogError("Replacer") << "the creation of the new particles failed somehow";	
		return std::auto_ptr<HepMC::GenEvent>(0);
	}
	else
	{
		LogInfo("Replacer") << particles.size() << " particles found, continue processing";
	}

	/// 3) prepare the event
	if (genEvt)
	{
	
			evt = new HepMC::GenEvent(*genEvt);
	
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
	else
	{
		reco::Particle::LorentzVector mother_particle_p4;
		for (std::vector<reco::Particle>::const_iterator it=particles.begin();it!=particles.end();it++)
			mother_particle_p4+=it->p4();

		reco::Particle::Point production_point = particles.begin()->vertex();

                GenVertex* startVtx = new GenVertex(FourVector(production_point.x()*10,production_point.y()*10,production_point.z()*10,0));
                startVtx->add_particle_in( new GenParticle( FourVector(0,0,7000,7000), 2212, 3 ) );
                startVtx->add_particle_in( new GenParticle( FourVector(0,0,-7000,7000), 2212, 3 ) );

                GenVertex * decayvtx = new GenVertex(FourVector(production_point.x()*10,production_point.y()*10,production_point.z()*10,0));
		HepMC::GenParticle * mother_particle = new HepMC::GenParticle((FourVector)mother_particle_p4, motherParticleID_, (generatorMode_=="Pythia" ? 3 : 2), Flow(), Polarization(0,0));
                if (transformationMode_ == 3) {
                  //std::cout << "Overriding mother particle id\n" << std::endl;
                  int muPDG = particles.begin()->pdgId();
                  int id = -24*muPDG/std::abs(muPDG);
                  mother_particle->set_pdg_id(id);
                }

                startVtx->add_particle_out(mother_particle);
                decayvtx->add_particle_in(mother_particle);
  		evt = new HepMC::GenEvent();
		for (std::vector<reco::Particle>::const_iterator it=particles.begin();it!=particles.end();it++)
		{
                        //std::cout << "XXX" << it->p4().pt() << " " << it->pdgId() << std::endl;
			decayvtx->add_particle_out(new HepMC::GenParticle((FourVector)it->p4(), it->pdgId(), 1, Flow(), Polarization(0,0)));			
		}

		evt->add_vertex(startVtx);
		evt->add_vertex(decayvtx);
	}
	repairBarcodes(evt);
	
	HepMC::GenEvent * retevt = 0;
	HepMC::GenEvent * tempevt = 0;

	/// 3) process the event
	int nr_of_trials=0;

	unsigned int cntVisPt_all = 0;
	unsigned int cntVisPt_pass = 0;
	
        HepMC::IO_HEPEVT conv;
	for (int i = 0; i<maxNumberOfAttempts_; i++)
	{
		++cntVisPt_all;
		if (generatorMode_ == "Pythia")	// Pythia
		{
			LogError("Replacer") << "Pythia is currently not supported!";
			return std::auto_ptr<HepMC::GenEvent>(evt);
		}

		if (generatorMode_ == "Tauola")	// TAUOLA
		{
			conv.write_event(evt);
			tempevt=tauola_->decay(evt);
		}

		if (testEvent(tempevt))
		{
			if (retevt==0) {
			  retevt=tempevt;
                        } else {
                          delete tempevt; 
                        }
			++cntVisPt_pass;
		} else {
                  delete tempevt;
                }
	}

	tried = cntVisPt_all;
	passed = cntVisPt_pass;

	std::cout << /*minVisibleTransverseMomentum_ <<*/ " " << cntVisPt_pass << "\t" << cntVisPt_all << "\n";
	if (!retevt)
	{
		LogError("Replacer") << "failed to create an event which satisfies the minimum visible transverse momentum cuts ";
		attempts=-1;
                if(outTree) outTree->Fill();
		return std::auto_ptr<HepMC::GenEvent>(0);
	}
	attempts=nr_of_trials;
	if(outTree) outTree->Fill();	

	// recover the status codes
	if (genEvt)
	{
		for (GenEvent::particle_iterator it=retevt->particles_begin();it!=retevt->particles_end();it++)
		{
			if ((*it)->end_vertex())
				(*it)->set_status(2);
			else
				(*it)->set_status(1);
		}
	}

        std::auto_ptr<HepMC::GenEvent> ret(retevt);

	if (printEvent_)
		retevt->print(std::cout);

	delete part1;
	delete part2;
	delete zvtx;
	delete evt;
	return ret;
}

// ------------ method called once each job just before starting event loop  ------------
void ParticleReplacerClass::beginRun(const edm::Run& iRun,const edm::EventSetup& iSetup)
{
	tauola_->init(iSetup);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ParticleReplacerClass::endJob()
{
	tauola_->statistics();
}

bool ParticleReplacerClass::testEvent(HepMC::GenEvent * evt)
{
	using namespace HepMC;
        using namespace edm;
	
	if (minVisPtCuts_.empty()) //ibleTransverseMomentum_<=0)
		return true;

  std::vector<double> mus;
  std::vector<double> elecs;
  std::vector<double> hads;
  std::vector<double> taus;

  for (GenEvent::particle_iterator it=evt->particles_begin();it!=evt->particles_end();it++)
	{
		if (abs((*it)->pdg_id())==15 && (*it)->end_vertex())
    {
    	FourVector vis_mom();
    	math::PtEtaPhiMLorentzVector visible_momentum;
    	std::queue<const GenParticle *> decaying_particles;
			decaying_particles.push(*it);
			int t=0;
      enum { ELEC, MU, HAD } type = HAD;
			while(!decaying_particles.empty() && (++t < 30))
			{
				const GenParticle * front = decaying_particles.front();
	    	decaying_particles.pop();

				if (!front->end_vertex())
				{
					int pdgId=abs(front->pdg_id());
					if (pdgId>10 && pdgId!=12 && pdgId!=14 && pdgId!=16)
						visible_momentum+=(math::PtEtaPhiMLorentzVector)front->momentum();

          if(pdgId == 11) type = ELEC;
          if(pdgId == 13) type = MU;
				}
				else
				{
					GenVertex * temp_vert = front->end_vertex();
					for (GenVertex::particles_out_const_iterator it2=temp_vert->particles_out_const_begin();it2!=temp_vert->particles_out_const_end();it2++)
						decaying_particles.push((*it2));
				}
    	}

      double vis_pt = visible_momentum.pt();
      taus.push_back(vis_pt);
      if(type == MU) mus.push_back(vis_pt);
      if(type == ELEC) elecs.push_back(vis_pt);
      if(type == HAD) hads.push_back(vis_pt);
		}
	}

  std::sort(taus.begin(), taus.end(), std::greater<double>());
  std::sort(elecs.begin(), elecs.end(), std::greater<double>());
  std::sort(mus.begin(), mus.end(), std::greater<double>());
  std::sort(hads.begin(), hads.end(), std::greater<double>());

  for(std::vector<std::vector<MinVisPtCut> >::const_iterator iter = minVisPtCuts_.begin(); iter != minVisPtCuts_.end(); ++iter)
  {
    std::vector<MinVisPtCut>::const_iterator iter2;
    for(iter2 = iter->begin(); iter2 != iter->end(); ++iter2)
    {
      std::vector<double>* collection;
      switch(iter2->type_)
      {
      case MinVisPtCut::ELEC: collection = &elecs; break;
      case MinVisPtCut::MU: collection = &mus; break;
      case MinVisPtCut::HAD: collection = &hads; break;
      case MinVisPtCut::TAU: collection = &taus; break;
      default: assert(false); break;
      }

      // subcut fail
      if(iter2->index_ >= collection->size() || (*collection)[iter2->index_] < iter2->pt_)
        break;
    }

    // no subcut failed: This cut passed
    if(iter2 == iter->end())
      return true;
  }

  LogInfo("Replacer") << "refusing the event as the sum of the visible transverse momenta is too small\n";
 	return false;
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
	for (GenEvent::vertex_iterator it=evt->vertices_begin(), next;it!=evt->vertices_end();it=next)
	{
		next=it;++next;
		while (!(*it)->suggest_barcode(-1*(++max_barc)))
			;
	}

	max_barc=0;
	for (GenEvent::particle_iterator it=evt->particles_begin(), next;it!=evt->particles_end();it=next)
	{
		next=it;++next;
		while (!(*it)->suggest_barcode(++max_barc))
			;
	}
}

///	transform a muon pair into a tau pair
void ParticleReplacerClass::transformMuMu2TauTau(reco::Particle * muon1, reco::Particle * muon2)
{
	using namespace edm;
	using namespace reco;
	using namespace std;
	
	reco::Particle::LorentzVector muon1_momentum = muon1->p4();
	reco::Particle::LorentzVector muon2_momentum =  muon2->p4();
	reco::Particle::LorentzVector z_momentum = muon1_momentum + muon2_momentum;

	ROOT::Math::Boost booster(z_momentum.BoostToCM());
	ROOT::Math::Boost invbooster(booster.Inverse());
	
	reco::Particle::LorentzVector Zb = booster(z_momentum);

	reco::Particle::LorentzVector muon1b = booster(muon1_momentum);
	reco::Particle::LorentzVector muon2b = booster(muon2_momentum);
	
	double tau_mass2 = targetParticleMass_*targetParticleMass_;

	double muonxb_mom2 = muon1b.x()*muon1b.x() + muon1b.y()*muon1b.y() + muon1b.z() * muon1b.z();
	double tauxb_mom2 = 0.25 * Zb.t() * Zb.t() - tau_mass2;

	float scaling1 = sqrt(tauxb_mom2 / muonxb_mom2);
	float scaling2 = scaling1;

	float tauEnergy= Zb.t() / 2.;

	if (tauEnergy*tauEnergy<tau_mass2)
		return;
	
	reco::Particle::LorentzVector tau1b_mom = reco::Particle::LorentzVector(scaling1*muon1b.x(),scaling1*muon1b.y(),scaling1*muon1b.z(),tauEnergy);
	reco::Particle::LorentzVector tau2b_mom = reco::Particle::LorentzVector(scaling2*muon2b.x(),scaling2*muon2b.y(),scaling2*muon2b.z(),tauEnergy);

	reco::Particle::LorentzVector tau1_mom = (invbooster(tau1b_mom));
	reco::Particle::LorentzVector tau2_mom = (invbooster(tau2b_mom));
	
	// some additional checks
	// the following tests guarantee a deviation of less
	// than 0.1% for the following values of the original
	// muons and the placed taus
	//	invariant mass
	//	transverse momentum
	assert(std::abs((muon1_momentum+muon2_momentum).mass()-(tau1_mom+tau2_mom).mass())/(muon1_momentum+muon2_momentum).mass()<0.001);
	assert(std::abs((muon1_momentum+muon2_momentum).pt()-(tau1_mom+tau2_mom).pt())/(muon1_momentum+muon2_momentum).pt()<0.001);

	muon1->setP4(tau1_mom);
	muon2->setP4(tau2_mom);

	muon1->setPdgId(targetParticlePdgID_*muon1->pdgId()/abs(muon1->pdgId()));
	muon2->setPdgId(targetParticlePdgID_*muon2->pdgId()/abs(muon2->pdgId()));

	muon1->setStatus(1);
	muon2->setStatus(1);

	return;
}
///     transform a muon pair into tau nu (as coming from a W boson)
void ParticleReplacerClass::transformMuMu2TauNu(reco::Particle * part1, reco::Particle * part2)
{
	using namespace edm;
	using namespace reco;
	using namespace std;

	reco::Particle::LorentzVector muon1_momentum = part1->p4();
	reco::Particle::LorentzVector muon2_momentum =  part2->p4();
	reco::Particle::LorentzVector z_momentum = muon1_momentum + muon2_momentum;

	ROOT::Math::Boost booster(z_momentum.BoostToCM());
	ROOT::Math::Boost invbooster(booster.Inverse());

	reco::Particle::LorentzVector Zb = booster(z_momentum);

	const double breitWignerWidth_Z = 2.4952;
	const double breitWignerWidth_W = 2.141;
	const double knownMass_W = 80.398;
	const double knownMass_Z = 91.1876;
		      
	double Wb_mass = ( Zb.mass() - knownMass_Z ) * ( breitWignerWidth_W / breitWignerWidth_Z ) + knownMass_W;
	std::cout << "Wb_mass: " << Wb_mass << "\n";

	reco::Particle::LorentzVector muon1b = booster(muon1_momentum);
	reco::Particle::LorentzVector muon2b = booster(muon2_momentum);

	double tau_mass2 = targetParticleMass_*targetParticleMass_;

	double muonxb_mom2 = muon1b.x()*muon1b.x() + muon1b.y()*muon1b.y() + muon1b.z() * muon1b.z();
	double tauxb_mom2 = 0.25 * Zb.t() * Zb.t() - tau_mass2;

	float scaling1 = sqrt(tauxb_mom2 / muonxb_mom2) * Wb_mass/Zb.mass();
	float scaling2 = scaling1;

	float tauEnergy= Zb.t() / 2.;

	if (tauEnergy*tauEnergy<tau_mass2)
		      return;

	reco::Particle::LorentzVector tau1b_mom = reco::Particle::LorentzVector(scaling1*muon1b.x(),scaling1*muon1b.y(),scaling1*muon1b.z(),tauEnergy* Wb_mass/Zb.mass());
	reco::Particle::LorentzVector tau2b_mom = reco::Particle::LorentzVector(scaling2*muon2b.x(),scaling2*muon2b.y(),scaling2*muon2b.z(),tauEnergy* Wb_mass/Zb.mass());

	std::cout << "muon1b_momentum: " << muon1b << "\n";
	std::cout << "muon2b_momentum: " << muon2b << "\n";

	std::cout << "tau1b_momentum: " << tau1b_mom << "\n";
	std::cout << "tau2b_momentum: " << tau2b_mom << "\n";

	std::cout << "zb_momentum: " << Zb << "\n";
	std::cout << "wb_momentum: " << (tau1b_mom+tau2b_mom) << "\n";
		              
	// some checks
	// the following test guarantees a deviation
	// of less than 0.1% for phi and theta for the
	// original muons and the placed taus
	// (in the centre-of-mass system of the z boson)
	assert((muon1b.phi()-tau1b_mom.phi())/muon1b.phi()<0.001);
	assert((muon2b.phi()-tau2b_mom.phi())/muon2b.phi()<0.001);
	assert((muon1b.theta()-tau1b_mom.theta())/muon1b.theta()<0.001);
	assert((muon2b.theta()-tau2b_mom.theta())/muon2b.theta()<0.001);        

	reco::Particle::LorentzVector tau1_mom = (invbooster(tau1b_mom));
	reco::Particle::LorentzVector tau2_mom = (invbooster(tau2b_mom));

	// some additional checks
	// the following tests guarantee a deviation of less
	// than 0.1% for the following values of the original
	// muons and the placed taus
	//      invariant mass
	//      transverse momentum
	//assert(((muon1_momentum+muon1_momentum).mass()-(tau1_mom+tau2_mom).mass())/(muon1_momentum+muon1_momentum).mass()<0.001);
	//assert(((muon1_momentum+muon2_momentum).pt()-(tau1_mom+tau2_mom).pt())/(muon1_momentum+muon1_momentum).pt()<0.001);

	part1->setP4(tau1_mom);
	part2->setP4(tau2_mom);

	part1->setPdgId(15*part1->pdgId()/abs(part1->pdgId()));
	part2->setPdgId(16*part2->pdgId()/abs(part2->pdgId()));

	part1->setStatus(1);
	part2->setStatus(1);

	return;
}


//define this as a plug-in
//DEFINE_FWK_MODULE(Replacer);

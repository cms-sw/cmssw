// -*- C++ -*-
//
// Package:    Replacer
// Class:      Replacer
//
/**\class Replacer Replacer.cc TauAnalysis/MCEmbeddingTools/src/Replacer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Manuel Zeise
//
//


// system include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <stack>



#include "Math/LorentzVector.h"
#include "Math/VectorUtil.h"
#include "PhysicsTools/UtilAlgos/interface/DeltaR.h"

//#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "HepMC/IO_HEPEVT.h"
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"
#include "HepMC/PythiaWrapper.h"
#include "TauAnalysis/MCEmbeddingTools/interface/extraPythia.h"

#include "GeneratorInterface/CommonInterface/interface/TauolaInterface.h"
#include "GeneratorInterface/CommonInterface/interface/TauolaWrapper.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

// #include "TauAnalysis/MCEmbeddingTools/interface/MuonSelector.h"

// #include <iostream>
using namespace std;
using namespace edm;

class ParticleReplacerClass : public edm::EDProducer
{
public:
	explicit ParticleReplacerClass(const edm::ParameterSet&);
	~ParticleReplacerClass();

	virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup);
	virtual void beginJob(const edm::EventSetup& );
	virtual void endJob();
	
private:
	void initPythia(const edm::ParameterSet& pset);
	void initPythiaTauola(const edm::ParameterSet& pset);

	/// Interface to the PYGIVE/TXGIVE pythia routine, with add'l protections
	bool call_pygive(const std::string& iParm );
	bool call_txgive(const std::string& iParm );
	bool call_txgive_init();

	HepMC::GenEvent * processEventWithTauola(HepMC::GenEvent * evt);
	HepMC::GenEvent * processEventWithPythia(HepMC::GenEvent * evt);
//	bool makeEvent(HepMC::GenEvent*,const reco::MuonCollection&,HepMC::GenVertex*);

	void cleanEvent(HepMC::GenEvent * evt, HepMC::GenVertex * vtx);
	void repairBarcodes(HepMC::GenEvent * evt);

	/// replace mode:
	//	0	fullEvent	(incl. all other particles)
	//	1	taus only	(create an event only with the two decaying taus)
	unsigned int replacementMode_;
	unsigned int generatorMode_;
	bool noInitialisation_;

	string HepMCSource_;
	string selectedParticles_;

	int motherParticleID_;
	bool useExternalGenerators_ ;
	bool useTauola_ ;
	bool useTauolaPolarization_ ;
	TauolaInterface tauola_;

	bool printEvent_;
// 	MuonSelector* muonSelector;
};



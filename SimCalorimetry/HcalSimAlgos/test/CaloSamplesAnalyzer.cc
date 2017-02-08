// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

//ROOT headers
#include <TTree.h>

//calo headers
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HcalTestNumbering.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h" 
 
//STL headers 
#include <vector>
#include <unordered_map>
#include <string>

//
// class declaration
//

class CaloSamplesAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
	public:
		explicit CaloSamplesAnalyzer(const edm::ParameterSet&);
		~CaloSamplesAnalyzer();
	
		static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
		
		struct CaloNtuple {
			unsigned run = 0;
			unsigned lumiblock = 0;
			unsigned long long event = 0;
			unsigned id = 0;
			int subdet = 0;
			int ieta = 0;
			int iphi = 0;
			int depth = 0;
			double samplingFactor = 0.;
			double fCtoGeV = 0.;
			double photoelectronsToAnalog = 0.;
			double simHitToPhotoelectrons = 0.;
			double tof = 0.;
			std::vector<double> energy;
			std::vector<double> time;
			std::vector<double> signalTot;
			std::vector<double> signalTotPrecise;
		};
	
	private:
		void beginJob() override;
		void doBeginRun_(const edm::Run&, const edm::EventSetup&) override;
		void analyze(const edm::Event&, const edm::EventSetup&) override;
		void doEndRun_(const edm::Run&, const edm::EventSetup&) override {}
		void endJob() override {}
		
		// ----------member data ---------------------------
		edm::Service<TFileService> fs;
		TTree* tree;
		const CaloGeometry* theGeometry;
		const HcalDDDRecConstants * theRecNumber;
		CaloHitResponse* theResponse;
		HcalSimParameterMap * theParameterMap;
		//for tree branches
		CaloNtuple entry;
		std::unordered_map<unsigned,CaloNtuple> treemap;
		//misc. params
		bool TestNumbering;
		//tokens
		edm::EDGetTokenT<std::vector<PCaloHit>> tok_sim;
		edm::EDGetTokenT<std::vector<CaloSamples>> tok_calo;
};

//
// constructors and destructor
//
CaloSamplesAnalyzer::CaloSamplesAnalyzer(const edm::ParameterSet& iConfig) :
	tree(NULL), theGeometry(NULL), theRecNumber(NULL), theResponse(new CaloHitResponse(NULL,(CaloShapes*)NULL)), theParameterMap(new HcalSimParameterMap(iConfig)),
	tok_sim(consumes<std::vector<PCaloHit>>(edm::InputTag(iConfig.getParameter<std::string>("hitsProducer"), "HcalHits"))),
	tok_calo(consumes<std::vector<CaloSamples>>(iConfig.getParameter<edm::InputTag>("CaloSamplesTag")))
{
	usesResource("TFileService");
}


CaloSamplesAnalyzer::~CaloSamplesAnalyzer()
{
	delete theResponse;
	delete theParameterMap;
}


//
// member functions
//

void CaloSamplesAnalyzer::beginJob()
{
	tree = fs->make<TTree>("tree","tree");
	
	tree->Branch("run"                   , &entry.run                   , "run/i");
	tree->Branch("lumiblock"             , &entry.lumiblock             , "lumiblock/i");
	tree->Branch("event"                 , &entry.event                 , "event/l");
	tree->Branch("id"                    , &entry.id                    , "id/i");
	tree->Branch("subdet"                , &entry.subdet                , "subdet/I");
	tree->Branch("ieta"                  , &entry.ieta                  , "ieta/I");
	tree->Branch("iphi"                  , &entry.iphi                  , "iphi/I");
	tree->Branch("depth"                 , &entry.depth                 , "depth/I");
	tree->Branch("samplingFactor"        , &entry.samplingFactor        , "samplingFactor/D");
	tree->Branch("fCtoGeV"               , &entry.fCtoGeV               , "fCtoGeV/D");
	tree->Branch("photoelectronsToAnalog", &entry.photoelectronsToAnalog, "photoelectronsToAnalog/D");
	tree->Branch("simHitToPhotoelectrons", &entry.simHitToPhotoelectrons, "simHitToPhotoelectrons/D");
	tree->Branch("energy"                , "vector<double>"             , &entry.energy, 32000, 0);
	tree->Branch("time"                  , "vector<double>"             , &entry.time, 32000, 0);
	tree->Branch("tof"                   , &entry.tof                   , "tof/D");
	tree->Branch("signalTot"             , "vector<double>"             , &entry.signalTot, 32000, 0);
	tree->Branch("signalTotPrecise"      , "vector<double>"             , &entry.signalTotPrecise, 32000, 0);
}

void CaloSamplesAnalyzer::doBeginRun_(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
	edm::ESHandle<CaloGeometry> geometry;
	iSetup.get<CaloGeometryRecord>().get(geometry);
	edm::ESHandle<HcalDDDRecConstants> pHRNDC;
	iSetup.get<HcalRecNumberingRecord>().get(pHRNDC);

	// See if it's been updated
	if (&*geometry != theGeometry) {
		theGeometry = &*geometry;
		theRecNumber= &*pHRNDC;
		theResponse->setGeometry(theGeometry);
	}
}

// ------------ method called on each new Event  ------------
void
CaloSamplesAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	treemap.clear();

	//conditions
	edm::ESHandle<HcalDbService> conditions;
	iSetup.get<HcalDbRecord>().get(conditions);
	theParameterMap->setDbService(conditions.product());
	
	// Event information
	edm::EventAuxiliary aux = iEvent.eventAuxiliary();
	unsigned run = aux.run();
	unsigned lumiblock = aux.luminosityBlock();
	unsigned long long event = aux.event();
	
	edm::Handle<std::vector<CaloSamples>> h_CS;
	iEvent.getByToken(tok_calo,h_CS);
	
	//loop over CaloSamples, extract info per channel
	for(const auto& iCS : *(h_CS.product())){
		DetId did(iCS.id());
		if(did.det()!=DetId::Hcal) continue;
		HcalDetId hid(did);
		CaloNtuple& ntup = treemap[hid.rawId()];
		//check for existence of entry
		if(ntup.id==0){
			ntup.id = hid.rawId();
			ntup.subdet = hid.subdet();
			ntup.ieta = hid.ieta();
			ntup.iphi = hid.iphi();
			ntup.depth = hid.depth();
			//get sim parameters
			if(hid.subdet()==HcalForward){
				const HFSimParameters& pars = dynamic_cast<const HFSimParameters&>(theParameterMap->simParameters(hid));
				ntup.samplingFactor = pars.samplingFactor();
				ntup.fCtoGeV = pars.fCtoGeV(hid);
				ntup.photoelectronsToAnalog = pars.photoelectronsToAnalog(hid);
				ntup.simHitToPhotoelectrons = pars.simHitToPhotoelectrons(hid);
			}
			else {
				const HcalSimParameters& pars = dynamic_cast<const HcalSimParameters&>(theParameterMap->simParameters(hid));
				ntup.samplingFactor = pars.samplingFactor(hid);
				ntup.fCtoGeV = pars.fCtoGeV(hid);
				ntup.photoelectronsToAnalog = pars.photoelectronsToAnalog(hid);
				ntup.simHitToPhotoelectrons = pars.simHitToPhotoelectrons(hid);			
			}
			//get time of flight
			ntup.tof = theResponse->timeOfFlight(hid);
		}
		//get pulse info every time
		if(ntup.signalTot.size()==0) ntup.signalTot.resize(iCS.size(),0.0);
		for(int i = 0; i < iCS.size(); ++i){
			ntup.signalTot[i] += iCS[i];
		}
		if(ntup.signalTotPrecise.size()==0) ntup.signalTotPrecise.resize(iCS.preciseSize(),0.0);
		for(int i = 0; i < iCS.preciseSize(); ++i){
			ntup.signalTotPrecise[i] += iCS.preciseAt(i);
		}
	}
	
	edm::Handle<std::vector<PCaloHit>> h_SH;
	iEvent.getByToken(tok_sim,h_SH);
	
	//loop over simhits, extract info per channel
	for(const auto& iSH : *(h_SH.product())){
		HcalDetId hid;
		unsigned int id = iSH.id();
		if(TestNumbering){
			int subdet, z, depth, eta, phi, lay;
			HcalTestNumbering::unpackHcalIndex(id, subdet, z, depth, eta, phi, lay);
			int sign = (z==0) ? -1 : 1;
			HcalDDDRecConstants::HcalID cid = theRecNumber->getHCID(subdet, eta, phi, lay, depth);
			hid = HcalDetId((HcalSubdetector)subdet, sign*cid.eta, cid.phi, cid.depth);
		}
		else hid = HcalDetId(id);
		auto ntupIt = treemap.find(hid.rawId());
		if(ntupIt==treemap.end()) continue;
		CaloNtuple& ntup = ntupIt->second;
		//append simhit info
		ntup.energy.push_back(iSH.energy());
		ntup.time.push_back(iSH.time());
	}
	
	//one tree entry per map entry
	entry.run = run;
	entry.lumiblock = lumiblock;
	entry.event = event;
	for(const auto& ntup : treemap){
		entry.id = ntup.second.id;
		entry.subdet = ntup.second.subdet;
		entry.ieta = ntup.second.ieta;
		entry.iphi = ntup.second.iphi;
		entry.depth = ntup.second.depth;
		entry.samplingFactor = ntup.second.samplingFactor;
		entry.fCtoGeV = ntup.second.fCtoGeV;
		entry.photoelectronsToAnalog = ntup.second.photoelectronsToAnalog;
		entry.simHitToPhotoelectrons = ntup.second.simHitToPhotoelectrons;
		entry.tof = ntup.second.tof;
		entry.energy = ntup.second.energy;
		entry.time = ntup.second.time;
		entry.signalTot = ntup.second.signalTot;
		entry.signalTotPrecise = ntup.second.signalTotPrecise;
		tree->Fill();
	}
}
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
CaloSamplesAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(CaloSamplesAnalyzer);

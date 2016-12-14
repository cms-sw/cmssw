// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

//ROOT headers
#include <TProfile.h>
#include <TH1.h>
#include <TF1.h>

//SiPM headers
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPM.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h" 
 
//STL headers 
#include <vector>
#include <utility>
#include <sstream>
#include <string>
#include <algorithm>
#include <cmath>

//
// class declaration
//

class SiPMNonlinearityAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
	public:
		explicit SiPMNonlinearityAnalyzer(const edm::ParameterSet&);
		~SiPMNonlinearityAnalyzer();
	
		static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
	
	private:
		void beginJob() override {}
		void analyze(const edm::Event&, const edm::EventSetup&) override;
		void endJob() override {}
		
		// ----------member data ---------------------------
		edm::Service<TFileService> fs;
		unsigned pixels;
		unsigned npeMin, npeMax, npeStep, nReps;
		unsigned nBins, binMin, binMax;
		double tau, dt;
		unsigned nPreciseBins;
		std::string fitname;
};

//
// constructors and destructor
//
SiPMNonlinearityAnalyzer::SiPMNonlinearityAnalyzer(const edm::ParameterSet& iConfig) : 
	pixels(iConfig.getParameter<unsigned>("pixels")), npeMin(iConfig.getParameter<unsigned>("npeMin")), npeMax(iConfig.getParameter<unsigned>("npeMax")), 
	npeStep(iConfig.getParameter<unsigned>("npeStep")), nReps(iConfig.getParameter<unsigned>("nReps")), 
	nBins(iConfig.getParameter<unsigned>("nBins")), binMin(iConfig.getParameter<unsigned>("binMin")), binMax(iConfig.getParameter<unsigned>("binMax")), 
	tau(iConfig.getParameter<double>("tau")), dt(iConfig.getParameter<double>("dt")),
	nPreciseBins(iConfig.getParameter<unsigned>("nPreciseBins")), fitname(iConfig.getParameter<std::string>("fitname"))
{
	usesResource("TFileService");
}


SiPMNonlinearityAnalyzer::~SiPMNonlinearityAnalyzer()
{
}


//
// member functions
//

// ------------ method called on each new Event  ------------
void
SiPMNonlinearityAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	edm::Service<edm::RandomNumberGenerator> rng;
	CLHEP::HepRandomEngine* engine = &rng->getEngine(iEvent.streamID());
	
	//instantiate SiPM classes
	HcalSiPM sipm(pixels,tau);
	
	fs->file().cd();
	TProfile* profio = fs->make<TProfile>("input_vs_output","",nBins,binMin,binMax);
	profio->GetXaxis()->SetTitle("input [pe]");
	profio->GetYaxis()->SetTitle("output [pe]");
	TProfile* profoi = fs->make<TProfile>("output_vs_input","",nBins,binMin,binMax);
	profoi->GetXaxis()->SetTitle("output [pe]");
	profoi->GetYaxis()->SetTitle("input [pe]");
	TH1F* corr = fs->make<TH1F>("correction","",nBins,binMin,binMax);
	corr->GetXaxis()->SetTitle("output [pe]");
	corr->GetYaxis()->SetTitle("correction factor");
	
	//shift to TS 4
	static const int tsOffset = 100;
	unsigned npe = npeMin;
	while(npe <= npeMax){
		if(npe%1000==0) std::cout << "npe = " << npe << std::endl;
		for(unsigned rep = 0; rep < nReps; ++rep){
			//smear pes according to Y11 time distribution
			std::vector<unsigned> photonHist(nPreciseBins,0);
			for(unsigned pe = 0; pe < npe; ++pe){
				double t_pe = HcalPulseShapes::generatePhotonTime(engine);
				int t_bin = int(t_pe + tsOffset + 0.5);
				if(t_bin > 0 and (static_cast<unsigned>(t_bin) < photonHist.size())) photonHist[t_bin] += 1;
			}
			
			//reset SiPM
			sipm.setNCells(pixels);
			
			//evaluate SiPM response
			double elapsedTime = 0.;
			unsigned sumPE = 0;
			double sumHits = 0.;
			for(unsigned tbin = 0; tbin < photonHist.size(); ++tbin){
				unsigned pe = photonHist[tbin];
				if(pe>0){
					sumPE += pe;
					sumHits += sipm.hitCells(engine, pe, 0., elapsedTime);
				}
				elapsedTime += dt;
			}
			
			//fill profiles
			profio->Fill(sumPE,sumHits);
			profoi->Fill(sumHits,sumPE);
		}
		npe += npeStep;
	}
	
	//calculate correction factor
	for(int b = 0; b < profoi->GetNbinsX(); ++b){
		if(profoi->GetBinContent(b)==0) continue;
		corr->SetBinContent(b,profoi->GetBinContent(b)/profoi->GetBinCenter(b));
	}
	//fit with requested function
	corr->Fit(fitname.c_str(),"Q");
	
	//print fit results
	TF1* fit = corr->GetFunction(fitname.c_str());
	std::cout << "chisquare / ndf = " << fit->GetChisquare() << " / " << fit->GetNDF() << std::endl;
	std::cout << "parameters = ";
	std::copy(fit->GetParameters(),fit->GetParameters()+fit->GetNpar(),std::ostream_iterator<double>(std::cout,", "));
	std::cout << std::endl;
}
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
SiPMNonlinearityAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(SiPMNonlinearityAnalyzer);

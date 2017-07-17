#include "Validation/RecoVertex/interface/BeamSpotHistogramMaker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TProfile.h"


BeamSpotHistogramMaker::BeamSpotHistogramMaker(edm::ConsumesCollector&& iC):
  _currdir(0), _histoParameters(), _rhm(iC) { }

BeamSpotHistogramMaker::BeamSpotHistogramMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC):
  _currdir(0),
  _histoParameters(iConfig.getUntrackedParameter<edm::ParameterSet>("histoParameters",edm::ParameterSet())),
  _rhm(iC)
{ }


BeamSpotHistogramMaker::~BeamSpotHistogramMaker() {

  delete _currdir;

}


void BeamSpotHistogramMaker::book(const std::string dirname) {

  edm::Service<TFileService> tfserv;
  TFileDirectory* currdir = &(tfserv->tFileDirectory());

  if(dirname!="") {
    currdir = new TFileDirectory(tfserv->mkdir(dirname));
    _currdir = currdir;
  }

  edm::LogInfo("HistogramBooking") << "BeamSpot histogram booking in directory " << dirname;

  _hbsxrun = _rhm.makeTH1F("bsxrun","BeamSpot X position",
			    _histoParameters.getUntrackedParameter<unsigned int>("nBinX",200),
			    _histoParameters.getUntrackedParameter<double>("xMin",-1.),
			    _histoParameters.getUntrackedParameter<double>("xMax",1.));

  _hbsyrun = _rhm.makeTH1F("bsyrun","BeamSpot Y position",
			    _histoParameters.getUntrackedParameter<unsigned int>("nBinY",200),
			    _histoParameters.getUntrackedParameter<double>("yMin",-1.),
			    _histoParameters.getUntrackedParameter<double>("yMax",1.));

  _hbszrun = _rhm.makeTH1F("bszrun","BeamSpot Z position",
			    _histoParameters.getUntrackedParameter<unsigned int>("nBinZ",200),
			    _histoParameters.getUntrackedParameter<double>("zMin",-1.),
			    _histoParameters.getUntrackedParameter<double>("zMax",1.));

  _hbssigmaxrun = _rhm.makeTH1F("bssigmaxrun","BeamSpot sigmaX",
				_histoParameters.getUntrackedParameter<unsigned int>("nBinSigmaX",200),
				_histoParameters.getUntrackedParameter<double>("sigmaXMin",0.),
				_histoParameters.getUntrackedParameter<double>("sigmaXMax",0.025));

  _hbssigmayrun = _rhm.makeTH1F("bssigmayrun","BeamSpot sigmaY",
				_histoParameters.getUntrackedParameter<unsigned int>("nBinSigmaY",200),
				_histoParameters.getUntrackedParameter<double>("sigmaYMin",0.),
				_histoParameters.getUntrackedParameter<double>("sigmaYMax",0.025));

  _hbssigmazrun = _rhm.makeTH1F("bssigmazrun","BeamSpot sigmaZ",
				_histoParameters.getUntrackedParameter<unsigned int>("nBinSigmaZ",200),
				_histoParameters.getUntrackedParameter<double>("sigmaZMin",0.),
				_histoParameters.getUntrackedParameter<double>("sigmaZMax",15.));

  _hbsxvsorbrun = _rhm.makeTProfile("bsxvsorbrun","BeamSpot X position vs orbit number",1600,0.5,1600.*16384+0.5);
  _hbsyvsorbrun = _rhm.makeTProfile("bsyvsorbrun","BeamSpot Y position vs orbit number",1600,0.5,1600.*16384+0.5);
  _hbszvsorbrun = _rhm.makeTProfile("bszvsorbrun","BeamSpot Z position vs orbit number",1600,0.5,1600.*16384+0.5);
  _hbssigmaxvsorbrun = _rhm.makeTProfile("bssigmaxvsorbrun","BeamSpot sigmaX vs orbit number",1600,0.5,1600.*16384+0.5);
  _hbssigmayvsorbrun = _rhm.makeTProfile("bssigmayvsorbrun","BeamSpot sigmaY vs orbit number",1600,0.5,1600.*16384+0.5);
  _hbssigmazvsorbrun = _rhm.makeTProfile("bssigmazvsorbrun","BeamSpot sigmaZ vs orbit number",1600,0.5,1600.*16384+0.5);

}

void BeamSpotHistogramMaker::beginRun(const unsigned int nrun) {

  char runname[100];
  sprintf(runname,"run_%d",nrun);

  TFileDirectory* currdir = _currdir;
  if(currdir==0) {
    edm::Service<TFileService> tfserv;
    currdir = &(tfserv->tFileDirectory());
  }

  _rhm.beginRun(nrun,*currdir);

  (*_hbsxrun)->GetXaxis()->SetTitle("X [cm]");   (*_hbsxrun)->GetYaxis()->SetTitle("Events");
  (*_hbsyrun)->GetXaxis()->SetTitle("Y [cm]");   (*_hbsyrun)->GetYaxis()->SetTitle("Events");
  (*_hbszrun)->GetXaxis()->SetTitle("Z [cm]");   (*_hbszrun)->GetYaxis()->SetTitle("Events");
  (*_hbssigmaxrun)->GetXaxis()->SetTitle("sigmaX [cm]");   (*_hbssigmaxrun)->GetYaxis()->SetTitle("Events");
  (*_hbssigmayrun)->GetXaxis()->SetTitle("sigmaY [cm]");   (*_hbssigmayrun)->GetYaxis()->SetTitle("Events");
  (*_hbssigmazrun)->GetXaxis()->SetTitle("sigmaZ [cm]");   (*_hbssigmazrun)->GetYaxis()->SetTitle("Events");

  (*_hbsxvsorbrun)->GetXaxis()->SetTitle("time [orbit#]");   (*_hbsxvsorbrun)->GetYaxis()->SetTitle("X [cm]");
  (*_hbsxvsorbrun)->SetCanExtend(TH1::kAllAxes);
  (*_hbsyvsorbrun)->GetXaxis()->SetTitle("time [orbit#]");   (*_hbsyvsorbrun)->GetYaxis()->SetTitle("Y [cm]");
  (*_hbsyvsorbrun)->SetCanExtend(TH1::kAllAxes);
  (*_hbszvsorbrun)->GetXaxis()->SetTitle("time [orbit#]");   (*_hbszvsorbrun)->GetYaxis()->SetTitle("Z [cm]");
  (*_hbszvsorbrun)->SetCanExtend(TH1::kAllAxes);
  (*_hbssigmaxvsorbrun)->GetXaxis()->SetTitle("time [orbit#]");   (*_hbssigmaxvsorbrun)->GetYaxis()->SetTitle("sigmaX [cm]");
  (*_hbssigmaxvsorbrun)->SetCanExtend(TH1::kAllAxes);
  (*_hbssigmayvsorbrun)->GetXaxis()->SetTitle("time [orbit#]");   (*_hbssigmayvsorbrun)->GetYaxis()->SetTitle("sigmaY [cm]");
  (*_hbssigmayvsorbrun)->SetCanExtend(TH1::kAllAxes);
  (*_hbssigmazvsorbrun)->GetXaxis()->SetTitle("time [orbit#]");   (*_hbssigmazvsorbrun)->GetYaxis()->SetTitle("sigmaZ [cm]");
  (*_hbssigmazvsorbrun)->SetCanExtend(TH1::kAllAxes);


}

void BeamSpotHistogramMaker::fill(const unsigned int orbit, const reco::BeamSpot& bs) {

  if(_hbsxrun && *_hbsxrun )  (*_hbsxrun)->Fill(bs.x0());
  if(_hbsxvsorbrun && *_hbsxvsorbrun )  (*_hbsxvsorbrun)->Fill(orbit,bs.x0());

  if(_hbsyrun && *_hbsyrun )  (*_hbsyrun)->Fill(bs.y0());
  if(_hbsyvsorbrun && *_hbsyvsorbrun )  (*_hbsyvsorbrun)->Fill(orbit,bs.y0());

  if(_hbszrun && *_hbszrun )  (*_hbszrun)->Fill(bs.z0());
  if(_hbszvsorbrun && *_hbszvsorbrun )  (*_hbszvsorbrun)->Fill(orbit,bs.z0());

  if(_hbssigmaxrun && *_hbssigmaxrun )  (*_hbssigmaxrun)->Fill(bs.BeamWidthX());
  if(_hbssigmayrun && *_hbssigmayrun )  (*_hbssigmayrun)->Fill(bs.BeamWidthY());
  if(_hbssigmazrun && *_hbssigmazrun )  (*_hbssigmazrun)->Fill(bs.sigmaZ());
  if(_hbssigmaxvsorbrun && *_hbssigmaxvsorbrun )  (*_hbssigmaxvsorbrun)->Fill(orbit,bs.BeamWidthX());
  if(_hbssigmayvsorbrun && *_hbssigmayvsorbrun )  (*_hbssigmayvsorbrun)->Fill(orbit,bs.BeamWidthY());
  if(_hbssigmazvsorbrun && *_hbssigmazvsorbrun )  (*_hbssigmazvsorbrun)->Fill(orbit,bs.sigmaZ());

}


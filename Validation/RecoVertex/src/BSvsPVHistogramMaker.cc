#include "Validation/RecoVertex/interface/BSvsPVHistogramMaker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"


BSvsPVHistogramMaker::BSvsPVHistogramMaker(edm::ConsumesCollector&& iC):
  _currdir(0), m_maxLS(100), useSlope_(true), _trueOnly(true),
  _runHisto(true), _runHistoProfile(true), _runHistoBXProfile(true), _runHistoBX2D(false), _histoParameters(), _rhm(iC) { }

BSvsPVHistogramMaker::BSvsPVHistogramMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC):
  _currdir(0),
  m_maxLS(iConfig.getParameter<unsigned int>("maxLSBeforeRebin")),
  useSlope_(iConfig.getParameter<bool>("useSlope")),
  _trueOnly(iConfig.getUntrackedParameter<bool>("trueOnly",true)),
  _runHisto(iConfig.getUntrackedParameter<bool>("runHisto",true)),
  _runHistoProfile(iConfig.getUntrackedParameter<bool>("runHistoProfile",true)),
  _runHistoBXProfile(iConfig.getUntrackedParameter<bool>("runHistoBXProfile",true)),
  _runHistoBX2D(iConfig.getUntrackedParameter<bool>("runHistoBX2D",false)),
  _histoParameters(iConfig.getUntrackedParameter<edm::ParameterSet>("histoParameters",edm::ParameterSet())),
  _rhm(iC)
{ }


BSvsPVHistogramMaker::~BSvsPVHistogramMaker() {

  delete _currdir;

}


void BSvsPVHistogramMaker::book(const std::string dirname) {

  edm::Service<TFileService> tfserv;
  TFileDirectory* currdir = &(tfserv->tFileDirectory());

  if(dirname!="") {
    currdir = new TFileDirectory(tfserv->mkdir(dirname));
    _currdir = currdir;
  }

  edm::LogInfo("HistogramBooking") << "Vertex histogram booking in directory " << dirname;

  _hdeltax = currdir->make<TH1F>("deltax","(PV-BS) X position",
			       _histoParameters.getUntrackedParameter<unsigned int>("nBinX",200),
			       _histoParameters.getUntrackedParameter<double>("xMin",-1.),
			       _histoParameters.getUntrackedParameter<double>("xMax",1.)
			       );
  _hdeltax->GetXaxis()->SetTitle("#Delta(X) [cm]");   _hdeltax->GetYaxis()->SetTitle("Vertices");

  _hdeltay = currdir->make<TH1F>("deltay","(PV-BS) Y position",
			       _histoParameters.getUntrackedParameter<unsigned int>("nBinY",200),
			       _histoParameters.getUntrackedParameter<double>("yMin",-1.),
			       _histoParameters.getUntrackedParameter<double>("yMax",1.)
			       );
  _hdeltay->GetXaxis()->SetTitle("#Delta(Y) [cm]");   _hdeltay->GetYaxis()->SetTitle("Vertices");

  _hdeltaz = currdir->make<TH1F>("deltaz","(PV-BS) Z position",
			       _histoParameters.getUntrackedParameter<unsigned int>("nBinZ",200),
			       _histoParameters.getUntrackedParameter<double>("zMin",-20.),
			       _histoParameters.getUntrackedParameter<double>("zMax",20.)
			       );
  _hdeltaz->GetXaxis()->SetTitle("#Delta(Z) [cm]");   _hdeltaz->GetYaxis()->SetTitle("Vertices");

  _hdeltaxvsz = currdir->make<TProfile>("deltaxvsz","(PV-BS) X position vs Z",
			       _histoParameters.getUntrackedParameter<unsigned int>("nBinZProfile",40),
			       _histoParameters.getUntrackedParameter<double>("zMinProfile",-20.),
			       _histoParameters.getUntrackedParameter<double>("zMaxProfile",20.)
			       );
  _hdeltaxvsz->GetXaxis()->SetTitle("Z [cm]");   _hdeltaxvsz->GetYaxis()->SetTitle("#Delta(X) [cm]");

  _hdeltayvsz = currdir->make<TProfile>("deltayvsz","(PV-BS) Y position vs Z",
			       _histoParameters.getUntrackedParameter<unsigned int>("nBinZProfile",40),
			       _histoParameters.getUntrackedParameter<double>("zMinProfile",-20.),
			       _histoParameters.getUntrackedParameter<double>("zMaxProfile",20.)
			       );
  _hdeltayvsz->GetXaxis()->SetTitle("Z [cm]");   _hdeltayvsz->GetYaxis()->SetTitle("#Delta(Y) [cm]");




  if(_runHisto) {
    _hdeltaxrun = _rhm.makeTH1F("deltaxrun","(PV-BS) X position",
			      _histoParameters.getUntrackedParameter<unsigned int>("nBinX",200),
			      _histoParameters.getUntrackedParameter<double>("xMin",-1.),
			      _histoParameters.getUntrackedParameter<double>("xMax",1.));

    _hdeltayrun = _rhm.makeTH1F("deltayrun","(PV-BS) Y position",
			      _histoParameters.getUntrackedParameter<unsigned int>("nBinY",200),
			      _histoParameters.getUntrackedParameter<double>("yMin",-1.),
			      _histoParameters.getUntrackedParameter<double>("yMax",1.));

    _hdeltazrun = _rhm.makeTH1F("deltazrun","(PV-BS) Z position",
			      _histoParameters.getUntrackedParameter<unsigned int>("nBinZ",200),
			      _histoParameters.getUntrackedParameter<double>("zMin",-20.),
			      _histoParameters.getUntrackedParameter<double>("zMax",20.));

    _hdeltaxvszrun = _rhm.makeTProfile("deltaxvszrun","(PV-BS) X position vs Z",
				       _histoParameters.getUntrackedParameter<unsigned int>("nBinZProfile",40),
				       _histoParameters.getUntrackedParameter<double>("zMinProfile",-20.),
				       _histoParameters.getUntrackedParameter<double>("zMaxProfile",20.)
				       );

    _hdeltayvszrun = _rhm.makeTProfile("deltayvszrun","(PV-BS) Y position vs Z",
				       _histoParameters.getUntrackedParameter<unsigned int>("nBinZProfile",40),
				       _histoParameters.getUntrackedParameter<double>("zMinProfile",-20.),
				       _histoParameters.getUntrackedParameter<double>("zMaxProfile",20.)
				       );

    if(_runHistoProfile) {
      _hdeltaxvsorbrun = _rhm.makeTProfile("deltaxvsorbrun","(PV-BS) X position vs orbit number",4*m_maxLS,0.5,m_maxLS*262144+0.5);
      _hdeltayvsorbrun = _rhm.makeTProfile("deltayvsorbrun","(PV-BS) Y position vs orbit number",4*m_maxLS,0.5,m_maxLS*262144+0.5);
      _hdeltazvsorbrun = _rhm.makeTProfile("deltazvsorbrun","(PV-BS) Z position vs orbit number",4*m_maxLS,0.5,m_maxLS*262144+0.5);
    }
    if(_runHistoBXProfile) {
      _hdeltaxvsbxrun = _rhm.makeTProfile("deltaxvsbxrun","(PV-BS) X position vs BX number",3564,-0.5,3563.5);
      _hdeltayvsbxrun = _rhm.makeTProfile("deltayvsbxrun","(PV-BS) Y position vs BX number",3564,-0.5,3563.5);
      _hdeltazvsbxrun = _rhm.makeTProfile("deltazvsbxrun","(PV-BS) Z position vs BX number",3564,-0.5,3563.5);
      if(_runHistoBX2D) {
	_hdeltaxvsbx2drun = _rhm.makeTH2F("deltaxvsbx2drun","(PV-BS) X position vs BX number",3564,-0.5,3563.5,
					  _histoParameters.getUntrackedParameter<unsigned int>("nBinX",200),
					  _histoParameters.getUntrackedParameter<double>("xMin",-1.),
					  _histoParameters.getUntrackedParameter<double>("xMax",1.));
	_hdeltayvsbx2drun = _rhm.makeTH2F("deltayvsbx2drun","(PV-BS) Y position vs BX number",3564,-0.5,3563.5,
					  _histoParameters.getUntrackedParameter<unsigned int>("nBinY",200),
					  _histoParameters.getUntrackedParameter<double>("yMin",-1.),
					  _histoParameters.getUntrackedParameter<double>("yMax",1.));
	_hdeltazvsbx2drun = _rhm.makeTH2F("deltazvsbx2drun","(PV-BS) Z position vs BX number",3564,-0.5,3563.5,
					  _histoParameters.getUntrackedParameter<unsigned int>("nBinZ",200),
					  _histoParameters.getUntrackedParameter<double>("zMin",-20.),
					  _histoParameters.getUntrackedParameter<double>("zMax",20.));
      }
    }

  }
}

void BSvsPVHistogramMaker::beginRun(const unsigned int nrun) {

  char runname[100];
  sprintf(runname,"run_%d",nrun);

  TFileDirectory* currdir = _currdir;
  if(currdir==0) {
    edm::Service<TFileService> tfserv;
    currdir = &(tfserv->tFileDirectory());
  }

  _rhm.beginRun(nrun,*currdir);

  if(_runHisto) {
    (*_hdeltaxrun)->GetXaxis()->SetTitle("#Delta(X) [cm]");   (*_hdeltaxrun)->GetYaxis()->SetTitle("Vertices");
    (*_hdeltayrun)->GetXaxis()->SetTitle("#Delta(Y) [cm]");   (*_hdeltayrun)->GetYaxis()->SetTitle("Vertices");
    (*_hdeltazrun)->GetXaxis()->SetTitle("#Delta(Z) [cm]");   (*_hdeltazrun)->GetYaxis()->SetTitle("Vertices");
    (*_hdeltaxvszrun)->GetXaxis()->SetTitle("Z [cm]");   (*_hdeltaxvszrun)->GetYaxis()->SetTitle("#Delta(X) [cm]");
    (*_hdeltayvszrun)->GetXaxis()->SetTitle("Z [cm]");   (*_hdeltayvszrun)->GetYaxis()->SetTitle("#Delta(Y) [cm]");

    if(_runHistoProfile) {
      (*_hdeltaxvsorbrun)->GetXaxis()->SetTitle("time [orbit#]");   (*_hdeltaxvsorbrun)->GetYaxis()->SetTitle("#Delta(X) [cm]");
      (*_hdeltaxvsorbrun)->SetBit(TH1::kCanRebin);
      (*_hdeltayvsorbrun)->GetXaxis()->SetTitle("time [orbit#]");   (*_hdeltayvsorbrun)->GetYaxis()->SetTitle("#Delta(Y) [cm]");
      (*_hdeltayvsorbrun)->SetBit(TH1::kCanRebin);
      (*_hdeltazvsorbrun)->GetXaxis()->SetTitle("time [orbit#]");   (*_hdeltazvsorbrun)->GetYaxis()->SetTitle("#Delta(Z) [cm]");
      (*_hdeltazvsorbrun)->SetBit(TH1::kCanRebin);
    }
    if(_runHistoBXProfile) {
      (*_hdeltaxvsbxrun)->GetXaxis()->SetTitle("BX");   (*_hdeltaxvsbxrun)->GetYaxis()->SetTitle("#Delta(X) [cm]");
      (*_hdeltayvsbxrun)->GetXaxis()->SetTitle("BX");   (*_hdeltayvsbxrun)->GetYaxis()->SetTitle("#Delta(Y) [cm]");
      (*_hdeltazvsbxrun)->GetXaxis()->SetTitle("BX");   (*_hdeltazvsbxrun)->GetYaxis()->SetTitle("#Delta(Z) [cm]");
      if(_runHistoBX2D) {
	(*_hdeltaxvsbx2drun)->GetXaxis()->SetTitle("BX");   (*_hdeltaxvsbx2drun)->GetYaxis()->SetTitle("#Delta(X) [cm]");
	(*_hdeltayvsbx2drun)->GetXaxis()->SetTitle("BX");   (*_hdeltayvsbx2drun)->GetYaxis()->SetTitle("#Delta(Y) [cm]");
	(*_hdeltazvsbx2drun)->GetXaxis()->SetTitle("BX");   (*_hdeltazvsbx2drun)->GetYaxis()->SetTitle("#Delta(Z) [cm]");
      }
    }

  }
}

void BSvsPVHistogramMaker::fill(const unsigned int orbit, const int bx, const reco::VertexCollection& vertices, const reco::BeamSpot& bs) {

  for(reco::VertexCollection::const_iterator vtx=vertices.begin();vtx!=vertices.end();++vtx) {

    if(!(_trueOnly && vtx->isFake())) {

      /*
      double deltax = vtx->x()-bs.x0();
      double deltay = vtx->y()-bs.y0();
      double deltaz = vtx->z()-bs.z0();
      */
      double deltax = vtx->x()-x(bs,vtx->z());
      double deltay = vtx->y()-y(bs,vtx->z());
      double deltaz = vtx->z()-bs.z0();

      _hdeltax->Fill(deltax);
      _hdeltay->Fill(deltay);
      _hdeltaz->Fill(deltaz);
      _hdeltaxvsz->Fill(vtx->z(),deltax);
      _hdeltayvsz->Fill(vtx->z(),deltay);

      if(_runHisto) {
	if(_hdeltaxrun && *_hdeltaxrun )  (*_hdeltaxrun)->Fill(deltax);
	if(_hdeltayrun && *_hdeltayrun )  (*_hdeltayrun)->Fill(deltay);
	if(_hdeltazrun && *_hdeltazrun )  (*_hdeltazrun)->Fill(deltaz);
	if(_hdeltaxvszrun && *_hdeltaxvszrun )  (*_hdeltaxvszrun)->Fill(vtx->z(),deltax);
	if(_hdeltayvszrun && *_hdeltayvszrun )  (*_hdeltayvszrun)->Fill(vtx->z(),deltay);
	if(_runHistoProfile) {
	  if(_hdeltaxvsorbrun && *_hdeltaxvsorbrun )  (*_hdeltaxvsorbrun)->Fill(orbit,deltax);
	  if(_hdeltayvsorbrun && *_hdeltayvsorbrun )  (*_hdeltayvsorbrun)->Fill(orbit,deltay);
	  if(_hdeltazvsorbrun && *_hdeltazvsorbrun )  (*_hdeltazvsorbrun)->Fill(orbit,deltaz);
	}
	if(_runHistoBXProfile) {
	  if(_hdeltaxvsbxrun && *_hdeltaxvsbxrun )  (*_hdeltaxvsbxrun)->Fill(bx,deltax);
	  if(_hdeltayvsbxrun && *_hdeltayvsbxrun )  (*_hdeltayvsbxrun)->Fill(bx,deltay);
	  if(_hdeltazvsbxrun && *_hdeltazvsbxrun )  (*_hdeltazvsbxrun)->Fill(bx,deltaz);
	  if(_runHistoBX2D) {
	    if(_hdeltaxvsbx2drun && *_hdeltaxvsbx2drun )  (*_hdeltaxvsbx2drun)->Fill(bx,deltax);
	    if(_hdeltayvsbx2drun && *_hdeltayvsbx2drun )  (*_hdeltayvsbx2drun)->Fill(bx,deltay);
	    if(_hdeltazvsbx2drun && *_hdeltazvsbx2drun )  (*_hdeltazvsbx2drun)->Fill(bx,deltaz);
	  }
	}
      }
    }
  }
}

void BSvsPVHistogramMaker::fill(const edm::Event& iEvent, const reco::VertexCollection& vertices, const reco::BeamSpot& bs) {

  fill(iEvent.orbitNumber(),iEvent.bunchCrossing(),vertices,bs);

}

double BSvsPVHistogramMaker::x(const reco::BeamSpot& bs, const double z) const {

  double x = bs.x0();

  //  if(useSlope_) x += bs.dxdz()*z;
  if(useSlope_) x += bs.dxdz()*(z-bs.z0());

  //  if(useSlope_) x = bs.x(z);

  return x;

}

double BSvsPVHistogramMaker::y(const reco::BeamSpot& bs, const double z) const {

  double y = bs.y0();

  //  if(useSlope_) y += bs.dydz()*z;
  if(useSlope_) y += bs.dydz()*(z-bs.z0());

    //  if(useSlope_) y = bs.y(z);

  return y;

}


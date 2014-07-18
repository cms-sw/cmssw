#include "Validation/RecoVertex/interface/VertexHistogramMaker.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TProfile.h"


VertexHistogramMaker::VertexHistogramMaker(edm::ConsumesCollector&& iC)
  : m_currdir(0), m_maxLS(100), m_weightThreshold(0.5), m_trueOnly(true)
  , m_runHisto(true), m_runHistoProfile(true), m_runHistoBXProfile(true), m_runHistoBXProfile2D(false), m_runHisto2D(false)
  , m_bsConstrained(false)
  , m_histoParameters()
  , m_rhm(iC)
  , m_fhm(iC) { }

VertexHistogramMaker::VertexHistogramMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC)
  : m_currdir(0)
  , m_maxLS(iConfig.getParameter<unsigned int>("maxLSBeforeRebin"))
  , m_weightThreshold(iConfig.getUntrackedParameter<double>("weightThreshold",0.5))
  , m_trueOnly(iConfig.getUntrackedParameter<bool>("trueOnly",true))
  , m_runHisto(iConfig.getUntrackedParameter<bool>("runHisto",true))
  , m_runHistoProfile(iConfig.getUntrackedParameter<bool>("runHistoProfile",true))
  , m_runHistoBXProfile(iConfig.getUntrackedParameter<bool>("runHistoBXProfile",true))
  , m_runHistoBXProfile2D(iConfig.getUntrackedParameter<bool>("runHistoBXProfile2D",false))
  , m_runHisto2D(iConfig.getUntrackedParameter<bool>("runHisto2D",false))
  , m_bsConstrained(iConfig.getParameter<bool>("bsConstrained"))
  , m_histoParameters(iConfig.getUntrackedParameter<edm::ParameterSet>("histoParameters",edm::ParameterSet()))
  , m_lumiDetailsToken( iC.consumes< LumiDetails, edm::InLumi >( edm::InputTag( std::string( "lumiProducer" ) ) ) )
  , m_rhm(iC, false),m_fhm(iC, true)
{ }


VertexHistogramMaker::~VertexHistogramMaker() {

  delete m_currdir;

}


void VertexHistogramMaker::book(const std::string dirname) {

  edm::Service<TFileService> tfserv;
  TFileDirectory* currdir = &(tfserv->tFileDirectory());

  if(dirname!="") {
    currdir = new TFileDirectory(tfserv->mkdir(dirname));
    m_currdir = currdir;
  }

  edm::LogInfo("HistogramBooking") << "Vertex histogram booking in directory " << dirname;

  m_hnvtx = currdir->make<TH1F>("nvtx","Number of Vertices",60,-0.5,59.5);
  m_hnvtx->GetXaxis()->SetTitle("vertices");   m_hnvtx->GetYaxis()->SetTitle("Events");

  m_hntruevtx = currdir->make<TH1F>("ntruevtx","Number of True Vertices",60,-0.5,59.5);
  m_hntruevtx->GetXaxis()->SetTitle("vertices");   m_hntruevtx->GetYaxis()->SetTitle("Events");

  m_hntruevtxvslumi = currdir->make<TProfile>("ntruevtxvslumi","Number of True Vertices vs BX lumi",250,0.,10.);
  m_hntruevtxvslumi->GetXaxis()->SetTitle("BX lumi [10^{30}cm^{-2}s^{-1}]");   m_hntruevtxvslumi->GetYaxis()->SetTitle("Vertices");

  m_hntruevtxvslumi2D = currdir->make<TH2D>("ntruevtxvslumi2D","Number of True Vertices vs BX lumi",250,0.,10.,100,-0.5,99.5);
  m_hntruevtxvslumi2D->GetXaxis()->SetTitle("BX lumi [10^{30}cm^{-2}s^{-1}]");   m_hntruevtxvslumi2D->GetYaxis()->SetTitle("Vertices");

  m_hntracks = currdir->make<TH1F>("ntracks","Number of Tracks",300,-0.5,299.5);
  m_hntracks->GetXaxis()->SetTitle("tracks");   m_hntracks->GetYaxis()->SetTitle("Vertices");

  m_hsqsumptsq = currdir->make<TH1F>("sqsumptsq","sqrt(sum pt**2)",1000,0.,1000.);
  m_hsqsumptsq->GetXaxis()->SetTitle("sqrt(#Sigma pt^{2}) (GeV)");   m_hsqsumptsq->GetYaxis()->SetTitle("Vertices");

  char htitle[300];
  sprintf(htitle,"sqrt(sum pt**2) of Tracks weight > %f",m_weightThreshold);
  m_hsqsumptsqheavy = currdir->make<TH1F>("sqsumptsqheavy",htitle,1000,0.,1000.);
  m_hsqsumptsqheavy->GetXaxis()->SetTitle("sqrt(#Sigma pt^{2}) (GeV)");   m_hsqsumptsqheavy->GetYaxis()->SetTitle("Vertices");

  sprintf(htitle,"Number of Tracks weight > %f",m_weightThreshold);
  m_hnheavytracks = currdir->make<TH1F>("nheavytracks",htitle,200,-0.5,199.5);
  m_hnheavytracks->GetXaxis()->SetTitle("tracks");   m_hnheavytracks->GetYaxis()->SetTitle("Vertices");

  m_hndof = currdir->make<TH1F>("ndof","Number of degree of freedom",250,-0.5,499.5);
  m_hndof->GetXaxis()->SetTitle("ndof");   m_hndof->GetYaxis()->SetTitle("Vertices");

  m_hndofvstracks = currdir->make<TH2F>("ndofvstracks","Ndof vs Ntracks",300,-0.5,299.5,250,-0.5,499.5);
  m_hndofvstracks->GetXaxis()->SetTitle("tracks");   m_hndofvstracks->GetYaxis()->SetTitle("ndof");

  m_hndofvsvtxz = currdir->make<TProfile>("ndofvsvtxz","Ndof vs Vertex Z position",200,
					 m_histoParameters.getUntrackedParameter<double>("zMin",-20.),
					 m_histoParameters.getUntrackedParameter<double>("zMax",20.));
  m_hndofvsvtxz->GetXaxis()->SetTitle("Z [cm]");   m_hndofvsvtxz->GetYaxis()->SetTitle("ndof");

  m_hntracksvsvtxz = currdir->make<TProfile>("ntracksvsvtxz","Ntracks vs Vertex Z position",200,
					 m_histoParameters.getUntrackedParameter<double>("zMin",-20.),
					 m_histoParameters.getUntrackedParameter<double>("zMax",20.));
  m_hntracksvsvtxz->GetXaxis()->SetTitle("Z [cm]");   m_hntracksvsvtxz->GetYaxis()->SetTitle("tracks");

  m_haveweightvsvtxz = currdir->make<TProfile>("aveweightvsvtxz","Average weight vs Vertex Z position",200,
					 m_histoParameters.getUntrackedParameter<double>("zMin",-20.),
					 m_histoParameters.getUntrackedParameter<double>("zMax",20.));
  m_haveweightvsvtxz->GetXaxis()->SetTitle("Z [cm]");   m_haveweightvsvtxz->GetYaxis()->SetTitle("Average weight");

  m_haveweightvsvtxzchk = currdir->make<TProfile>("aveweightvsvtxzchk","Average weight vs Vertex Z position (check)",200,
						  m_histoParameters.getUntrackedParameter<double>("zMin",-20.),
						  m_histoParameters.getUntrackedParameter<double>("zMax",20.));
  m_haveweightvsvtxzchk->GetXaxis()->SetTitle("Z [cm]");   m_haveweightvsvtxzchk->GetYaxis()->SetTitle("Average weight");

  m_hweights = currdir->make<TH1F>("weights","Tracks weights",51,0.,1.02);
  m_hweights->GetXaxis()->SetTitle("weights");   m_hweights->GetYaxis()->SetTitle("Tracks");

  m_haveweight = currdir->make<TH1F>("aveweight","Tracks average weights sum",51,0.,1.02);
  m_haveweight->GetXaxis()->SetTitle("Average weight");   m_haveweight->GetYaxis()->SetTitle("Vertices");


  m_hvtxx = currdir->make<TH1F>("vtxx","Vertex X position",
			       m_histoParameters.getUntrackedParameter<unsigned int>("nBinX",200),
			       m_histoParameters.getUntrackedParameter<double>("xMin",-1.),
			       m_histoParameters.getUntrackedParameter<double>("xMax",1.)
			       );
  m_hvtxx->GetXaxis()->SetTitle("X [cm]");   m_hvtxx->GetYaxis()->SetTitle("Vertices");

  m_hvtxy = currdir->make<TH1F>("vtxy","Vertex Y position",
			       m_histoParameters.getUntrackedParameter<unsigned int>("nBinY",200),
			       m_histoParameters.getUntrackedParameter<double>("yMin",-1.),
			       m_histoParameters.getUntrackedParameter<double>("yMax",1.)
			       );
  m_hvtxy->GetXaxis()->SetTitle("Y [cm]");   m_hvtxy->GetYaxis()->SetTitle("Vertices");

  m_hvtxz = currdir->make<TH1F>("vtxz","Vertex Z position",
			       m_histoParameters.getUntrackedParameter<unsigned int>("nBinZ",200),
			       m_histoParameters.getUntrackedParameter<double>("zMin",-20.),
			       m_histoParameters.getUntrackedParameter<double>("zMax",20.)
			       );
  m_hvtxz->GetXaxis()->SetTitle("Z [cm]");   m_hvtxz->GetYaxis()->SetTitle("Vertices");

  if(m_runHisto) {
    m_hvtxxrun = m_rhm.makeTH1F("vtxxrun","Vertex X position",
			      m_histoParameters.getUntrackedParameter<unsigned int>("nBinX",200),
			      m_histoParameters.getUntrackedParameter<double>("xMin",-1.),
			      m_histoParameters.getUntrackedParameter<double>("xMax",1.));

    m_hvtxyrun = m_rhm.makeTH1F("vtxyrun","Vertex Y position",
			      m_histoParameters.getUntrackedParameter<unsigned int>("nBinY",200),
			      m_histoParameters.getUntrackedParameter<double>("yMin",-1.),
			      m_histoParameters.getUntrackedParameter<double>("yMax",1.));

    m_hvtxzrun = m_rhm.makeTH1F("vtxzrun","Vertex Z position",
			      m_histoParameters.getUntrackedParameter<unsigned int>("nBinZ",200),
			      m_histoParameters.getUntrackedParameter<double>("zMin",-20.),
			      m_histoParameters.getUntrackedParameter<double>("zMax",20.));

    if(m_runHistoProfile) {
      m_hvtxxvsorbrun = m_rhm.makeTProfile("vtxxvsorbrun","Vertex X position vs orbit number",4*m_maxLS,0.5,m_maxLS*262144+0.5);
      m_hvtxyvsorbrun = m_rhm.makeTProfile("vtxyvsorbrun","Vertex Y position vs orbit number",4*m_maxLS,0.5,m_maxLS*262144+0.5);
      m_hvtxzvsorbrun = m_rhm.makeTProfile("vtxzvsorbrun","Vertex Z position vs orbit number",4*m_maxLS,0.5,m_maxLS*262144+0.5);
      m_hnvtxvsorbrun = m_rhm.makeTProfile("nvtxvsorbrun","Number of true vertices vs orbit number",m_maxLS,0.5,m_maxLS*262144+0.5);
    }

    if(m_runHisto2D) {
      m_hnvtxvsbxvsorbrun = m_rhm.makeTProfile2D("nvtxvsbxvsorbrun","Number of true vertices vs BX vs orbit number",
						 3564,-0.5,3563.5,m_maxLS,0.5,m_maxLS*262144+0.5);
      m_hnvtxvsorbrun2D = m_rhm.makeTH2F("nvtxvsorbrun2D","Number of true vertices vs orbit number",
					 m_maxLS,0.5,m_maxLS*262144+0.5,60,-0.5,59.5);
    }

    if(m_runHistoBXProfile) {
      m_hvtxxvsbxrun = m_fhm.makeTProfile("vtxxvsbxrun","Vertex X position vs BX number",3564,-0.5,3563.5);
      m_hvtxyvsbxrun = m_fhm.makeTProfile("vtxyvsbxrun","Vertex Y position vs BX number",3564,-0.5,3563.5);
      m_hvtxzvsbxrun = m_fhm.makeTProfile("vtxzvsbxrun","Vertex Z position vs BX number",3564,-0.5,3563.5);

      m_hnvtxvsbxrun = m_rhm.makeTProfile("nvtxvsbxrun","Number of true vertices vs BX number",3564,-0.5,3563.5);

      if(m_runHistoBXProfile2D) {
	m_hnvtxvsbxvslumirun = m_fhm.makeTProfile2D("nvtxvsbxvslumirun","Number of vertices vs BX and BX lumi",3564,-0.5,3563.5,250,0.,10.);
      }
      if(m_runHisto2D) {
	m_hvtxxvsbx2drun = m_fhm.makeTH2F("vtxxvsbx2drun","Vertex X position vs BX number",3564,-0.5,3563.5,
					m_histoParameters.getUntrackedParameter<unsigned int>("nBinX",200),
					m_histoParameters.getUntrackedParameter<double>("xMin",-1.),
					m_histoParameters.getUntrackedParameter<double>("xMax",1.));
	m_hvtxyvsbx2drun = m_fhm.makeTH2F("vtxyvsbx2drun","Vertex Y position vs BX number",3564,-0.5,3563.5,
					m_histoParameters.getUntrackedParameter<unsigned int>("nBinY",200),
					m_histoParameters.getUntrackedParameter<double>("yMin",-1.),
					m_histoParameters.getUntrackedParameter<double>("yMax",1.));
	m_hvtxzvsbx2drun = m_fhm.makeTH2F("vtxzvsbx2drun","Vertex Z position vs BX number",3564,-0.5,3563.5,
					m_histoParameters.getUntrackedParameter<unsigned int>("nBinZ",200),
					m_histoParameters.getUntrackedParameter<double>("zMin",-20.),
					m_histoParameters.getUntrackedParameter<double>("zMax",20.));
      }
    }


  }
}

void VertexHistogramMaker::beginRun(const edm::Run& iRun) {

  TFileDirectory* currdir = m_currdir;
  if(currdir==0) {
    edm::Service<TFileService> tfserv;
    currdir = &(tfserv->tFileDirectory());
  }

  m_rhm.beginRun(iRun,*currdir);
  m_fhm.beginRun(iRun,*currdir);


  if(m_runHisto) {
    (*m_hvtxxrun)->GetXaxis()->SetTitle("X [cm]");   (*m_hvtxxrun)->GetYaxis()->SetTitle("Vertices");
    (*m_hvtxyrun)->GetXaxis()->SetTitle("Y [cm]");   (*m_hvtxyrun)->GetYaxis()->SetTitle("Vertices");
    (*m_hvtxzrun)->GetXaxis()->SetTitle("Z [cm]");   (*m_hvtxzrun)->GetYaxis()->SetTitle("Vertices");

    if(m_runHistoProfile) {
      (*m_hvtxxvsorbrun)->GetXaxis()->SetTitle("time [orbit#]");   (*m_hvtxxvsorbrun)->GetYaxis()->SetTitle("X [cm]");
      (*m_hvtxxvsorbrun)->SetBit(TH1::kCanRebin);
      (*m_hvtxyvsorbrun)->GetXaxis()->SetTitle("time [orbit#]");   (*m_hvtxyvsorbrun)->GetYaxis()->SetTitle("Y [cm]");
      (*m_hvtxyvsorbrun)->SetBit(TH1::kCanRebin);
      (*m_hvtxzvsorbrun)->GetXaxis()->SetTitle("time [orbit#]");   (*m_hvtxzvsorbrun)->GetYaxis()->SetTitle("Z [cm]");
      (*m_hvtxzvsorbrun)->SetBit(TH1::kCanRebin);
      (*m_hnvtxvsorbrun)->GetXaxis()->SetTitle("time [orbit#]");   (*m_hnvtxvsorbrun)->GetYaxis()->SetTitle("Nvertices");
      (*m_hnvtxvsorbrun)->SetBit(TH1::kCanRebin);
    }

    if(m_runHistoBXProfile) {
      (*m_hvtxxvsbxrun)->GetXaxis()->SetTitle("BX");   (*m_hvtxxvsbxrun)->GetYaxis()->SetTitle("X [cm]");
      (*m_hvtxyvsbxrun)->GetXaxis()->SetTitle("BX");   (*m_hvtxyvsbxrun)->GetYaxis()->SetTitle("Y [cm]");
      (*m_hvtxzvsbxrun)->GetXaxis()->SetTitle("BX");   (*m_hvtxzvsbxrun)->GetYaxis()->SetTitle("Z [cm]");
      (*m_hnvtxvsbxrun)->GetXaxis()->SetTitle("BX");   (*m_hnvtxvsbxrun)->GetYaxis()->SetTitle("Nvertices");
      if(m_runHistoBXProfile2D) {
	(*m_hnvtxvsbxvslumirun)->GetXaxis()->SetTitle("BX");   (*m_hnvtxvsbxvslumirun)->GetYaxis()->SetTitle("BX lumi [10^{30}cm^{-2}s^{-1}]");
      }
      if(m_runHisto2D) {
	(*m_hvtxxvsbx2drun)->GetXaxis()->SetTitle("BX");   (*m_hvtxxvsbx2drun)->GetYaxis()->SetTitle("X [cm]");
	(*m_hvtxyvsbx2drun)->GetXaxis()->SetTitle("BX");   (*m_hvtxyvsbx2drun)->GetYaxis()->SetTitle("Y [cm]");
	(*m_hvtxzvsbx2drun)->GetXaxis()->SetTitle("BX");   (*m_hvtxzvsbx2drun)->GetYaxis()->SetTitle("Z [cm]");
      }
    }

    if(m_runHisto2D) {
      (*m_hnvtxvsbxvsorbrun)->GetXaxis()->SetTitle("BX#"); (*m_hnvtxvsbxvsorbrun)->GetYaxis()->SetTitle("time [orbit#]");
      (*m_hnvtxvsbxvsorbrun)->SetBit(TH1::kCanRebin);
      (*m_hnvtxvsorbrun2D)->GetXaxis()->SetTitle("time [orbit#]");   (*m_hnvtxvsorbrun2D)->GetYaxis()->SetTitle("Nvertices");
      (*m_hnvtxvsorbrun2D)->SetBit(TH1::kCanRebin);
    }
  }
}

void VertexHistogramMaker::fill(const unsigned int orbit, const int bx, const float bxlumi, const reco::VertexCollection& vertices, const double weight) {

  m_hnvtx->Fill(vertices.size(),weight);

  int ntruevtx = 0;
  for(reco::VertexCollection::const_iterator vtx=vertices.begin();vtx!=vertices.end();++vtx) {
    if(!vtx->isFake()) ntruevtx++;

    if(!(m_trueOnly && vtx->isFake())) {

      double aveweight = m_bsConstrained ? vtx->ndof()/(2.*vtx->tracksSize()) : (vtx->ndof()+3)/(2.*vtx->tracksSize());

      m_hntracks->Fill(vtx->tracksSize(),weight);
      m_hndof->Fill(vtx->ndof(),weight);
      m_haveweight->Fill(aveweight,weight);
      m_hndofvstracks->Fill(vtx->tracksSize(),vtx->ndof(),weight);
      m_hndofvsvtxz->Fill(vtx->z(),vtx->ndof(),weight);
      m_hntracksvsvtxz->Fill(vtx->z(),vtx->tracksSize(),weight);
      m_haveweightvsvtxz->Fill(vtx->z(),aveweight,weight);

      m_hvtxx->Fill(vtx->x(),weight);
      m_hvtxy->Fill(vtx->y(),weight);
      m_hvtxz->Fill(vtx->z(),weight);

      if(m_runHisto) {
	if(m_hvtxxrun && *m_hvtxxrun )  (*m_hvtxxrun)->Fill(vtx->x(),weight);
	if(m_hvtxyrun && *m_hvtxyrun )  (*m_hvtxyrun)->Fill(vtx->y(),weight);
	if(m_hvtxzrun && *m_hvtxzrun )  (*m_hvtxzrun)->Fill(vtx->z(),weight);
	if(m_runHistoProfile) {
	  if(m_hvtxxvsorbrun && *m_hvtxxvsorbrun )  (*m_hvtxxvsorbrun)->Fill(orbit,vtx->x(),weight);
	  if(m_hvtxyvsorbrun && *m_hvtxyvsorbrun )  (*m_hvtxyvsorbrun)->Fill(orbit,vtx->y(),weight);
	  if(m_hvtxzvsorbrun && *m_hvtxzvsorbrun )  (*m_hvtxzvsorbrun)->Fill(orbit,vtx->z(),weight);
	}
	if(m_runHistoBXProfile) {
	  if(m_hvtxxvsbxrun && *m_hvtxxvsbxrun )  (*m_hvtxxvsbxrun)->Fill(bx,vtx->x(),weight);
	  if(m_hvtxyvsbxrun && *m_hvtxyvsbxrun )  (*m_hvtxyvsbxrun)->Fill(bx,vtx->y(),weight);
	  if(m_hvtxzvsbxrun && *m_hvtxzvsbxrun )  (*m_hvtxzvsbxrun)->Fill(bx,vtx->z(),weight);
	  if(m_runHisto2D) {
	    if(m_hvtxxvsbx2drun && *m_hvtxxvsbx2drun )  (*m_hvtxxvsbx2drun)->Fill(bx,vtx->x(),weight);
	    if(m_hvtxyvsbx2drun && *m_hvtxyvsbx2drun )  (*m_hvtxyvsbx2drun)->Fill(bx,vtx->y(),weight);
	    if(m_hvtxzvsbx2drun && *m_hvtxzvsbx2drun )  (*m_hvtxzvsbx2drun)->Fill(bx,vtx->z(),weight);
	  }
	}
      }

      int nheavytracks = 0;
      double sumpt2 = 0.;
      double sumpt2heavy = 0.;

      for(reco::Vertex::trackRef_iterator trk=vtx->tracks_begin();trk!=vtx->tracks_end();++trk) {

	sumpt2 += (*trk)->pt()*(*trk)->pt();

	if(vtx->trackWeight(*trk) > m_weightThreshold) {
	  nheavytracks++;
	  sumpt2heavy += (*trk)->pt()*(*trk)->pt();
	}

	m_hweights->Fill(vtx->trackWeight(*trk),weight);
	m_haveweightvsvtxzchk->Fill(vtx->z(),vtx->trackWeight(*trk),weight);

      }

      m_hnheavytracks->Fill(nheavytracks,weight);
      m_hsqsumptsq->Fill(sqrt(sumpt2),weight);
      m_hsqsumptsqheavy->Fill(sqrt(sumpt2heavy),weight);


    }


  }

  m_hntruevtx->Fill(ntruevtx,weight);

  if(bxlumi >= 0.) {
    m_hntruevtxvslumi->Fill(bxlumi,ntruevtx,weight);
    m_hntruevtxvslumi2D->Fill(bxlumi,ntruevtx,weight);
  }

  if(m_runHisto) {
    if(m_runHistoProfile) {
      if(m_hnvtxvsorbrun && *m_hnvtxvsorbrun )  (*m_hnvtxvsorbrun)->Fill(orbit,ntruevtx,weight);
    }
    if(m_runHistoBXProfile) {
      if(m_hnvtxvsbxrun && *m_hnvtxvsbxrun )  (*m_hnvtxvsbxrun)->Fill(bx,ntruevtx,weight);
      if(m_runHistoBXProfile2D) {
	if(m_hnvtxvsbxvslumirun && *m_hnvtxvsbxvslumirun && bxlumi >= 0.)  (*m_hnvtxvsbxvslumirun)->Fill(bx,bxlumi,ntruevtx,weight);
      }
    }
    if(m_runHisto2D) {
      if(m_hnvtxvsbxvsorbrun && *m_hnvtxvsbxvsorbrun )  (*m_hnvtxvsbxvsorbrun)->Fill(bx,orbit,ntruevtx,weight);
      if(m_hnvtxvsorbrun2D && *m_hnvtxvsorbrun2D )  {
	if(ntruevtx < (*m_hnvtxvsorbrun2D)->GetYaxis()->GetXmax() && ntruevtx > (*m_hnvtxvsorbrun2D)->GetYaxis()->GetXmin()) {
	  (*m_hnvtxvsorbrun2D)->Fill(orbit,ntruevtx,weight);
	}
      }
    }
  }


}

void VertexHistogramMaker::fill(const edm::Event& iEvent, const reco::VertexCollection& vertices, const double weight) {

  // get luminosity

  edm::Handle<LumiDetails> ld;
  iEvent.getLuminosityBlock().getByToken( m_lumiDetailsToken, ld );

  float bxlumi = -1.;

  if(ld.isValid()) {
    if(ld->isValid()) {
      bxlumi = ld->lumiValue(LumiDetails::kOCC1,iEvent.bunchCrossing())*6.37;
    }
  }

  fill(iEvent.orbitNumber(),iEvent.bunchCrossing(),bxlumi,vertices,weight);

}

// -*- C++ -*-
//
// Package:    SimRomanPot/CTPPSOpticsParameterisation
// Class:      PlotsProducer
// 
/**\class PlotsProducer PlotsProducer.cc SimRomanPot/CTPPSOpticsParameterisation/plugins/PlotsProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Laurent Forthomme
//         Created:  Fri, 26 May 2017 12:42:12 GMT
//
//
//
//
#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Common/interface/View.h"

#include "SimDataFormats/CTPPS/interface/CTPPSSimProtonTrack.h"
//#include "SimDataFormats/CTPPS/interface/CTPPSSimHit.h"

#include "SimRomanPot/CTPPSOpticsParameterisation/interface/ProtonReconstructionAlgorithm.h"

#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TGraph.h"
#include "TGraphErrors.h"

#include <map>

class PlotsProducer : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
  public:
    explicit PlotsProducer( const edm::ParameterSet& );
    ~PlotsProducer();

    static void fillDescriptions( edm::ConfigurationDescriptions& descriptions );

  private:
    virtual void beginJob() override;
    virtual void analyze( const edm::Event&, const edm::EventSetup& ) override;
    virtual void endJob() override;

    edm::EDGetTokenT< edm::View<CTPPSSimProtonTrack> > genProtonsToken_, recoProtons45Token_, recoProtons56Token_;
    edm::EDGetTokenT< edm::View<CTPPSSimHit> > hitsToken_;

    //edm::ParameterSet beamConditions_;
    std::vector<edm::ParameterSet> detectorPackages_;

    std::map<unsigned int,TH2D*> m_rp_h2_y_vs_x_;

    TH1D* h_de_vtx_x_[2];
    TH1D* h_de_vtx_y_[2];
    TH1D* h_de_th_x_[2];
    TH1D* h_de_th_y_[2];
    TH1D* h_de_xi_[2];

    TH2D* h2_de_vtx_x_vs_de_xi_[2];
    TH2D* h2_de_vtx_y_vs_de_xi_[2];
    TH2D* h2_de_th_x_vs_de_xi_[2];
    TH2D* h2_de_th_y_vs_de_xi_[2];
    TH2D* h2_de_vtx_y_vs_de_th_y_[2];

    TProfile* p_de_vtx_x_vs_xi_[2];
    TProfile* p_de_vtx_y_vs_xi_[2];
    TProfile* p_de_th_x_vs_xi_[2];
    TProfile* p_de_th_y_vs_xi_[2];
    TProfile* p_de_xi_vs_xi_[2];
};

PlotsProducer::PlotsProducer( const edm::ParameterSet& iConfig ) :
  genProtonsToken_   ( consumes< edm::View<CTPPSSimProtonTrack> >( iConfig.getParameter<edm::InputTag>( "genProtonsTag" ) ) ),
  recoProtons45Token_( consumes< edm::View<CTPPSSimProtonTrack> >( iConfig.getParameter<edm::InputTag>( "recoProtons45Tag" ) ) ),
  recoProtons56Token_( consumes< edm::View<CTPPSSimProtonTrack> >( iConfig.getParameter<edm::InputTag>( "recoProtons56Tag" ) ) ),
  hitsToken_         ( consumes< edm::View<CTPPSSimHit> >( iConfig.getParameter<edm::InputTag>( "potsHitsTag" ) ) ),
  //beamConditions_  ( iConfig.getParameter<edm::ParameterSet>( "beamConditions" ) ),
  detectorPackages_( iConfig.getParameter< std::vector<edm::ParameterSet> >( "detectorPackages" ) )
{
  usesResource( "TFileService" );

  // prepare output
  edm::Service<TFileService> fs;

  // prepare plots - hit distributions
  for ( const auto& det : detectorPackages_ ) {
    const unsigned int pot_id = det.getParameter<unsigned int>( "potId" );
    m_rp_h2_y_vs_x_.insert( std::make_pair( pot_id, fs->make<TH2D>( Form( "h2_rp_hits_%d", pot_id ) , ";x;y", 300, 0., 30E-3, 200, -10E-3, +10E-3 ) ) );
  }

  unsigned short i = 0;
  for ( unsigned int sect : { 45, 56 } ) {
    // prepare plots - histograms
    h_de_vtx_x_[i] = fs->make<TH1D>( Form( "h_de_vtx_x_%d", sect ), Form( ";vtx_{x}^{reco,%d} - vtx_{x}^{sim}", sect ), 100, -40E-6, +40E-6 );
    h_de_vtx_y_[i] = fs->make<TH1D>( Form( "h_de_vtx_y_%d", sect ), Form( ";vtx_{y}^{reco,%d} - vtx_{y}^{sim}", sect ), 100, -250E-6, +250E-6 );
    h_de_th_x_[i] = fs->make<TH1D>( Form( "h_de_th_x_%d", sect ), Form( ";th_{x}^{reco,%d} - th_{x}^{sim}", sect ), 100, -100E-6, +100E-6 );
    h_de_th_y_[i] = fs->make<TH1D>( Form( "h_de_th_y_%d", sect ), Form( ";th_{y}^{reco,%d} - th_{y}^{sim}", sect ), 100, -100E-6, +100E-6 );
    h_de_xi_[i] = fs->make<TH1D>( Form( "h_de_xi_%d", sect ), Form( ";#xi^{reco,%d} - #xi^{sim}", sect ), 100, -5E-3, +5E-3 );

    // prepare plots - 2D histograms
    h2_de_vtx_x_vs_de_xi_[i] = fs->make<TH2D>( Form( "h2_de_vtx_x_vs_de_xi_%d", sect ), Form( ";#Delta#xi^{%d};#Deltavtx_{x}^{%d}", sect, sect ), 50, -5E-3, +5E-3, 50, -40E-6, +40E-6 );
    h2_de_vtx_y_vs_de_xi_[i] = fs->make<TH2D>( Form( "h2_de_vtx_y_vs_de_xi_%d", sect ), Form( ";#Delta#xi^{%d};#Deltavtx_{y}^{%d}", sect, sect ), 50, -5E-3, +5E-3, 50, -250E-6, +250E-6 );
    h2_de_th_x_vs_de_xi_[i] = fs->make<TH2D>( Form( "h2_de_th_x_vs_de_xi_%d", sect ), Form( ";#Delta#xi^{%d};#Deltath_{x}^{%d}", sect, sect ), 50, -5E-3, +5E-3, 50, -100E-6, +100E-6 );
    h2_de_th_y_vs_de_xi_[i] = fs->make<TH2D>( Form( "h2_de_th_y_vs_de_xi_%d", sect ), Form( ";#Delta#xi^{%d};#Deltath_{y}^{%d}", sect, sect ), 50, -5E-3, +5E-3, 50, -100E-6, +100E-6 );
    h2_de_vtx_y_vs_de_th_y_[i] = fs->make<TH2D>( Form( "h2_de_vtx_y_vs_de_th_y_%d", sect ), Form( ";#Deltath_{y}^{%d};#Deltavtx_{y}^{%d}", sect, sect ), 50, -100E-6, +100E-6, 50, -250E-6, +250E-6 );

    // prepare plots - profiles
    p_de_vtx_x_vs_xi_[i] = fs->make<TProfile>( Form( "p_de_vtx_x_vs_xi_%d", sect ), Form( ";#xi;#Deltavtx_{x}^{%d}", sect ), 20, 0., 0.20 );
    p_de_vtx_y_vs_xi_[i] = fs->make<TProfile>( Form( "p_de_vtx_y_vs_xi_%d", sect ), Form( ";#xi;#Deltavtx_{y}^{%d}", sect ), 20, 0., 0.20 );
    p_de_th_x_vs_xi_[i] = fs->make<TProfile>( Form( "p_de_th_x_vs_xi_%d", sect ), Form( ";#xi;#Deltath_{x}^{%d}", sect ), 20, 0., 0.20 );
    p_de_th_y_vs_xi_[i] = fs->make<TProfile>( Form( "p_de_th_y_vs_xi_%d", sect ), Form( ";#xi;#Deltath_{y}^{%d}", sect ), 20, 0., 0.20 );
    p_de_xi_vs_xi_[i] = fs->make<TProfile>( Form( "p_de_xi_vs_xi_%d", sect ), Form( ";#xi;#Delta#xi^{%d}", sect ), 20, 0., 0.20 );

    i++;
  }
}

PlotsProducer::~PlotsProducer()
{}

void
PlotsProducer::analyze( const edm::Event& iEvent, const edm::EventSetup& )
{
  edm::Handle< edm::View<CTPPSSimHit> > hits;
  iEvent.getByToken( hitsToken_, hits );

  for ( const auto& hit : *hits ) {
    m_rp_h2_y_vs_x_[hit.potId().detectorDecId()/10]->Fill( hit.getX0(), hit.getY0() );
  }

  edm::Handle< edm::View<CTPPSSimProtonTrack> > gen_protons, reco_protons[2];
  iEvent.getByToken( genProtonsToken_, gen_protons );
  if ( gen_protons->size()>1 ) {
    throw cms::Exception("PlotsProducer") << "Not yet supporting multiple generated protons per event";
  }
  const CTPPSSimProtonTrack gen_pro = *( gen_protons->ptrAt( 0 ) );
  const double gen_xi = gen_pro.xi();

  iEvent.getByToken( recoProtons45Token_, reco_protons[0] );
  iEvent.getByToken( recoProtons56Token_, reco_protons[1] );

  for ( unsigned short i=0; i<2; ++i ) {
    for ( const auto& rec_pro : *reco_protons[i] ) {
      const double xi = rec_pro.xi();
      const double de_vtx_x = rec_pro.vertex().x()-gen_pro.vertex().x();
      const double de_vtx_y = rec_pro.vertex().y()-gen_pro.vertex().y();
      const double de_th_x = rec_pro.direction().x()-gen_pro.direction().x();
      const double de_th_y = rec_pro.direction().y()-gen_pro.direction().y();
      const double de_xi = xi-gen_xi;

      h_de_vtx_x_[i]->Fill( de_vtx_x );
      h_de_vtx_y_[i]->Fill( de_vtx_y );
      h_de_th_x_[i]->Fill( de_th_x );
      h_de_th_y_[i]->Fill( de_th_y );
      h_de_xi_[i]->Fill( de_xi );

      h2_de_vtx_x_vs_de_xi_[i]->Fill( de_xi, de_vtx_x );
      h2_de_vtx_y_vs_de_xi_[i]->Fill( de_xi, de_vtx_y );
      h2_de_th_x_vs_de_xi_[i]->Fill( de_xi, de_th_x );
      h2_de_th_y_vs_de_xi_[i]->Fill( de_xi, de_th_y );
      h2_de_vtx_y_vs_de_th_y_[i]->Fill( de_th_y, de_vtx_y );

      p_de_vtx_x_vs_xi_[i]->Fill( gen_xi, de_vtx_x );
      p_de_vtx_y_vs_xi_[i]->Fill( gen_xi, de_vtx_y );
      p_de_th_x_vs_xi_[i]->Fill( gen_xi, de_th_x );
      p_de_th_y_vs_xi_[i]->Fill( gen_xi, de_th_y );
      p_de_xi_vs_xi_[i]->Fill( gen_xi, de_xi );
    }
  }
/*
	// save plots
	gDirectory = f_out;

	for (const auto &it : m_rp_h2_y_vs_x)
	{
		char buf[100];
		sprintf(buf, "h2_y_vs_x_RP%u", it.first);
		it.second->Write(buf);
	}

	gDirectory = f_out->mkdir("sector 45");
	h_de_vtx_x_45->Write();
	h_de_vtx_y_45->Write();
	h_de_th_x_45->Write();
	h_de_th_y_45->Write();
	h_de_xi_45->Write();

	h2_de_vtx_x_vs_de_xi_45->Write();
	h2_de_vtx_y_vs_de_xi_45->Write();
	h2_de_th_x_vs_de_xi_45->Write();
	h2_de_th_y_vs_de_xi_45->Write();
	h2_de_vtx_y_vs_de_th_y_45->Write();

	p_de_vtx_x_vs_xi_45->Write();
	p_de_vtx_y_vs_xi_45->Write();
	p_de_th_x_vs_xi_45->Write();
	p_de_th_y_vs_xi_45->Write();
	p_de_xi_vs_xi_45->Write();

	ProfileToRMSGraph(p_de_vtx_x_vs_xi_45, "g_rms_de_vtx_x_vs_xi_45")->Write();
	ProfileToRMSGraph(p_de_vtx_y_vs_xi_45, "g_rms_de_vtx_y_vs_xi_45")->Write();
	ProfileToRMSGraph(p_de_th_x_vs_xi_45, "g_rms_de_th_x_vs_xi_45")->Write();
	ProfileToRMSGraph(p_de_th_y_vs_xi_45, "g_rms_de_th_y_vs_xi_45")->Write();
	ProfileToRMSGraph(p_de_xi_vs_xi_45, "g_rms_de_xi_vs_xi_45")->Write();

	gDirectory = f_out->mkdir("sector 56");
	h_de_vtx_x_56->Write();
	h_de_vtx_y_56->Write();
	h_de_th_x_56->Write();
	h_de_th_y_56->Write();
	h_de_xi_56->Write();

	h2_de_vtx_x_vs_de_xi_56->Write();
	h2_de_vtx_y_vs_de_xi_56->Write();
	h2_de_th_x_vs_de_xi_56->Write();
	h2_de_th_y_vs_de_xi_56->Write();
	h2_de_vtx_y_vs_de_th_y_56->Write();

	ProfileToRMSGraph(p_de_vtx_x_vs_xi_56, "g_rms_de_vtx_x_vs_xi_56")->Write();
	ProfileToRMSGraph(p_de_vtx_y_vs_xi_56, "g_rms_de_vtx_y_vs_xi_56")->Write();
	ProfileToRMSGraph(p_de_th_x_vs_xi_56, "g_rms_de_th_x_vs_xi_56")->Write();
	ProfileToRMSGraph(p_de_th_y_vs_xi_56, "g_rms_de_th_y_vs_xi_56")->Write();
	ProfileToRMSGraph(p_de_xi_vs_xi_56, "g_rms_de_xi_vs_xi_56")->Write();
*/
}

// ------------ method called once each job just before starting event loop  ------------
void
PlotsProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
PlotsProducer::endJob()
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
PlotsProducer::fillDescriptions( edm::ConfigurationDescriptions& descriptions ) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault( desc );
}

//define this as a plug-in
DEFINE_FWK_MODULE( PlotsProducer );



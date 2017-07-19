/****************************************************************************
 *
 * This is a part of CTPPS validation software
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 *
 ****************************************************************************/

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/ProtonReco/interface/ProtonTrack.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TGraph.h"
#include "TGraphErrors.h"

#include <map>

class ParamValidation : public edm::one::EDAnalyzer<edm::one::SharedResources>
{
  public:
    explicit ParamValidation( const edm::ParameterSet& );
    ~ParamValidation();

    static void fillDescriptions( edm::ConfigurationDescriptions& descriptions );

  private:
    virtual void beginJob() override;
    virtual void analyze( const edm::Event&, const edm::EventSetup& ) override;
    virtual void endJob() override;

    edm::EDGetTokenT<edm::HepMCProduct> genProtonsToken_;
    edm::EDGetTokenT< edm::View<reco::ProtonTrack> > recoProtonsToken_;
    edm::EDGetTokenT< edm::View<CTPPSLocalTrackLite> > tracksToken_;

    //edm::ParameterSet beamConditions_;
    double sqrtS_;
    std::vector<edm::ParameterSet> detectorPackages_;

    std::map<unsigned int,TH2D*> m_rp_h2_y_vs_x_;

    // gen-level plots
    TH1D* h_gen_vtx_x_, *h_gen_vtx_y_, *h_gen_th_x_, *h_gen_th_y_, *h_gen_xi_;
    // reco-gen 1D plots
    TH1D* h_de_vtx_x_[2], *h_de_vtx_y_[2], *h_de_th_x_[2], *h_de_th_y_[2], *h_de_xi_[2];
    // reco-gen 2D (vs delta(xi), ...) plots
    TH2D* h2_de_vtx_x_vs_de_xi_[2], *h2_de_vtx_y_vs_de_xi_[2], *h2_de_th_x_vs_de_xi_[2], *h2_de_th_y_vs_de_xi_[2], *h2_de_vtx_y_vs_de_th_y_[2];
    // profiles
    TProfile* p_de_vtx_x_vs_xi_[2], *p_de_vtx_y_vs_xi_[2], *p_de_th_x_vs_xi_[2], *p_de_th_y_vs_xi_[2], *p_de_xi_vs_xi_[2];
};

ParamValidation::ParamValidation( const edm::ParameterSet& iConfig ) :
  genProtonsToken_ ( consumes<edm::HepMCProduct>( iConfig.getParameter<edm::InputTag>( "genProtonsTag" ) ) ),
  recoProtonsToken_( consumes< edm::View<reco::ProtonTrack> >( iConfig.getParameter<edm::InputTag>( "recoProtonsTag" ) ) ),
  tracksToken_     ( consumes< edm::View<CTPPSLocalTrackLite> >( iConfig.getParameter<edm::InputTag>( "potsTracksTag" ) ) ),
  sqrtS_           ( iConfig.getParameter<edm::ParameterSet>( "beamConditions" ).getParameter<double>( "sqrtS" ) ),
  detectorPackages_( iConfig.getParameter< std::vector<edm::ParameterSet> >( "detectorPackages" ) )
{
  usesResource( "TFileService" );

  // prepare output
  edm::Service<TFileService> fs;

  // prepare plots - generator level
  TFileDirectory gen_dir = fs->mkdir( "gen level" );
  h_gen_vtx_x_ = gen_dir.make<TH1D>( "h_gen_vtx_x", ";vtx_{x}^{gen}", 100, -1.e-4, 1.e-4 );
  h_gen_vtx_y_ = gen_dir.make<TH1D>( "h_gen_vtx_y", ";vtx_{y}^{gen}", 100, -1.e-3, 1.e-3 );
  h_gen_th_x_ = gen_dir.make<TH1D>( "h_gen_th_x", ";#theta_{x}^{gen}", 100, -1.e-4, 1.e-4 );
  h_gen_th_y_ = gen_dir.make<TH1D>( "h_gen_th_y", ";#theta_{y}^{gen}", 100, -1.e-4, 1.e-4 );
  h_gen_xi_ = gen_dir.make<TH1D>( "h_gen_xi", ";#xi^{gen}", 100, 0.0, 0.25 );

  unsigned short i = 0;
  for ( unsigned int sect : { 45, 56 } ) {
    TFileDirectory sect_dir = fs->mkdir( Form( "sector %d", sect ) );
    // prepare plots - histograms
    h_de_vtx_x_[i] = sect_dir.make<TH1D>( Form( "h_de_vtx_x_%d", sect ), Form( ";vtx_{x}^{reco,%d} - vtx_{x}^{gen}", sect ), 100, -40.0e-6, +40.0e-6 );
    h_de_vtx_y_[i] = sect_dir.make<TH1D>( Form( "h_de_vtx_y_%d", sect ), Form( ";vtx_{y}^{reco,%d} - vtx_{y}^{gen}", sect ), 100, -250.0e-6, +250.0e-6 );
    h_de_th_x_[i] = sect_dir.make<TH1D>( Form( "h_de_th_x_%d", sect ), Form( ";#theta_{x}^{reco,%d} - #theta_{x}^{gen}", sect ), 100, -100.0e-6, +100.0e-6 );
    h_de_th_y_[i] = sect_dir.make<TH1D>( Form( "h_de_th_y_%d", sect ), Form( ";#theta_{y}^{reco,%d} - #theta_{y}^{gen}", sect ), 100, -100.0e-6, +100.0e-6 );
    h_de_xi_[i] = sect_dir.make<TH1D>( Form( "h_de_xi_%d", sect ), Form( ";#xi^{reco,%d} - #xi^{gen}", sect ), 100, -5.0-3, +5.0e-3 );

    // prepare plots - 2D histograms
    h2_de_vtx_x_vs_de_xi_[i] = sect_dir.make<TH2D>( Form( "h2_de_vtx_x_vs_de_xi_%d", sect ), Form( ";#Delta#xi^{%d};#Deltavtx_{x}^{%d}", sect, sect ), 50, -5.0e-3, +5.0e-3, 50, -40.0e-6, +40.0e-6 );
    h2_de_vtx_y_vs_de_xi_[i] = sect_dir.make<TH2D>( Form( "h2_de_vtx_y_vs_de_xi_%d", sect ), Form( ";#Delta#xi^{%d};#Deltavtx_{y}^{%d}", sect, sect ), 50, -5.0e-3, +5.0e-3, 50, -250.0e-6, +250.0e-6 );
    h2_de_th_x_vs_de_xi_[i] = sect_dir.make<TH2D>( Form( "h2_de_th_x_vs_de_xi_%d", sect ), Form( ";#Delta#xi^{%d};#Delta#theta_{x}^{%d}", sect, sect ), 50, -5.0e-3, +5.0e-3, 50, -100.0e-6, +100.0e-6 );
    h2_de_th_y_vs_de_xi_[i] = sect_dir.make<TH2D>( Form( "h2_de_th_y_vs_de_xi_%d", sect ), Form( ";#Delta#xi^{%d};#Delta#theta_{y}^{%d}", sect, sect ), 50, -5.0e-3, +5.0e-3, 50, -100.0e-6, +100.0e-6 );
    h2_de_vtx_y_vs_de_th_y_[i] = sect_dir.make<TH2D>( Form( "h2_de_vtx_y_vs_de_th_y_%d", sect ), Form( ";#Delta#theta_{y}^{%d};#Deltavtx_{y}^{%d}", sect, sect ), 50, -100.0e-6, +100.0e-6, 50, -250.0e-6, +250.0e-6 );

    // prepare plots - profiles
    p_de_vtx_x_vs_xi_[i] = sect_dir.make<TProfile>( Form( "p_de_vtx_x_vs_xi_%d", sect ), Form( ";#xi;#Deltavtx_{x}^{%d}", sect ), 20, 0.0, 0.20 );
    p_de_vtx_y_vs_xi_[i] = sect_dir.make<TProfile>( Form( "p_de_vtx_y_vs_xi_%d", sect ), Form( ";#xi;#Deltavtx_{y}^{%d}", sect ), 20, 0.0, 0.20 );
    p_de_th_x_vs_xi_[i] = sect_dir.make<TProfile>( Form( "p_de_th_x_vs_xi_%d", sect ), Form( ";#xi;#Delta#theta_{x}^{%d}", sect ), 20, 0.0, 0.20 );
    p_de_th_y_vs_xi_[i] = sect_dir.make<TProfile>( Form( "p_de_th_y_vs_xi_%d", sect ), Form( ";#xi;#Delta#theta_{y}^{%d}", sect ), 20, 0.0, 0.20 );
    p_de_xi_vs_xi_[i] = sect_dir.make<TProfile>( Form( "p_de_xi_vs_xi_%d", sect ), Form( ";#xi;#Delta#xi^{%d}", sect ), 20, 0.0, 0.20 );

    i++;
  }

  // prepare plots - hit distributions
  TFileDirectory hm_dir = fs->mkdir( "hitmaps" );
  for ( const auto& det : detectorPackages_ ) {
    const TotemRPDetId pot_id( det.getParameter<unsigned int>( "potId" ) );
    m_rp_h2_y_vs_x_.insert( std::make_pair( pot_id, hm_dir.make<TH2D>( Form( "h2_rp_hits_arm%d_rp%d", pot_id.arm(), pot_id.rp() ) , ";x (mm);y (mm)", 300, 0.0, 30.0, 200, -10.0, +10.0 ) ) );
  }
}

ParamValidation::~ParamValidation()
{}

void
ParamValidation::analyze( const edm::Event& iEvent, const edm::EventSetup& )
{
  edm::Handle< edm::View<CTPPSLocalTrackLite> > tracks;
  iEvent.getByToken( tracksToken_, tracks );

  for ( const auto& trk : *tracks ) {
    const TotemRPDetId det_id( trk.getRPId() );
    m_rp_h2_y_vs_x_[det_id.rawId()]->Fill( trk.getX(), trk.getY() );
  }

  edm::Handle<edm::HepMCProduct> protons;
  iEvent.getByToken( genProtonsToken_, protons );
  const HepMC::GenEvent& evt = protons->getHepMCData();
  /*if ( evt.particles_size()>1 ) {
    throw cms::Exception("ParamValidation") << "Not yet supporting multiple generated protons per event";
  }*/

  edm::Handle< edm::View<reco::ProtonTrack> > reco_protons;
  iEvent.getByToken( recoProtonsToken_, reco_protons );

  for ( HepMC::GenEvent::particle_const_iterator p=evt.particles_begin(); p!=evt.particles_end(); ++p ) {
    const HepMC::GenParticle* gen_pro = *p;
    if ( gen_pro->pdg_id()!=2212 ) continue; // only transport stable protons
    if ( gen_pro->status()!=1 && gen_pro->status()<83 ) continue;

    const HepMC::FourVector& gen_pos = gen_pro->production_vertex()->position();

    const double gen_xi = 1.-gen_pro->momentum().e()/sqrtS_*2.;
    const double gen_th_x = atan2( gen_pro->momentum().px(), gen_pro->momentum().pz() );
    const double gen_th_y = atan2( gen_pro->momentum().py(), gen_pro->momentum().pz() );
    const double gen_vtx_x = gen_pos.x(), gen_vtx_y = gen_pos.y();

    h_gen_vtx_x_->Fill( gen_vtx_x );
    h_gen_vtx_y_->Fill( gen_vtx_y );
    h_gen_th_x_->Fill( gen_th_x );
    h_gen_th_y_->Fill( gen_th_y );
    h_gen_xi_->Fill( gen_xi );

    const unsigned short side_id = ( gen_pro->momentum().pz()>0 ) ? 0 : 1;
    for ( const auto& rec_pro : *reco_protons ) {
      if ( rec_pro.lhcSector!=side_id ) continue;
      const double rec_xi = rec_pro.xi();

      //std::cout << "(" << reco_protons[side_id]->size() << ")--> sector " << i << ": " << gen_xi << " / " << rec_xi << std::endl;

      const double de_vtx_x = rec_pro.vertex().x()-gen_vtx_x;
      const double de_vtx_y = rec_pro.vertex().y()-gen_vtx_y;
      const double de_th_x = rec_pro.direction().x()-gen_th_x;
      const double de_th_y = rec_pro.direction().y()-gen_th_y;
      const double de_xi = rec_xi-gen_xi;

      h_de_vtx_x_[side_id]->Fill( de_vtx_x );
      h_de_vtx_y_[side_id]->Fill( de_vtx_y );
      h_de_th_x_[side_id]->Fill( de_th_x );
      h_de_th_y_[side_id]->Fill( de_th_y );
      h_de_xi_[side_id]->Fill( de_xi );

      h2_de_vtx_x_vs_de_xi_[side_id]->Fill( de_xi, de_vtx_x );
      h2_de_vtx_y_vs_de_xi_[side_id]->Fill( de_xi, de_vtx_y );
      h2_de_th_x_vs_de_xi_[side_id]->Fill( de_xi, de_th_x );
      h2_de_th_y_vs_de_xi_[side_id]->Fill( de_xi, de_th_y );
      h2_de_vtx_y_vs_de_th_y_[side_id]->Fill( de_th_y, de_vtx_y );

      p_de_vtx_x_vs_xi_[side_id]->Fill( gen_xi, de_vtx_x );
      p_de_vtx_y_vs_xi_[side_id]->Fill( gen_xi, de_vtx_y );
      p_de_th_x_vs_xi_[side_id]->Fill( gen_xi, de_th_x );
      p_de_th_y_vs_xi_[side_id]->Fill( gen_xi, de_th_y );
      p_de_xi_vs_xi_[side_id]->Fill( gen_xi, de_xi );
    }
  }
}

void
ParamValidation::beginJob()
{}

void
ParamValidation::endJob()
{
/*
  ProfileToRMSGraph(p_de_vtx_x_vs_xi_45, "g_rms_de_vtx_x_vs_xi_45")->Write();
  ProfileToRMSGraph(p_de_vtx_y_vs_xi_45, "g_rms_de_vtx_y_vs_xi_45")->Write();
  ProfileToRMSGraph(p_de_th_x_vs_xi_45, "g_rms_de_th_x_vs_xi_45")->Write();
  ProfileToRMSGraph(p_de_th_y_vs_xi_45, "g_rms_de_th_y_vs_xi_45")->Write();
  ProfileToRMSGraph(p_de_xi_vs_xi_45, "g_rms_de_xi_vs_xi_45")->Write();

  ProfileToRMSGraph(p_de_vtx_x_vs_xi_56, "g_rms_de_vtx_x_vs_xi_56")->Write();
  ProfileToRMSGraph(p_de_vtx_y_vs_xi_56, "g_rms_de_vtx_y_vs_xi_56")->Write();
  ProfileToRMSGraph(p_de_th_x_vs_xi_56, "g_rms_de_th_x_vs_xi_56")->Write();
  ProfileToRMSGraph(p_de_th_y_vs_xi_56, "g_rms_de_th_y_vs_xi_56")->Write();
  ProfileToRMSGraph(p_de_xi_vs_xi_56, "g_rms_de_xi_vs_xi_56")->Write();
*/
}

void
ParamValidation::fillDescriptions( edm::ConfigurationDescriptions& descriptions ) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault( desc );
}

//define this as a plug-in
DEFINE_FWK_MODULE( ParamValidation );


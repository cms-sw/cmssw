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
#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

#include "TH1D.h"
#include "TH2D.h"

#include <map>

class ScoringPlaneValidation : public edm::one::EDAnalyzer<edm::one::SharedResources>
{
  public:
    explicit ScoringPlaneValidation( const edm::ParameterSet& );
    ~ScoringPlaneValidation() {}

    static void fillDescriptions( edm::ConfigurationDescriptions& descriptions );

  private:
    virtual void beginJob() override {}
    virtual void analyze( const edm::Event&, const edm::EventSetup& ) override;
    virtual void endJob() override {}

    edm::EDGetTokenT<edm::View<CTPPSLocalTrackLite> > spTracksToken_, recoTracksToken_;

    std::vector<edm::ParameterSet> detectorPackages_;

    std::map<unsigned int,TH1D*> m_rp_h_xpos_[2], m_rp_h_ypos_[2];
    std::map<unsigned int,TH2D*> m_rp_h2_xpos_vs_xpos_, m_rp_h2_ypos_vs_ypos_;
    std::map<unsigned int,TH1D*> m_rp_h_de_x_, m_rp_h_de_y_;
};

ScoringPlaneValidation::ScoringPlaneValidation( const edm::ParameterSet& iConfig ) :
  spTracksToken_  ( consumes<edm::View<CTPPSLocalTrackLite> >( iConfig.getParameter<edm::InputTag>( "spTracksTag" ) ) ),
  recoTracksToken_( consumes<edm::View<CTPPSLocalTrackLite> >( iConfig.getParameter<edm::InputTag>( "recoTracksTag" ) ) ),
  detectorPackages_( iConfig.getParameter< std::vector<edm::ParameterSet> >( "detectorPackages" ) )
{
  usesResource( "TFileService" );

  // prepare output
  edm::Service<TFileService> fs;

  // prepare plots - hit distributions
  for ( const auto& det : detectorPackages_ ) {
    const CTPPSDetId pot_id( det.getParameter<unsigned int>( "potId" ) );
    TFileDirectory pot_dir = fs->mkdir( Form( "arm%d_pot%d", pot_id.arm(), pot_id.rp() ) );
    unsigned short i = 0;
    for ( const auto& nm : { "scorpl", "reco" } ) {
      m_rp_h_xpos_[i].insert( std::make_pair( pot_id, pot_dir.make<TH1D>( Form( "h_rp_xpos_arm%d_rp%d_%s", pot_id.arm(), pot_id.rp(), nm ), ";x (mm)", 300, -10.0, 50.0 ) ) );
      m_rp_h_ypos_[i].insert( std::make_pair( pot_id, pot_dir.make<TH1D>( Form( "h_rp_ypos_arm%d_rp%d_%s", pot_id.arm(), pot_id.rp(), nm ), ";y (mm)", 300, -30.0, 30.0 ) ) );
      i++;
    }
    m_rp_h2_xpos_vs_xpos_.insert( std::make_pair( pot_id, pot_dir.make<TH2D>( Form( "h2_rp_xpos_corr_arm%d_rp%d", pot_id.arm(), pot_id.rp() ), ";x (scoring plane) (mm);x (reco) (mm)", 300, -10.0, 50.0, 300, -10.0, 50.0 ) ) );
    m_rp_h2_ypos_vs_ypos_.insert( std::make_pair( pot_id, pot_dir.make<TH2D>( Form( "h2_rp_ypos_corr_arm%d_rp%d", pot_id.arm(), pot_id.rp() ), ";y (scoring plane) (mm);y (reco) (mm)", 300, -30.0, 30.0, 300, -30.0, 30.0 ) ) );
    m_rp_h_de_x_.insert( std::make_pair( pot_id, pot_dir.make<TH1D>( Form( "h_rp_de_x_arm%d_rp%d", pot_id.arm(), pot_id.rp() ), ";x (reco) - x (scoring plane) (mm)", 100, -0.1, 0.1 ) ) );
    m_rp_h_de_y_.insert( std::make_pair( pot_id, pot_dir.make<TH1D>( Form( "h_rp_de_y_arm%d_rp%d", pot_id.arm(), pot_id.rp() ), ";y (reco) - y (scoring plane) (mm)", 100, -0.1, 0.1 ) ) );
  }
}

void
ScoringPlaneValidation::analyze( const edm::Event& iEvent, const edm::EventSetup& )
{
  edm::Handle<edm::View<CTPPSLocalTrackLite> > sp_tracks, reco_tracks;
  iEvent.getByToken( spTracksToken_, sp_tracks );
  iEvent.getByToken( recoTracksToken_, reco_tracks );

  for ( const auto& trk : *sp_tracks ) {
    const CTPPSDetId det_id( trk.getRPId() );
    m_rp_h_xpos_[0][det_id]->Fill( trk.getX() );
    m_rp_h_ypos_[0][det_id]->Fill( trk.getY() );
  }
  for ( const auto& trk : *reco_tracks ) {
    const CTPPSDetId det_id( trk.getRPId() );
    m_rp_h_xpos_[1][det_id]->Fill( trk.getX() );
    m_rp_h_ypos_[1][det_id]->Fill( trk.getY() );
    for ( const auto& trk_sp : *sp_tracks ) {
      if ( CTPPSDetId( trk_sp.getRPId() ) != det_id ) continue;
      m_rp_h2_xpos_vs_xpos_[det_id]->Fill( trk.getX(), trk_sp.getX() );
      m_rp_h2_ypos_vs_ypos_[det_id]->Fill( trk.getY(), trk_sp.getY() );
      m_rp_h_de_x_[det_id]->Fill( trk.getX()-trk_sp.getX() );
      m_rp_h_de_y_[det_id]->Fill( trk.getY()-trk_sp.getY() );
    }
  }
}

void
ScoringPlaneValidation::fillDescriptions( edm::ConfigurationDescriptions& descriptions ) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault( desc );
}

//define this as a plug-in
DEFINE_FWK_MODULE( ScoringPlaneValidation );


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

#include "TCanvas.h"
#include "TH1D.h"

#include <map>

class ScoringPlaneValidation : public edm::one::EDAnalyzer<edm::one::SharedResources>
{
  public:
    explicit ScoringPlaneValidation( const edm::ParameterSet& );
    ~ScoringPlaneValidation();

    static void fillDescriptions( edm::ConfigurationDescriptions& descriptions );

  private:
    virtual void beginJob() override;
    virtual void analyze( const edm::Event&, const edm::EventSetup& ) override;
    virtual void endJob() override;

    edm::EDGetTokenT<edm::View<CTPPSLocalTrackLite> > spTracksToken_, recoTracksToken_;

    std::vector<edm::ParameterSet> detectorPackages_;

    std::map<unsigned int,TH1D*> m_rp_h_xpos_[2], m_rp_h_ypos_[2];
    //std::map<unsigned int,TCanvas*> m_rp_c_xpos_, m_rp_c_ypos_;
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
      m_rp_h_xpos_[i].insert( std::make_pair( pot_id, pot_dir.make<TH1D>( Form( "h_rp_xpos_arm%d_rp%d_%s", pot_id.arm(), pot_id.rp(), nm ), ";x (mm)", 200, 0.0, 20.0 ) ) );
      m_rp_h_ypos_[i].insert( std::make_pair( pot_id, pot_dir.make<TH1D>( Form( "h_rp_ypos_arm%d_rp%d_%s", pot_id.arm(), pot_id.rp(), nm ), ";y (mm)", 200, -10.0, 10.0 ) ) );
      i++;
    }
    //m_rp_c_xpos_.insert( std::make_pair( pot_id, pot_dir.make<TCanvas>( Form( "c_rp_xpos_arm%d_rp%d", pot_id.arm(), pot_id.rp() ) ) ) );
    //m_rp_c_ypos_.insert( std::make_pair( pot_id, pot_dir.make<TCanvas>( Form( "c_rp_ypos_arm%d_rp%d", pot_id.arm(), pot_id.rp() ) ) ) );
  }
}

ScoringPlaneValidation::~ScoringPlaneValidation()
{}

void
ScoringPlaneValidation::analyze( const edm::Event& iEvent, const edm::EventSetup& )
{
  edm::Handle<edm::View<CTPPSLocalTrackLite> > sp_tracks, reco_tracks;
  iEvent.getByToken( spTracksToken_, sp_tracks );
  iEvent.getByToken( recoTracksToken_, reco_tracks );

  for ( const auto& trk : *sp_tracks ) {
    const CTPPSDetId det_id( trk.getRPId() );
    m_rp_h_xpos_[0][det_id]->Fill( trk.getX()*1.0 );
    m_rp_h_ypos_[0][det_id]->Fill( trk.getY()*1.0 );
  }
  for ( const auto& trk : *reco_tracks ) {
    const CTPPSDetId det_id( trk.getRPId() );
    m_rp_h_xpos_[1][det_id]->Fill( trk.getX()*1.0 );
    m_rp_h_ypos_[1][det_id]->Fill( trk.getY()*1.0 );
  }
}

void
ScoringPlaneValidation::beginJob()
{}

void
ScoringPlaneValidation::endJob()
{}

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


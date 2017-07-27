/****************************************************************************
 *
 * This is a part of CTPPS offline software
 * Authors:
 *   Leszek Grzanka
 *   Jan Kašpar
 *   Laurent Forthomme
 *
 ****************************************************************************/

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "CondFormats/CTPPSOpticsObjects/interface/LHCOpticsApproximator.h"

#include "CLHEP/Random/RandGauss.h"

#include <unordered_map>

class CTPPSScoringPlaneSimulation : public edm::stream::EDProducer<>
{
  public:
    explicit CTPPSScoringPlaneSimulation( const edm::ParameterSet& );
    ~CTPPSScoringPlaneSimulation();

    static void fillDescriptions( edm::ConfigurationDescriptions& descriptions );

  private:
    struct CTPPSPotInfo {
      CTPPSPotInfo() : detid( 0 ), resolution( 0.0 ), z_position( 0.0 ), approximator( 0 ) {}
      CTPPSPotInfo( const CTPPSDetId& det_id, double resol, double z_position, LHCOpticsApproximator* approx ) :
        detid( det_id ), resolution( resol ), z_position( z_position ), approximator( approx ) {}

      CTPPSDetId detid;
      double resolution;
      double z_position;
      LHCOpticsApproximator* approximator;
    };

    virtual void produce( edm::Event&, const edm::EventSetup& ) override;
    void transportProtonTrack( const HepMC::GenParticle*, std::vector<CTPPSLocalTrackLite>& );

    edm::EDGetTokenT<edm::HepMCProduct> protonsToken_;

    double sqrtS_;
    double halfCrossingAngleSector45_, halfCrossingAngleSector56_;
    double yOffsetSector45_, yOffsetSector56_;

    bool simulateDetectorsResolution_;

    bool checkApertures_;

    edm::FileInPath opticsFileBeam1_, opticsFileBeam2_;
    std::vector<edm::ParameterSet> detectorPackages_;

    std::vector<CTPPSPotInfo> pots_;

    CLHEP::HepRandomEngine* rnd_;

    static const bool invertBeamCoordinatesSystem_;
};

const bool CTPPSScoringPlaneSimulation::invertBeamCoordinatesSystem_ = true;

CTPPSScoringPlaneSimulation::CTPPSScoringPlaneSimulation( const edm::ParameterSet& iConfig ) :
  protonsToken_( consumes<edm::HepMCProduct>( iConfig.getParameter<edm::InputTag>( "beamParticlesTag" ) ) ),
  sqrtS_                      ( iConfig.getParameter<double>( "sqrtS" ) ),
  halfCrossingAngleSector45_  ( iConfig.getParameter<double>( "halfCrossingAngleSector45" ) ),
  halfCrossingAngleSector56_  ( iConfig.getParameter<double>( "halfCrossingAngleSector56" ) ),
  yOffsetSector45_            ( iConfig.getParameter<double>( "yOffsetSector45" ) ),
  yOffsetSector56_            ( iConfig.getParameter<double>( "yOffsetSector56" ) ),
  simulateDetectorsResolution_( iConfig.getParameter<bool>( "simulateDetectorsResolution" ) ),
  checkApertures_             ( iConfig.getParameter<bool>( "checkApertures" ) ),
  opticsFileBeam1_            ( iConfig.getParameter<edm::FileInPath>( "opticsFileBeam1" ) ),
  opticsFileBeam2_            ( iConfig.getParameter<edm::FileInPath>( "opticsFileBeam2" ) ),
  detectorPackages_           ( iConfig.getParameter< std::vector<edm::ParameterSet> >( "detectorPackages" ) ),
  rnd_( 0 )
{
  produces< std::vector<CTPPSLocalTrackLite> >();

  auto f_in_optics_beam1 = std::make_unique<TFile>( opticsFileBeam1_.fullPath().c_str() ),
       f_in_optics_beam2 = std::make_unique<TFile>( opticsFileBeam2_.fullPath().c_str() );

  // load optics and interpolators
  for ( const auto& rp : detectorPackages_ ) {
    const std::string interp_name = rp.getParameter<std::string>( "interpolatorName" );
    const unsigned int raw_detid = rp.getParameter<unsigned int>( "potId" );
    const double det_resol = rp.getParameter<double>( "resolution" );
    const double z_position = rp.getParameter<double>( "zPosition" );
    CTPPSDetId detid( raw_detid );

    if ( detid.arm()==0 ) // sector 45 -- beam 2
      pots_.emplace_back( detid, det_resol, z_position, dynamic_cast<LHCOpticsApproximator*>( f_in_optics_beam2->Get( interp_name.c_str() ) ) );
    if ( detid.arm()==1 ) // sector 56 -- beam 1
      pots_.emplace_back( detid, det_resol, z_position, dynamic_cast<LHCOpticsApproximator*>( f_in_optics_beam1->Get( interp_name.c_str() ) ) );
  }
}

CTPPSScoringPlaneSimulation::~CTPPSScoringPlaneSimulation()
{}

void
CTPPSScoringPlaneSimulation::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  std::unique_ptr< std::vector<CTPPSLocalTrackLite> > pOut( new std::vector<CTPPSLocalTrackLite> );

  if ( simulateDetectorsResolution_ ) {
    edm::Service<edm::RandomNumberGenerator> rng;
    rnd_ = &( rng->getEngine( iEvent.streamID() ) );
  }

  edm::Handle<edm::HepMCProduct> protons;
  iEvent.getByToken( protonsToken_, protons );
  const HepMC::GenEvent& evt = protons->getHepMCData();

  // run simulation
  for ( HepMC::GenEvent::particle_const_iterator p=evt.particles_begin(); p!=evt.particles_end(); ++p ) {
    if ( ( *p )->status()!=1 || ( *p )->pdg_id()!=2212 ) continue; // only transport stable protons

    std::vector<CTPPSLocalTrackLite> tracks;
    transportProtonTrack( *p, tracks );
    //FIXME add an association map proton track <-> sim tracks
    for ( const auto& trk : tracks ) {
      pOut->push_back( trk );
    }
  }

  iEvent.put( std::move( pOut ) );
}

void
CTPPSScoringPlaneSimulation::transportProtonTrack( const HepMC::GenParticle* in_trk, std::vector<CTPPSLocalTrackLite>& out_tracks )
{
  /// implemented according to LHCOpticsApproximator::Transport_m_GeV
  /// xi is positive for diffractive protons, thus proton momentum p = (1-xi) * p_nom
  /// horizontal component of proton momentum: p_x = th_x * (1-xi) * p_nom

  const HepMC::GenVertex* vtx = in_trk->production_vertex();
  const HepMC::FourVector mom = in_trk->momentum();
  const double xi = 1.-mom.e()/sqrtS_*2.0;
  // convert physics kinematics to the LHC reference frame
  double th_x = atan2( mom.px(), fabs( mom.pz() ) );
  double th_y = atan2( mom.py(), fabs( mom.pz() ) );
  while ( th_x<-M_PI ) th_x += 2*M_PI; while ( th_x>+M_PI ) th_x -= 2*M_PI;
  while ( th_y<-M_PI ) th_y += 2*M_PI; while ( th_y>+M_PI ) th_y -= 2*M_PI;
  if ( mom.pz()>0.0 ) { th_x = -th_x; }

  double vtx_x = -vtx->position().x(), vtx_y = vtx->position().y(); // express in metres

  double vtx_y_offset = 0.0;
  // CMS convention
  if ( mom.z()>0.0 ) { // sector 45
    vtx_y_offset = yOffsetSector45_;
  }
  if ( mom.z()<0.0 ) { // sector 56
    vtx_y_offset = yOffsetSector56_;
  }

  // transport the proton into each pot
  for ( const auto& rp : pots_ ) {
    // first check the side
    if ( rp.detid.arm()==0 && mom.z()<0.0 ) continue; // sector 45
    if ( rp.detid.arm()==1 && mom.z()>0.0 ) continue; // sector 56

    // transport proton to its corresponding RP
    std::array<double,5> kin_in = { { vtx_x, th_x * ( 1.-xi ), vtx_y + vtx_y_offset, th_y * ( 1.-xi ), -xi } }, kin_out;

    bool proton_transported = rp.approximator->Transport( kin_in.data(), kin_out.data(), checkApertures_, invertBeamCoordinatesSystem_ );

    // stop if proton not transportable
    if ( !proton_transported ) return;

    const double rp_resol = ( simulateDetectorsResolution_ ) ? rp.resolution : 0.0;

    // simulate detector resolution
    kin_out[0] += CLHEP::RandGauss::shoot( rnd_ ) * rp_resol; // vtx_x
    kin_out[2] += CLHEP::RandGauss::shoot( rnd_ ) * rp_resol; // vtx_y

    // add track
    out_tracks.emplace_back( rp.detid, kin_out[0], rp_resol, kin_out[2], rp_resol );
  }
}

void
CTPPSScoringPlaneSimulation::fillDescriptions( edm::ConfigurationDescriptions& descriptions )
{
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault( desc );
}

// define this as a plug-in
DEFINE_FWK_MODULE( CTPPSScoringPlaneSimulation );

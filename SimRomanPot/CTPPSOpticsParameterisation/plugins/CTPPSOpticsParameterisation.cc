// -*- C++ -*-
//
// Package:    SimRomanPot/CTPPSOpticsParameterisation
// Class:      CTPPSOpticsParameterisation
// 
/**\class CTPPSOpticsParameterisation CTPPSOpticsParameterisation.cc SimRomanPot/CTPPSOpticsParameterisation/plugins/CTPPSOpticsParameterisation.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Laurent Forthomme
//         Created:  Wed, 24 May 2017 07:40:20 GMT
//
//


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
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

#include "SimDataFormats/CTPPS/interface/LHCOpticsApproximator.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "CLHEP/Random/RandGauss.h"

#include <unordered_map>

class CTPPSOpticsParameterisation : public edm::stream::EDProducer<> {
  public:
    explicit CTPPSOpticsParameterisation( const edm::ParameterSet& );
    ~CTPPSOpticsParameterisation();

    static void fillDescriptions( edm::ConfigurationDescriptions& descriptions );

  private:
    struct CTPPSPotInfo {
      CTPPSPotInfo() : detid( 0 ), resolution( 0. ), approximator( 0 ) {}
      CTPPSPotInfo( const TotemRPDetId& det_id, double resol, LHCOpticsApproximator* approx ) : detid( det_id ), resolution( resol ), approximator( approx ) {}

      TotemRPDetId detid;
      double resolution;
      LHCOpticsApproximator* approximator;
    };

    virtual void produce( edm::Event&, const edm::EventSetup& ) override;
    void transportProtonTrack( const HepMC::GenParticle*, std::vector<CTPPSLocalTrackLite>& );

    edm::EDGetTokenT<edm::HepMCProduct> protonsToken_;

    edm::ParameterSet beamConditions_;
    double sqrtS_;
    double halfCrossingAngleSector45_, halfCrossingAngleSector56_;
    double yOffsetSector45_, yOffsetSector56_;

    bool simulateDetectorsResolution_;

    bool checkApertures_;
    bool invertBeamCoordinatesSystem_;

    edm::FileInPath opticsFileBeam1_, opticsFileBeam2_;
    std::vector<edm::ParameterSet> detectorPackages_;

    std::vector<CTPPSPotInfo> pots_;

    CLHEP::HepRandomEngine* rnd_;
};

CTPPSOpticsParameterisation::CTPPSOpticsParameterisation( const edm::ParameterSet& iConfig ) :
  protonsToken_( consumes<edm::HepMCProduct>( iConfig.getParameter<edm::InputTag>( "beamParticlesTag" ) ) ),
  beamConditions_             ( iConfig.getParameter<edm::ParameterSet>( "beamConditions" ) ),
  sqrtS_                      ( beamConditions_.getParameter<double>( "sqrtS" ) ),
  halfCrossingAngleSector45_  ( beamConditions_.getParameter<double>( "halfCrossingAngleSector45" ) ),
  halfCrossingAngleSector56_  ( beamConditions_.getParameter<double>( "halfCrossingAngleSector56" ) ),
  yOffsetSector45_            ( beamConditions_.getParameter<double>( "yOffsetSector45" ) ),
  yOffsetSector56_            ( beamConditions_.getParameter<double>( "yOffsetSector56" ) ),
  simulateDetectorsResolution_( iConfig.getParameter<bool>( "simulateDetectorsResolution" ) ),
  checkApertures_             ( iConfig.getParameter<bool>( "checkApertures" ) ),
  invertBeamCoordinatesSystem_( iConfig.getParameter<bool>( "invertBeamCoordinatesSystem" ) ),
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
    TotemRPDetId detid( raw_detid );

    if ( detid.arm()==0 ) // sector 45 -- beam 2
      pots_.emplace_back( detid, det_resol, dynamic_cast<LHCOpticsApproximator*>( f_in_optics_beam2->Get( interp_name.c_str() ) ) );
    if ( detid.arm()==1 ) // sector 56 -- beam 1
      pots_.emplace_back( detid, det_resol, dynamic_cast<LHCOpticsApproximator*>( f_in_optics_beam1->Get( interp_name.c_str() ) ) );
  }
}

CTPPSOpticsParameterisation::~CTPPSOpticsParameterisation()
{}

void
CTPPSOpticsParameterisation::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
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
CTPPSOpticsParameterisation::transportProtonTrack( const HepMC::GenParticle* in_trk, std::vector<CTPPSLocalTrackLite>& out_tracks )
{
  /// implemented according to LHCOpticsApproximator::Transport_m_GeV
  /// xi is positive for diffractive protons, thus proton momentum p = (1-xi) * p_nom
  /// horizontal component of proton momentum: p_x = th_x * (1-xi) * p_nom

  // transport the proton into each pot
  for ( const auto& rp : pots_ ) {
    const HepMC::FourVector mom = in_trk->momentum();

    // first check the side
    if ( rp.detid.arm()==0 && mom.z()<0. ) continue;
    if ( rp.detid.arm()==1 && mom.z()>0. ) continue;

    const HepMC::GenVertex* vtx = in_trk->production_vertex();
    // convert physics kinematics to the LHC reference frame
    double th_x = atan2( mom.x(), mom.z() ), th_y = atan2( mom.y(), mom.z() );
    if ( mom.z()<0. ) { th_x = M_PI-th_x; th_y = M_PI-th_y; }
    double vtx_x = vtx->position().x(), vtx_y = vtx->position().y();
    if ( rp.detid.arm()==0 ) {
      th_x += halfCrossingAngleSector45_;
      vtx_y += yOffsetSector45_;
    }

    if ( rp.detid.arm()==1 ) {
      th_x += halfCrossingAngleSector56_;
      vtx_y += yOffsetSector56_;
    }

    const double xi = 1.-mom.e()/( sqrtS_*0.5 );

    // transport proton to its corresponding RP
    double kin_in[5] = { vtx_x, th_x * ( 1.-xi ), vtx_y, th_y * ( 1.-xi ), -xi };
    double kin_out[5];

    bool proton_transported = rp.approximator->Transport( kin_in, kin_out, checkApertures_, invertBeamCoordinatesSystem_ );

    // stop if proton not transportable
    if ( !proton_transported ) return;

    const double rp_resol = ( simulateDetectorsResolution_ ) ? rp.resolution : 0.;

    // simulate detector resolution
    kin_out[0] += CLHEP::RandGauss::shoot( rnd_ ) * rp_resol; // vtx_x
    kin_out[2] += CLHEP::RandGauss::shoot( rnd_ ) * rp_resol; // vtx_y

    // add track
    out_tracks.emplace_back( rp.detid, kin_out[0], rp_resol, kin_out[2], rp_resol );
  }
}

void
CTPPSOpticsParameterisation::fillDescriptions( edm::ConfigurationDescriptions& descriptions )
{
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault( desc );
}

// define this as a plug-in
DEFINE_FWK_MODULE( CTPPSOpticsParameterisation );

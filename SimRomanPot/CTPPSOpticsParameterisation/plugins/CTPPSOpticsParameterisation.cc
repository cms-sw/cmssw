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

#include "SimDataFormats/CTPPS/interface/CTPPSSimProtonTrack.h"
#include "SimDataFormats/CTPPS/interface/CTPPSSimHit.h"
#include "SimDataFormats/CTPPS/interface/LHCOpticsApproximator.h"
//#include "SimDataFormats/CTPPS/interface/LHCApertureApproximator.h"

#include "CLHEP/Random/RandGauss.h"

class CTPPSOpticsParameterisation : public edm::stream::EDProducer<> {
  public:
    explicit CTPPSOpticsParameterisation( const edm::ParameterSet& );
    ~CTPPSOpticsParameterisation();

    static void fillDescriptions( edm::ConfigurationDescriptions& descriptions );

  private:
    virtual void beginStream( edm::StreamID ) override;
    virtual void produce( edm::Event&, const edm::EventSetup& ) override;
    virtual void endStream() override;

    void transportProtonTrack( const CTPPSSimProtonTrack&, std::vector<CTPPSSimHit>& );

    edm::EDGetTokenT< edm::View<CTPPSSimProtonTrack> > tracksToken_;

    edm::ParameterSet beamConditions_;
    double halfCrossingAngleSector45_, halfCrossingAngleSector56_;
    double yOffsetSector45_, yOffsetSector56_;

    bool simulateDetectorsResolution_;

    bool checkApertures_;
    bool invertBeamCoordinatesSystem_;

    edm::FileInPath opticsFileBeam1_, opticsFileBeam2_;
    std::vector<edm::ParameterSet> detectorPackages_;

    std::map<unsigned int,LHCOpticsApproximator*> optics_;

    CLHEP::HepRandomEngine* rnd_;
};

CTPPSOpticsParameterisation::CTPPSOpticsParameterisation( const edm::ParameterSet& iConfig ) :
  tracksToken_( consumes< edm::View<CTPPSSimProtonTrack> >( iConfig.getParameter<edm::InputTag>( "beamParticlesTag" ) ) ),
  beamConditions_             ( iConfig.getParameter<edm::ParameterSet>( "beamConditions" ) ),
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
  produces< std::vector<CTPPSSimHit> >();

  auto f_in_optics_beam1 = std::make_unique<TFile>( opticsFileBeam1_.fullPath().c_str() ),
       f_in_optics_beam2 = std::make_unique<TFile>( opticsFileBeam2_.fullPath().c_str() );

  // load optics and interpolators
  for ( const auto& rp : detectorPackages_ ) {
    const std::string interp_name = rp.getParameter<std::string>( "interpolatorName" );
    const unsigned int raw_detid = rp.getParameter<unsigned int>( "potId" );
    TotemRPDetId detid( TotemRPDetId::decToRawId( raw_detid*10 ) ); //FIXME

    if ( detid.arm()==0 ) {
      optics_.insert( std::make_pair( raw_detid, dynamic_cast<LHCOpticsApproximator*>( f_in_optics_beam2->Get( interp_name.c_str() ) ) ) );
    }
    if ( detid.arm()==1 ) {
      optics_.insert( std::make_pair( raw_detid, dynamic_cast<LHCOpticsApproximator*>( f_in_optics_beam1->Get( interp_name.c_str() ) ) ) );
    }
  }
}


CTPPSOpticsParameterisation::~CTPPSOpticsParameterisation()
{}


// ------------ method called to produce the data  ------------
void
CTPPSOpticsParameterisation::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  std::unique_ptr< std::vector<CTPPSSimHit> > pOut( new std::vector<CTPPSSimHit> );

  if ( simulateDetectorsResolution_ ) {
    edm::Service<edm::RandomNumberGenerator> rng;
    rnd_ = &( rng->getEngine( iEvent.streamID() ) );
  }

  edm::Handle< edm::View<CTPPSSimProtonTrack> > tracks;
  iEvent.getByToken( tracksToken_, tracks );

  // run reconstruction
  for ( const auto& trk : *tracks ) {
    std::vector<CTPPSSimHit> hits;
    transportProtonTrack( trk, hits );
    //FIXME add an association map proton track <-> sim hits
    for ( const auto& hit : hits ) {
      pOut->push_back( hit );
    }
  }

  iEvent.put( std::move( pOut ) );
}

//----------------------------------------------------------------------------------------------------

void
CTPPSOpticsParameterisation::transportProtonTrack( const CTPPSSimProtonTrack& in_trk, std::vector<CTPPSSimHit>& out_hits )
{
  /// implemented according to LHCOpticsApproximator::Transport_m_GeV
  /// xi is positive for diffractive protons, thus proton momentum p = (1 - xi) * p_nom
  /// horizontal component of proton momentum: p_x = th_x * (1 - xi) * p_nom

  // transport the proton into each pot
  for ( const auto& rp : detectorPackages_ ) {
    const unsigned int raw_detid = rp.getParameter<unsigned int>( "potId" );
    const TotemRPDetId detid( TotemRPDetId::decToRawId( raw_detid*10 ) ); //FIXME workaround for strips in 2016
    const double det_resol = rp.getParameter<double>( "resolution" );

    // convert physics kinematics to the LHC reference frame
    double th_x = in_trk.direction().x();
    double vtx_y = in_trk.vertex().y();
    if ( detid.arm()==0 ) {
      th_x += halfCrossingAngleSector45_;
      vtx_y += yOffsetSector45_;
    }

    if ( detid.arm()==1 ) {
      th_x += halfCrossingAngleSector56_;
      vtx_y += yOffsetSector56_;
    }

    // transport proton to its corresponding RP
    double kin_in[5] = { in_trk.vertex().x(), th_x * ( 1.-in_trk.xi() ), vtx_y, in_trk.vertex().y() * ( 1.-in_trk.xi() ), -in_trk.xi() };
    double kin_out[5];

    bool proton_transported = optics_[raw_detid]->Transport( kin_in, kin_out, checkApertures_, invertBeamCoordinatesSystem_ );

    // stop if proton not transportable
    if ( !proton_transported ) return;

    // simulate detector resolution
    if ( simulateDetectorsResolution_ ) {
      kin_out[0] += CLHEP::RandGauss::shoot( rnd_ ) * det_resol;  
      kin_out[2] += CLHEP::RandGauss::shoot( rnd_ ) * det_resol;  
    }

    // add track
    out_hits.emplace_back( detid, Local2DPoint( kin_out[0], kin_out[2] ), Local2DPoint( 12.e-6, 12.e-6 ) );
  }
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
CTPPSOpticsParameterisation::beginStream( edm::StreamID )
{}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
CTPPSOpticsParameterisation::endStream()
{}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
CTPPSOpticsParameterisation::fillDescriptions( edm::ConfigurationDescriptions& descriptions )
{
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault( desc );
}

// define this as a plug-in
DEFINE_FWK_MODULE( CTPPSOpticsParameterisation );

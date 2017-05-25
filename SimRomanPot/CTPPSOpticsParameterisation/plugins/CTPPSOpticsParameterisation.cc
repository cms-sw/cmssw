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

#include "DataFormats/Common/interface/View.h"

#include "SimDataFormats/CTPPS/interface/CTPPSSimProtonTrack.h"
#include "SimDataFormats/CTPPS/interface/CTPPSSimHit.h"
#include "SimDataFormats/CTPPS/interface/LHCOpticsApproximator.h"
#include "SimDataFormats/CTPPS/interface/LHCApertureApproximator.h"

#include "SimRomanPot/CTPPSOpticsParameterisation/interface/ProtonReconstructionAlgorithm.h"

class CTPPSOpticsParameterisation : public edm::stream::EDProducer<> {
  public:
    explicit CTPPSOpticsParameterisation( const edm::ParameterSet& );
    ~CTPPSOpticsParameterisation();

    static void fillDescriptions( edm::ConfigurationDescriptions& descriptions );

  private:
    virtual void beginStream( edm::StreamID ) override;
    virtual void produce( edm::Event&, const edm::EventSetup& ) override;
    virtual void endStream() override;

    //virtual void beginRun( const edm::Run&, const edm::EventSetup& ) override;
    //virtual void endRun( const edm::Run&, const edm::EventSetup& ) override;
    //virtual void beginLuminosityBlock( const edm::LuminosityBlock&, const edm::EventSetup& ) override;
    //virtual void endLuminosityBlock( const edm::LuminosityBlock&, const edm::EventSetup& ) override;

    //void BuildTrackCollection( LHCSector, double, double, double, double, double, const map<unsigned int, LHCOpticsApproximator*>&, TrackDataCollection& );
    void transportProtonTrack( const CTPPSSimProtonTrack&, std::vector<CTPPSSimHit>& );

    edm::EDGetTokenT< edm::View<CTPPSSimProtonTrack> > tracksToken_;

    edm::ParameterSet beamConditions_;
    //bool simulateDetectorsResolution_;

    edm::FileInPath opticsFileBeam1_, opticsFileBeam2_;
    std::vector<edm::ParameterSet> detectorPackages_;

    std::unique_ptr<ProtonReconstructionAlgorithm> prAlgo45_;
    std::unique_ptr<ProtonReconstructionAlgorithm> prAlgo56_;

    std::map<unsigned int,LHCOpticsApproximator*> optics_;
};

CTPPSOpticsParameterisation::CTPPSOpticsParameterisation( const edm::ParameterSet& iConfig ) :
  tracksToken_( consumes< edm::View<CTPPSSimProtonTrack> >( iConfig.getParameter<edm::InputTag>( "beamParticlesTag" ) ) ),
  beamConditions_( iConfig.getParameter<edm::ParameterSet>( "beamConditions" ) ),
  //simulateDetectorsResolution_( iConfig.getParameter<bool>( "simulateDetectorsResolution" ) ),
  opticsFileBeam1_( iConfig.getParameter<edm::FileInPath>( "opticsFileBeam1" ) ),
  opticsFileBeam2_( iConfig.getParameter<edm::FileInPath>( "opticsFileBeam2" ) ),
  detectorPackages_( iConfig.getParameter< std::vector<edm::ParameterSet> >( "detectorPackages" ) ),
  // reconstruction algorithms
  prAlgo45_( std::make_unique<ProtonReconstructionAlgorithm>( beamConditions_, opticsFileBeam2_.fullPath() ) ),
  prAlgo56_( std::make_unique<ProtonReconstructionAlgorithm>( beamConditions_, opticsFileBeam1_.fullPath() ) )
{
  produces< std::vector<CTPPSSimHit> >();

  // load optics
  TFile *f_in_optics_beam1 = TFile::Open( opticsFileBeam1_.fullPath().c_str() );
  optics_[102] = (LHCOpticsApproximator *) f_in_optics_beam1->Get("ip5_to_station_150_h_1_lhcb1");
  optics_[103] = (LHCOpticsApproximator *) f_in_optics_beam1->Get("ip5_to_station_150_h_2_lhcb1");
  TFile *f_in_optics_beam2 = TFile::Open( opticsFileBeam2_.fullPath().c_str() );
  optics_[2] = (LHCOpticsApproximator *) f_in_optics_beam2->Get("ip5_to_station_150_h_1_lhcb2");
  optics_[3] = (LHCOpticsApproximator *) f_in_optics_beam2->Get("ip5_to_station_150_h_2_lhcb2");
}


CTPPSOpticsParameterisation::~CTPPSOpticsParameterisation()
{}


// ------------ method called to produce the data  ------------
void
CTPPSOpticsParameterisation::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  std::unique_ptr< std::vector<CTPPSSimHit> > pOut( new std::vector<CTPPSSimHit> );

  edm::Handle< edm::View<CTPPSSimProtonTrack> > tracks;
  iEvent.getByToken( tracksToken_, tracks );

  // run reconstruction
  for ( const auto& trk : *tracks ) {
    std::vector<CTPPSSimHit> hits;
    transportProtonTrack( trk, hits );
    //FIXME add an association map proton track <-> sim hits
    std::cout << "---> "  << hits.size() << " hits" << std::endl;
  }
  /*ProtonData proton_45 = prAlgo45_->Reconstruct(sector45, tracks_45);
  ProtonData proton_56 = prAlgo56_->Reconstruct(sector45, tracks_56);

  if ( proton_45.isValid() ) {
    const double de_vtx_x = proton_45.vtx_x - vtx_x;
    const double de_vtx_y = proton_45.vtx_y - vtx_y;
    const double de_th_x = proton_45.th_x - th_x_45_phys;
    const double de_th_y = proton_45.th_y - th_y_45_phys;
    const double de_xi = proton_45.xi - xi_45;

    h_de_vtx_x_45->Fill(de_vtx_x);
    h_de_vtx_y_45->Fill(de_vtx_y);
    h_de_th_x_45->Fill(de_th_x);
    h_de_th_y_45->Fill(de_th_y);
    h_de_xi_45->Fill(de_xi);

    h2_de_vtx_x_vs_de_xi_45->Fill(de_xi, de_vtx_x);
    h2_de_vtx_y_vs_de_xi_45->Fill(de_xi, de_vtx_y);
    h2_de_th_x_vs_de_xi_45->Fill(de_xi, de_th_x);
    h2_de_th_y_vs_de_xi_45->Fill(de_xi, de_th_y);
    h2_de_vtx_y_vs_de_th_y_45->Fill(de_th_y, de_vtx_y);

    p_de_vtx_x_vs_xi_45->Fill(xi_45, de_vtx_x);
    p_de_vtx_y_vs_xi_45->Fill(xi_45, de_vtx_y);
    p_de_th_x_vs_xi_45->Fill(xi_45, de_th_x);
    p_de_th_y_vs_xi_45->Fill(xi_45, de_th_y);
    p_de_xi_vs_xi_45->Fill(xi_45, de_xi);
  }

  if ( proton_56.isValid() ) {
    const double de_vtx_x = proton_56.vtx_x - vtx_x;
    const double de_vtx_y = proton_56.vtx_y - vtx_y;
    const double de_th_x = proton_56.th_x - th_x_56_phys;
    const double de_th_y = proton_56.th_y - th_y_56_phys;
    const double de_xi = proton_56.xi - xi_56;

    h_de_vtx_x_56->Fill(de_vtx_x);
    h_de_vtx_y_56->Fill(de_vtx_y);
    h_de_th_x_56->Fill(de_th_x);
    h_de_th_y_56->Fill(de_th_y);
    h_de_xi_56->Fill(de_xi);

    h2_de_vtx_x_vs_de_xi_56->Fill(de_xi, de_vtx_x);
    h2_de_vtx_y_vs_de_xi_56->Fill(de_xi, de_vtx_y);
    h2_de_th_x_vs_de_xi_56->Fill(de_xi, de_th_x);
    h2_de_th_y_vs_de_xi_56->Fill(de_xi, de_th_y);
    h2_de_vtx_y_vs_de_th_y_56->Fill(de_th_y, de_vtx_y);

    p_de_vtx_x_vs_xi_56->Fill(xi_56, de_vtx_x);
    p_de_vtx_y_vs_xi_56->Fill(xi_56, de_vtx_y);
    p_de_th_x_vs_xi_56->Fill(xi_56, de_th_x);
    p_de_th_y_vs_xi_56->Fill(xi_56, de_th_y);
    p_de_xi_vs_xi_56->Fill(xi_56, de_xi);
  }*/

  iEvent.put( std::move( pOut ) );
}

//----------------------------------------------------------------------------------------------------

/// implemented according to LHCOpticsApproximator::Transport_m_GeV
/// xi is positive for diffractive protons, thus proton momentum p = (1 - xi) * p_nom
/// horizontal component of proton momentum: p_x = th_x * (1 - xi) * p_nom

void
CTPPSOpticsParameterisation::transportProtonTrack( const CTPPSSimProtonTrack& in_trk, std::vector<CTPPSSimHit>& out_hits )
{
  // settings
  const bool check_apertures = true;
  const bool invert_beam_coord_sytems = true;

  // transport the proton into each pot
  for ( const auto& rp : detectorPackages_ ) {
    const unsigned int raw_detid = rp.getParameter<unsigned int>( "rpId" );
    const TotemRPDetId detid( TotemRPDetId::decToRawId( raw_detid*10 ) ); //FIXME workaround for strips in 2016

    // convert physics kinematics to the LHC reference frame
    double th_x = in_trk.direction().x();
    double vtx_y = in_trk.vertex().y();
    if ( detid.arm()==0 ) {
      th_x += beamConditions_.getParameter<double>( "halfCrossingAngleSector45" );
      vtx_y += beamConditions_.getParameter<double>( "yOffsetSector45" );
    }

    if ( detid.arm()==1 ) {
      th_x += beamConditions_.getParameter<double>( "halfCrossingAngleSector56" );
      vtx_y += beamConditions_.getParameter<double>( "yOffsetSector56" );
    }

    // transport proton to its corresponding RP
    double kin_in[5] = { in_trk.vertex().x(), th_x * ( 1.-in_trk.xi() ), vtx_y, in_trk.vertex().y() * ( 1.-in_trk.xi() ), -in_trk.xi() };
    double kin_out[5];

    bool proton_transported = optics_[raw_detid]->Transport( kin_in, kin_out, check_apertures, invert_beam_coord_sytems );

    // stop if proton not transportable
    if ( !proton_transported ) return;

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

// ------------ method called when starting to processes a run  ------------
/*
void
CTPPSOpticsParameterisation::beginRun( const edm::Run&, const edm::EventSetup& )
{}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
CTPPSOpticsParameterisation::endRun( const edm::Run&, const edm::EventSetup& )
{}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
CTPPSOpticsParameterisation::beginLuminosityBlock( const edm::LuminosityBlock&, const edm::EventSetup& )
{}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
CTPPSOpticsParameterisation::endLuminosityBlock( const edm::LuminosityBlock&, const edm::EventSetup& )
{}
*/
 
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

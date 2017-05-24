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

    edm::EDGetTokenT< edm::View<CTPPSSimProtonTrack> > tracks45Token_, tracks56Token_;

    edm::ParameterSet beamConditions_;
    //bool simulateDetectorsResolution_;

    edm::FileInPath opticsFileBeam1_, opticsFileBeam2_;
    std::vector<edm::ParameterSet> detectorsList_;

    std::unique_ptr<ProtonReconstructionAlgorithm> prAlgo45_;
    std::unique_ptr<ProtonReconstructionAlgorithm> prAlgo56_;
};

CTPPSOpticsParameterisation::CTPPSOpticsParameterisation( const edm::ParameterSet& iConfig ) :
  tracks45Token_( consumes< edm::View<CTPPSSimProtonTrack> >( iConfig.getParameter<edm::InputTag>( "beam2ParticlesTag" ) ) ),
  tracks56Token_( consumes< edm::View<CTPPSSimProtonTrack> >( iConfig.getParameter<edm::InputTag>( "beam1ParticlesTag" ) ) ),
  beamConditions_( iConfig.getParameter<edm::ParameterSet>( "beamConditions" ) ),
  //simulateDetectorsResolution_( iConfig.getParameter<bool>( "simulateDetectorsResolution" ) ),
  opticsFileBeam1_( iConfig.getParameter<edm::FileInPath>( "opticsFileBeam1" ) ),
  opticsFileBeam2_( iConfig.getParameter<edm::FileInPath>( "opticsFileBeam2" ) ),
  detectorsList_( iConfig.getParameter< std::vector<edm::ParameterSet> >( "detectorsList" ) )
{
  produces< std::vector<CTPPSSimHit> >();

  // load optics
  std::map<unsigned int, LHCOpticsApproximator*> optics_45, optics_56; // map: RP id --> optics

  TFile *f_in_optics_beam1 = TFile::Open( opticsFileBeam1_.fullPath().c_str() );
  optics_56[102] = (LHCOpticsApproximator *) f_in_optics_beam1->Get("ip5_to_station_150_h_1_lhcb1");
  optics_56[103] = (LHCOpticsApproximator *) f_in_optics_beam1->Get("ip5_to_station_150_h_2_lhcb1");

  TFile *f_in_optics_beam2 = TFile::Open( opticsFileBeam2_.fullPath().c_str() );
  optics_45[2] = (LHCOpticsApproximator *) f_in_optics_beam2->Get("ip5_to_station_150_h_1_lhcb2");
  optics_45[3] = (LHCOpticsApproximator *) f_in_optics_beam2->Get("ip5_to_station_150_h_2_lhcb2");

  // initialise proton reconstruction
  ProtonReconstruction protonReconstruction;
  if ( prAlgo45_->Init( opticsFileBeam2_.fullPath() )!=0 )
    throw cms::Exception("CTPPSOpticsParameterisation") << "Failed to initialise the reconstruction algorithm for beam 1";
  if ( prAlgo56_->Init( opticsFileBeam1_.fullPath() )!=0 )
    throw cms::Exception("CTPPSOpticsParameterisation") << "Failed to initialise the reconstruction algorithm for beam 2";

}


CTPPSOpticsParameterisation::~CTPPSOpticsParameterisation()
{}


// ------------ method called to produce the data  ------------
void
CTPPSOpticsParameterisation::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  std::unique_ptr< std::vector<CTPPSSimHit> > pOut( new std::vector<CTPPSSimHit> );

  //Read 'ExampleData' from the Event
  edm::Handle< edm::View<CTPPSSimProtonTrack> > tracks_45, tracks_56;
  iEvent.getByToken( tracks45Token_, tracks_45 );
  iEvent.getByToken( tracks56Token_, tracks_56 );

  /*// run reconstruction
  ProtonData proton_45 = prAlgo45_->Reconstruct(sector45, tracks_45);
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

/*void
CTPPSOpticsParameterisation::BuildTrackCollection( LHCSector sector, double vtx_x, double vtx_y, double th_x, double th_y, double xi, const map<unsigned int, LHCOpticsApproximator*> &optics, TrackDataCollection &tracks )
{
  // settings
  const bool check_appertures = true;
  const bool invert_beam_coord_sytems = true;

  // start with no tracks
  tracks.clear();

  // convert physics kinematics to the LHC reference frame
  if (sector == sector45) {
    th_x += beamConditions.half_crossing_angle_45;
    vtx_y += beamConditions.vtx0_y_45;
  }

  if (sector == sector56) {
    th_x += beamConditions.half_crossing_angle_56;
    vtx_y += beamConditions.vtx0_y_56;
  }

  // transport proton to each RP
  for (const auto it : optics) {
    double kin_in[5];
    kin_in[0] = vtx_x;
    kin_in[1] = th_x * (1. - xi);
    kin_in[2] = vtx_y;
    kin_in[3] = th_y * (1. - xi);
    kin_in[4] = - xi;

    double kin_out[5];
    bool proton_trasported = it.second->Transport(kin_in, kin_out, check_appertures, invert_beam_coord_sytems);

    // stop if proton not transportable
    if (!proton_trasported) continue;

    // add track
    TrackData td;
    td.valid = true;
    td.x = kin_out[0];
    td.y = kin_out[2];
    td.x_unc = 12E-6;
    td.y_unc = 12E-6;

    tracks[it.first] = td;
  }
}*/

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

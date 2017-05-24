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

#include "SimDataFormats/CTPPS/interface/CTPPSSimHit.h"
#include "SimRomanPot/CTPPSOpticsParameterisation/ProtonReconstructionAlgorithm.h"

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

    void BuildTrackCollection( LHCSector, double, double, double, double, double, const map<unsigned int, LHCOpticsApproximator*>&, TrackDataCollection& );

    edm::ParameterSet beamConditions_;
    bool simulateVertexX_, simulateVertexY_;
    bool simulateScatteringAngleX_, simulateScatteringAngleY_;
    bool simulateBeamDivergence_;
    bool simulateXi_;
    bool simulateDetectorsResolution_;

    edm::FileInPath opticsFileBeam1_, opticsFileBeam2_;
    std::vector<edm::ParameterSet> detectorsList_;

    std::unique_ptr<ProtonReconstructionAlgorithm> prAlgorithm_;
};

CTPPSOpticsParameterisation::CTPPSOpticsParameterisation( const edm::ParameterSet& iConfig ) :
  beamConditions_( iConfig.getParameter<edm::ParameterSet>( "beamConditions" ) ),
  simulateVertexX_( iConfig.getParameter<bool>( "simulateVertexX" ) ), simulateVertexY_( iConfig.getParameter<bool>( "simulateVertexY" ) ),
  simulateScatteringAngleX_( iConfig.getParameter<bool>( "simulateScatteringAngleX" ) ), simulateScatteringAngleY_( iConfig.getParameter<bool>( "simulateScatteringAngleY" ) ),
  simulateBeamDivergence_( iConfig.getParameter<bool>( "simulateBeamDivergence" ) ),
  simulateXi_( iConfig.getParameter<bool>( "simulateXi" ) ),
  simulateDetectorsResolution_( iConfig.getParameter<bool>( "simulateDetectorsResolution" ) ),
  opticsFileBeam1_( iConfig.getParameter<edm::FileInPath>( "opticsFileBeam1" ) ),
  opticsFileBeam2_( iConfig.getParameter<edm::FileInPath>( "opticsFileBeam2" ) ),
  detectorsList_( iConfig.getParameter< std::vector<edm::ParameterSet> >( "detectorsList" ) )
{
  produces< std::vector<CTPPSSimHit> >();

  // load optics
  std::map<unsigned int, LHCOpticsApproximator*> optics_45, optics_56; // map: RP id --> optics

  TFile *f_in_optics_beam1 = TFile::Open(file_optics_beam1.c_str());
  optics_56[102] = (LHCOpticsApproximator *) f_in_optics_beam1->Get("ip5_to_station_150_h_1_lhcb1");
  optics_56[103] = (LHCOpticsApproximator *) f_in_optics_beam1->Get("ip5_to_station_150_h_2_lhcb1");

  TFile *f_in_optics_beam2 = TFile::Open(file_optics_beam2.c_str());
  optics_45[2] = (LHCOpticsApproximator *) f_in_optics_beam2->Get("ip5_to_station_150_h_1_lhcb2");
  optics_45[3] = (LHCOpticsApproximator *) f_in_optics_beam2->Get("ip5_to_station_150_h_2_lhcb2");

  // initialise proton reconstruction
  ProtonReconstruction protonReconstruction;
  if ( prAlgorithm_->Init( opticsFileBeam1_.fullPath(), opticsFileBeam2_.fullPath() )!=0 )
    throw cms::Exception("CTPPSOpticsParameterisation") << "Failed to initialise the reconstruction algorithm";

}


CTPPSOpticsParameterisation::~CTPPSOpticsParameterisation()
{}


// ------------ method called to produce the data  ------------
void
CTPPSOpticsParameterisation::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
/* This is an event example
  //Read 'ExampleData' from the Event
  Handle<ExampleData> pIn;
  iEvent.getByLabel("example",pIn);

*/
  std::unique_ptr< std::vector<CTPPSSimHit> > pOut( new std::vector<CTPPSSimHit> );

/* this is an EventSetup example
  //Read SetupData from the SetupRecord in the EventSetup
  ESHandle<SetupData> pSetup;
  iSetup.get<SetupRecord>().get(pSetup);
*/
  // run reconstruction
  ProtonData proton_45 = prAlgorithm_->Reconstruct(sector45, tracks_45);
  ProtonData proton_56 = prAlgorithm_->Reconstruct(sector45, tracks_56);

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
  }

  iEvent.put( std::move( pOut ) );
}

//----------------------------------------------------------------------------------------------------

/// implemented according to LHCOpticsApproximator::Transport_m_GeV
/// xi is positive for diffractive protons, thus proton momentum p = (1 - xi) * p_nom
/// horizontal component of proton momentum: p_x = th_x * (1 - xi) * p_nom

void
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
}

void
CTPPSOpticsParameterisation::generateEvent()
{
  // generate vertex
  double vtx_x = 0., vtx_y = 0.;

  if ( simulateVertexX_ ) vtx_x += gRandom->Gaus() * beamConditions.si_vtx;
  if ( simulateVertexY_ ) vtx_y += gRandom->Gaus() * beamConditions.si_vtx;

  // generate scattering angles (physics)
  double th_x_45_phys = 0., th_y_45_phys = 0.;
  double th_x_56_phys = 0., th_y_56_phys = 0.;

  if ( simulateScatteringAngleX_ ) {
    th_x_45_phys += gRandom->Gaus() * si_th_phys;
    th_x_56_phys += gRandom->Gaus() * si_th_phys;
  }

  if ( simulateScatteringAngleY_ ) {
    th_y_45_phys += gRandom->Gaus() * si_th_phys;
    th_y_56_phys += gRandom->Gaus() * si_th_phys;
  }

  // generate beam divergence, calculate complete angle
  double th_x_45 = th_x_45_phys, th_y_45 = th_y_45_phys;
  double th_x_56 = th_x_56_phys, th_y_56 = th_y_56_phys;

  if ( simulateBeamDivergence_ ) {
    th_x_45 += gRandom->Gaus() * beamConditions.si_beam_div;
    th_y_45 += gRandom->Gaus() * beamConditions.si_beam_div;

    th_x_56 += gRandom->Gaus() * beamConditions.si_beam_div;
    th_y_56 += gRandom->Gaus() * beamConditions.si_beam_div;
  }

  // generate xi
  double xi_45 = 0, xi_56 = 0;
  if ( simulateXi_ ) {
    xi_45 = xi_min + gRandom->Rndm() * (xi_max - xi_min);
    xi_56 = xi_min + gRandom->Rndm() * (xi_max - xi_min);
  }

  // proton transport
  TrackDataCollection tracks_45;
  BuildTrackCollection(sector45, vtx_x, vtx_y, th_x_45, th_y_45, xi_45, optics_45, tracks_45);

  TrackDataCollection tracks_56;
  BuildTrackCollection(sector56, vtx_x, vtx_y, th_x_56, th_y_56, xi_56, optics_56, tracks_56);

  /*// simulate detector resolution
  if ( simulateDetectorsResolution_ ) {
    for (auto &it : tracks_45) {
      it.second.x += gRandom->Gaus() * si_det;
      it.second.y += gRandom->Gaus() * si_det;
    }
    for (auto &it : tracks_56) {
      it.second.x += gRandom->Gaus() * si_det;
      it.second.y += gRandom->Gaus() * si_det;
    }
  }*/
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

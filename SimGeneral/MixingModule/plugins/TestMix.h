// -*- C++ -*-
//
// Class:      TestMix
// 
/**\class TestMix

 Description: test of Mixing Module

*/
//
// Original Author:  Ursula Berthon
//         Created:  Fri Sep 23 11:38:38 CEST 2005
//
//

// system include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <vector>
#include <string>

namespace edm
{

  //
  // class declaration
  //

  class TestMix : public edm::one::EDAnalyzer<> {
  public:
    explicit TestMix(const edm::ParameterSet&);
    virtual ~TestMix();

    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
    int level_;
    std::vector<std::string> track_containers_;
    std::vector<std::string> track_containers2_;

    edm::EDGetTokenT<CrossingFrame<PSimHit>> TrackerToken0_;
    edm::EDGetTokenT<CrossingFrame<PSimHit>> TrackerToken1_;
    edm::EDGetTokenT<CrossingFrame<PSimHit>> TrackerToken2_;
    edm::EDGetTokenT<CrossingFrame<PSimHit>> TrackerToken3_;
    edm::EDGetTokenT<CrossingFrame<PSimHit>> TrackerToken4_;

    edm::EDGetTokenT<CrossingFrame<PCaloHit>> CaloToken1_;

    edm::EDGetTokenT<CrossingFrame<SimTrack>> SimTrackToken_;
    edm::EDGetTokenT<CrossingFrame<SimVertex>> SimVertexToken_;
    edm::EDGetTokenT<CrossingFrame<HepMCProduct>> HepMCToken_;

  };
}//edm

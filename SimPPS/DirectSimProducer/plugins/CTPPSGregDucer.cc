/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Greg
 ****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLiteFwd.h" //Fix for not recognising the header

#include "CondFormats/DataRecord/interface/CTPPSInterpolatedOpticsRcd.h"
#include "CondFormats/PPSObjects/interface/LHCInterpolatedOpticalFunctionsSetCollection.h"

#include "CondFormats/PPSObjects/interface/CTPPSBeamParameters.h"
#include "CondFormats/DataRecord/interface/CTPPSBeamParametersRcd.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "CondFormats/DataRecord/interface/PPSDirectSimulationDataRcd.h"
#include "CondFormats/PPSObjects/interface/PPSDirectSimulationData.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"
#include "CondFormats/PPSObjects/interface/PPSPixelTopology.h"
#include "CondFormats/DataRecord/interface/PPSPixelTopologyRcd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include <unordered_map>

#include "TMath.h"
#include "TMatrixD.h"
#include "TVectorD.h"
#include "TF1.h"
#include "TF2.h"
#include "TFile.h"
#include "CLHEP/Random/RandFlat.h"

//----------------------------------------------------------------------------------------------------

class CTPPSGregDucer : public edm::stream::EDProducer<> {
public:
  explicit CTPPSGregDucer(const edm::ParameterSet &);
  ~CTPPSGregDucer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  void applyCut() const;

  edm::EDGetTokenT<CTPPSLocalTrackLiteCollection> tokenTracks_;

};
//----------------------------------------------------------------------------------------------------

CTPPSGregDucer::CTPPSGregDucer(const edm::ParameterSet &ps):
  tokenTracks_(consumes<CTPPSLocalTrackLiteCollection>(ps.getParameter<edm::InputTag>("tagTracks")))

  {}


//----------------------------------------------------------------------------------------------------

void CTPPSGregDucer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tagTracks", edm::InputTag("ctppsLocalTrackLiteProducer"))->setComment("Input tag for CTPPSLocalTrackLiteCollection");
  descriptions.add("ctppsGregDucer", desc);

}

//----------------------------------------------------------------------------------------------------

void CTPPSGregDucer::produce(edm::Event &event, const edm::EventSetup &iSetup) {
  edm::Handle<CTPPSLocalTrackLiteCollection> hTracks;
  event.getByToken(tokenTracks_, hTracks);
  applyCut();
    
}

//----------------------------------------------------------------------------------------------------

void CTPPSGregDucer::applyCut() const {
  // std::cout << "Greg Produced" << std::endl; //It works
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSGregDucer);

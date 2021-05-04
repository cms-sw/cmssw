#ifndef CSCDigiProducer_h
#define CSCDigiProducer_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimMuon/CSCDigitizer/src/CSCDigitizer.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

class CSCStripConditions;

class CSCDigiProducer : public edm::stream::EDProducer<> {
public:
  typedef CSCDigitizer::DigiSimLinks DigiSimLinks;

  explicit CSCDigiProducer(const edm::ParameterSet &ps);
  ~CSCDigiProducer() override;

  /**Produces the EDM products,*/
  void produce(edm::Event &e, const edm::EventSetup &c) override;

private:
  CSCDigitizer theDigitizer;
  CSCStripConditions *theStripConditions;

  edm::EDGetTokenT<CrossingFrame<PSimHit>> cf_token;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> geom_Token;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfield_Token;
  edm::ESGetToken<ParticleDataTable, edm::DefaultRecord> pdt_Token;
};

#endif

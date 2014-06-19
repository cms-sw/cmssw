#ifndef L1TkElectronStubMatchAlgo_HH
#define L1TkElectronStubMatchAlgo_HH

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"
#include "DataFormats/SiPixelDetId/interface/StackedTrackerDetId.h"


namespace L1TkElectronStubMatchAlgo {
  typedef edm::Ref<edmNew::DetSetVector<TTStub<Ref_PixelDigi_ > >, TTStub<Ref_PixelDigi_> > stubRef;
  typedef std::vector< stubRef >  stubRefCollection;

  unsigned int doMatch(l1extra::L1EmParticleCollection::const_iterator egIter, const edm::ParameterSet& conf, 
		       const edm::EventSetup& iSetup, edm::Event & iEvent, std::vector<double>& zvals);
  GlobalPoint calorimeterPosition(double phi, double eta, double e);
  unsigned int getLayerId(StackedTrackerDetId id);
  bool goodTwoPointZ(double innerR, double outerR, double innerZ, double outerZ );
  bool goodTwoPointPhi(double innerR, double outerR, double innerPhi, double outerPhi, double m_strength);
  double getDPhi(GlobalPoint epos, double eet, double r, double phi, double m_strength);
  double getZIntercept(GlobalPoint epos, double r, double z);
  double getPhiMiss(double eet, GlobalPoint spos1, GlobalPoint spos2);
  double getZMiss(GlobalPoint epos, double r1, double r2, double z1, double z2, bool bar);
  double getScaledZInterceptCut(unsigned int layer, double cut, double cfac, double eta);
  double getScaledZMissCut(int layer1, int layer2, double cut, double cfac, double eta);
  bool compareStubLayer(const stubRef& s1, const stubRef& s2);
  bool selectLayers(float eta, int l1, int l2); 
  double getCompatibleZPoint(double r1, double r2, double z1, double z2);
}
#endif

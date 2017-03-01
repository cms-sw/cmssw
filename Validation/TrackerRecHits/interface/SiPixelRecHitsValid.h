#ifndef SiPixelRecHitsValid_h
#define SiPixelRecHitsValid_h

/** \class SiPixelRecHitsValid
 * File: SiPixelRecHitsValid.h
 * \author Jason Shaev, JHU
 * Created: 6/7/06
 */

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include <string>

class DQMStore;
class DetId;
class MonitorElement;
class PSimHit;
class PixelGeomDetUnit;
class SiPixelRecHit;
class TrackerTopology;

class SiPixelRecHitsValid : public DQMEDAnalyzer {

   public:
	//Constructor
	SiPixelRecHitsValid(const edm::ParameterSet& conf);

	//Destructor
	~SiPixelRecHitsValid();

   protected:

	virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
	void beginJob();
	void bookHistograms(DQMStore::IBooker & ibooker,const edm::Run& run, const edm::EventSetup& es);

   private:
	void fillBarrel(const SiPixelRecHit &,const PSimHit &, DetId, const PixelGeomDetUnit *,	
			 const TrackerTopology *tTopo);
	void fillForward(const SiPixelRecHit &, const PSimHit &, DetId, const PixelGeomDetUnit *,
			 const TrackerTopology *tTopo);

	//Clusters BPIX
	MonitorElement* clustYSizeModule[8];
	MonitorElement* clustXSizeLayer[3];
	MonitorElement* clustChargeLayer1Modules[8];
	MonitorElement* clustChargeLayer2Modules[8];
	MonitorElement* clustChargeLayer3Modules[8];

	//Cluster FPIX
	MonitorElement* clustXSizeDisk1Plaquettes[7];
	MonitorElement* clustXSizeDisk2Plaquettes[7];
	MonitorElement* clustYSizeDisk1Plaquettes[7];
	MonitorElement* clustYSizeDisk2Plaquettes[7];
	MonitorElement* clustChargeDisk1Plaquettes[7];
	MonitorElement* clustChargeDisk2Plaquettes[7];

	//RecHits BPIX
	MonitorElement* recHitXResAllB;
	MonitorElement* recHitYResAllB;
	MonitorElement* recHitXFullModules;
	MonitorElement* recHitXHalfModules;
	MonitorElement* recHitYAllModules;
	MonitorElement* recHitXResFlippedLadderLayers[3];
	MonitorElement* recHitXResNonFlippedLadderLayers[3];
	MonitorElement* recHitYResLayer1Modules[8];
	MonitorElement* recHitYResLayer2Modules[8];
	MonitorElement* recHitYResLayer3Modules[8];
	MonitorElement* recHitBunchB;
	MonitorElement* recHitEventB;
	MonitorElement* recHitNsimHitLayer[3];

	//RecHits FPIX
	MonitorElement* recHitXResAllF;
	MonitorElement* recHitYResAllF;
	MonitorElement* recHitXPlaquetteSize1;
	MonitorElement* recHitXPlaquetteSize2;
	MonitorElement* recHitYPlaquetteSize2;
	MonitorElement* recHitYPlaquetteSize3;
	MonitorElement* recHitYPlaquetteSize4;
	MonitorElement* recHitYPlaquetteSize5;
	MonitorElement* recHitXResDisk1Plaquettes[7];
	MonitorElement* recHitXResDisk2Plaquettes[7];
	MonitorElement* recHitYResDisk1Plaquettes[7];
	MonitorElement* recHitYResDisk2Plaquettes[7];
	MonitorElement* recHitBunchF;
	MonitorElement* recHitEventF;
	MonitorElement* recHitNsimHitDisk1;
	MonitorElement* recHitNsimHitDisk2;

	// Pull distributions
	//RecHits BPIX
	MonitorElement* recHitXPullAllB;
	MonitorElement* recHitYPullAllB;

	MonitorElement* recHitXPullFlippedLadderLayers[3];
	MonitorElement* recHitXPullNonFlippedLadderLayers[3];
	MonitorElement* recHitYPullLayer1Modules[8];
	MonitorElement* recHitYPullLayer2Modules[8];
	MonitorElement* recHitYPullLayer3Modules[8];

	//RecHits FPIX
	MonitorElement* recHitXPullAllF;
	MonitorElement* recHitYPullAllF;

	MonitorElement* recHitXPullDisk1Plaquettes[7];
	MonitorElement* recHitXPullDisk2Plaquettes[7];
	MonitorElement* recHitYPullDisk1Plaquettes[7];
	MonitorElement* recHitYPullDisk2Plaquettes[7];

        TrackerHitAssociator::Config trackerHitAssociatorConfig_;
        edm::EDGetTokenT<SiPixelRecHitCollection> siPixelRecHitCollectionToken_;
};

#endif

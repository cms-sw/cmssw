// -*- C++ -*-
//
// Package:    StripClusterMCanalysis
// Class:      StripClusterMCanalysis
// 
/**\class StripClusterMCanalysis StripClusterMCanalysis.cc UserCode/WTFord/detAnalysis/StripClusterMCanalysis/src/StripClusterMCanalysis.cc

 Description:  Analyze siStrip clusters for multiple track crossings

 The original purpose was to explore the separation via dE/dx between clusters
 originating from a single minimum-ionizing charged particle and those from
 overlap of multiple particles.  To this end we create a flat root ntuple 
 providing for each cluster information extracted from associated simTracks and 
 simHits regarding the charge, path length, particle ID, originating physics
 process, etc.

 Implementation:
     <Notes on implementation>
*/
//
//   Original Author:  William T. Ford
//           Created:  Sat Nov 21 18:02:42 MST 2009
// Adapted for CMSSW:  September, 2015
// $Id: StripClusterMCanalysis.cc,v 1.3 2011/06/01 23:22:26 wtford Exp $
//
//

// this class header

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TROOT.h"

#include "TObject.h"
#include "TH1D.h"
#include "TTree.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// Data Formats
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"    // PR#7802

//
//  Particle interaction process codes are found in
// SimG4Core/Physics/src/G4ProcessTypeEnumerator.cc
// See also https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideMCTruth
//

//
// class declaration
//

class StripClusterMCanalysis : public edm::EDAnalyzer {
public:

  explicit StripClusterMCanalysis(const edm::ParameterSet&);

  ~StripClusterMCanalysis();

private:
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  typedef std::pair<unsigned int, unsigned int> simHitCollectionID;
  typedef std::pair<simHitCollectionID, unsigned int> simhitAddr;
  typedef std::map<simHitCollectionID, std::vector<PSimHit> > simhit_collectionMap;
  simhit_collectionMap SimHitCollMap_;

  void makeMap(const edm::Event& theEvent);

      // ----------member data ---------------------------

private:
  edm::ParameterSet conf_;
  edm::InputTag beamSpotLabel_;
  edm::InputTag primaryVertexLabel_;
  edm::InputTag stripClusterSourceLabel_;
  edm::InputTag stripSimLinkLabel_;
  int printOut_;

  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<reco::VertexCollection> primaryVertexToken_;
  edm::EDGetTokenT< edmNew::DetSetVector<SiStripCluster> > clusterToken_;
  edm::EDGetTokenT< edm::DetSetVector<StripDigiSimLink> > stripSimlinkToken_;
  std::vector< edm::EDGetTokenT<CrossingFrame<PSimHit> > > cfTokens_;
  std::vector< edm::EDGetTokenT<std::vector<PSimHit> > > simHitTokens_;

  edm::Handle< edm::DetSetVector<StripDigiSimLink> >  stripdigisimlink;

  int evCnt_;

  TTree* clusTree_;  // tree filled for each cluster
  struct ClusterStruct{
    int subDet;
    float thickness;
    int width;
    int NsimHits;
    int firstProcess;
    int secondProcess;
    int thirdProcess;
    int fourthProcess;
    int firstPID;
    int secondPID;
    int thirdPID;
    int fourthPID;
    int Ntp;
    float firstTkChg;
    float secondTkChg;
    float thirdTkChg;
    float fourthTkChg;
    float charge;
    float firstPmag;
    float secondPmag;
    float thirdPmag;
    float fourthPmag;
    float firstPathLength;
    float secondPathLength;
    float thirdPathLength;
    float fourthPathLength;
    float pathLstraight;
    float allHtPathLength;
    float Eloss;
    int sat;
    int tkFlip;
    int ovlap;
    int layer;
    int stereo;
    } clusNtp_;

  TTree* pvTree_;  // tree filled for each pixel vertex
  struct pvStruct{
    int isValid;
    int isFake;
    int Ntrks;
    int nDoF;
    float chisq;
    float xV;
    float yV;
    float zV;
    float xVsig;
    float yVsig;
    float zVsig;
    } pvNtp_;

  TH1F* hNpv;
  TH2F* hzV_Iev;
  TH2F* hNtrk_zVerrPri;
  TH2F* hNtrk_zVerrSec;
  TH1F* hZvPri;
  TH1F* hZvSec;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
StripClusterMCanalysis::StripClusterMCanalysis(const edm::ParameterSet& iConfig): 
  conf_(iConfig),
  beamSpotLabel_(iConfig.getParameter<edm::InputTag>("beamSpot")),
  primaryVertexLabel_(iConfig.getParameter<edm::InputTag>("primaryVertex")),
  stripClusterSourceLabel_(iConfig.getParameter<edm::InputTag>("stripClusters")),
  stripSimLinkLabel_(iConfig.getParameter<edm::InputTag>("stripSimLinks")),
  printOut_(iConfig.getUntrackedParameter<int>("printOut"))
{
   //now do whatever initialization is needed
  beamSpotToken_ = consumes<reco::BeamSpot>(beamSpotLabel_);
  primaryVertexToken_ = consumes<reco::VertexCollection>(primaryVertexLabel_);
  clusterToken_ = consumes< edmNew::DetSetVector<SiStripCluster> >(stripClusterSourceLabel_);
  stripSimlinkToken_ = consumes< edm::DetSetVector<StripDigiSimLink> >(stripSimLinkLabel_);

  std::vector<std::string> trackerContainers(iConfig.getParameter<std::vector<std::string> >("ROUList"));
  cfTokens_.reserve(trackerContainers.size());
  simHitTokens_.reserve(trackerContainers.size());
  for(auto const& trackerContainer : trackerContainers) {
    cfTokens_.push_back(consumes<CrossingFrame<PSimHit> >(edm::InputTag("mix", trackerContainer)));
    simHitTokens_.push_back(consumes<std::vector<PSimHit> >(edm::InputTag("g4SimHits", trackerContainer)));
  }
}

StripClusterMCanalysis::~StripClusterMCanalysis()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
StripClusterMCanalysis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;

  if (printOut_ > 0) std::cout << std::endl;

  evCnt_++;

  // Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);  // PR#7802
  const TrackerTopology* const tTopo = tTopoHandle.product();

  // Get the beam spot
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(beamSpotToken_, recoBeamSpotHandle);
  const reco::BeamSpot& bs = *recoBeamSpotHandle;      
  reco::Vertex::Point estPriVtx(bs.position());

  // and the primary vertex
  edm::Handle<reco::VertexCollection>  pixelVertexCollectionHandle;
  if (iEvent.getByToken(primaryVertexToken_, pixelVertexCollectionHandle)) {
    const reco::VertexCollection pixelVertexColl = *(pixelVertexCollectionHandle.product());
    hNpv->Fill(int(pixelVertexColl.size()));
    float zv = 0, zverr = 0;
    reco::VertexCollection::const_iterator vi = pixelVertexColl.begin();
    if (vi != pixelVertexColl.end() && vi->isValid()) {
      estPriVtx = vi->position();
      zv = vi->z();
      zverr = vi->zError();
    }
    hZvPri->Fill(zv);
//    iterate over pixel vertices and fill the pixel vertex Ntuple
    int iVtx = 0;
    for(reco::VertexCollection::const_iterator vi = pixelVertexColl.begin();
	vi != pixelVertexColl.end(); ++vi) {
      if (printOut_ > 0) std::cout << "  " << vi->tracksSize() << "  " << vi->z() << "+/-" << vi->zError() << std::endl;
      pvNtp_.isValid = int(vi->isValid());
      pvNtp_.isFake = int(vi->isFake());
      pvNtp_.Ntrks = vi->tracksSize();
      pvNtp_.nDoF = vi->ndof();
      pvNtp_.chisq = vi->chi2();
      pvNtp_.xV = vi->x();
      pvNtp_.yV = vi->y();
      pvNtp_.zV = vi->z();
      pvNtp_.xVsig = vi->xError();
      pvNtp_.yVsig = vi->yError();
      pvNtp_.zVsig = vi->zError();
      pvTree_->Fill();
      hzV_Iev->Fill(vi->z(), evCnt_);
      if (iVtx == 0) {
	hNtrk_zVerrPri->Fill(vi->zError(), vi->tracksSize());
      } else {
	hNtrk_zVerrSec->Fill(vi->zError(), vi->tracksSize());
	hZvSec->Fill(vi->z() - zv);
      }
      ++iVtx;
    }
    if (printOut_ > 0) std::cout << "  zv = " << zv << " +/- " << zverr << std::endl;
  } else {
    if (printOut_ > 0) std::cout << "No vertices found." << std::endl;
  }

  // Get the simHit info
  makeMap(iEvent);

  /////////////////////////
  // Cluster collections //
  /////////////////////////
  Handle< edmNew::DetSetVector<SiStripCluster> > dsv_SiStripCluster;
  iEvent.getByToken(clusterToken_, dsv_SiStripCluster);

  iEvent.getByToken(stripSimlinkToken_, stripdigisimlink);

  edm::ESHandle<TrackerGeometry> tkgeom;
  iSetup.get<TrackerDigiGeometryRecord>().get( tkgeom );
  if (!tkgeom.isValid()) {
    std::cout << "Unable to find TrackerDigiGeometryRecord in event!";
    return;
  }
  const TrackerGeometry &tracker(*tkgeom);

  // Loop over subdetectors
  int clusterCount = 0;
  int clustersNoDigiSimLink = 0;
  for(edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter=dsv_SiStripCluster->begin();
      DSViter != dsv_SiStripCluster->end(); DSViter++ ) {

    uint32_t detID = DSViter->id();
    DetId theDet(detID);
    int subDet = theDet.subdetId();

    // Find the path length of a straight track from the primary vertex
    const StripGeomDetUnit* DetUnit = 0;
    DetUnit = (const StripGeomDetUnit*) tracker.idToDetUnit(theDet);
    float thickness = 0;
    if (DetUnit != 0) thickness = DetUnit->surface().bounds().thickness();
    int layer = 0;
    int stereo = 0;
    if (subDet == int(StripSubdetector::TIB)) {
      layer = tTopo->tibLayer(detID);
      if (tTopo->tibIsStereo(detID)) stereo = 1;
    }
    if (printOut_ > 0) {
      std::cout << tTopo->print(detID);
      std::cout << " thickness " << thickness << std::endl;
    }

    LocalPoint locVtx = DetUnit->toLocal(GlobalPoint(estPriVtx.X(), estPriVtx.Y(), estPriVtx.Z()));
    float modPathLength = fabs(thickness*locVtx.mag()/locVtx.z());
    if (printOut_ > 0) {
      std::cout << "  Module at " << DetUnit->position() << std::endl;
      std::cout << "  PriVtx at " << locVtx;  printf("%s%7.4f%s%4d%s\n", " path ", modPathLength, ", ", DSViter->size(), " clusters");
    }

    edm::DetSetVector<StripDigiSimLink>::const_iterator isearch = stripdigisimlink->find(detID);

    // Traverse the clusters for this subdetector
    for(edmNew::DetSet<SiStripCluster>::const_iterator ClusIter= DSViter->begin();
	ClusIter!=DSViter->end();ClusIter++) {
      clusterCount++;

      // Look for a digisimlink matching this cluster
      if(isearch == stripdigisimlink->end()) {
	clustersNoDigiSimLink++;
	if (printOut_ > 0) std::cout << "No digisimlinks for this module" << std::endl;
	continue;
      }

      const SiStripCluster* clust = ClusIter;
      std::vector<uint8_t> amp=clust->amplitudes();
      int clusiz = amp.size();
      int first  = clust->firstStrip();
      int last   = first + clusiz;
      float charge = 0;
      bool saturates = false;
      for (int i=0; i<clusiz; ++i) {
	charge+=amp[i];
	if (amp[i] == 254 || amp[i] == 255) saturates = true;
      }
      if (printOut_ > 0) {
	std::cout << "Cluster (width, first, last) = (" << clusiz << ", " << first << ", " << last-1 << ")  charge = " << charge;
	if (saturates) std::cout << "  (saturates)";
	std::cout << endl;
      }
      float clusEloss = 0;
      int tkFlip = 0;
      int ovlap = 0;
      std::vector<unsigned int> trackID;
      std::vector<simhitAddr> CFaddr;
      std::vector<unsigned int> hitProcess;
      std::vector<int> hitPID;
      std::vector<float> trackCharge;
      std::vector<float> hitPmag;
      std::vector<float> hitPathLength;

      int prevIdx = -1;
      edm::DetSet<StripDigiSimLink> link_detset = (*isearch);
      for(edm::DetSet<StripDigiSimLink>::const_iterator linkiter = link_detset.data.begin(), linkEnd = link_detset.data.end();
	  linkiter != linkEnd; ++linkiter) {
	int theChannel = linkiter->channel();
	if (printOut_ > 3) printf("  %s%4d%s\n", "channel = ", theChannel, " before matching to cluster");
        if( theChannel >= first  && theChannel < last ) {
	  // This digisimlink points to a strip in the current cluster
	  int stripIdx = theChannel - first;
	  uint16_t rawAmpl = amp[stripIdx];
	  if (printOut_ > 1)
	    printf("  %s%4d%s%8d%s%8d%s%2d%s%8d%s%8.4f%s%5d\n", "channel = ", linkiter->channel(),
		   " TrackID = ", linkiter->SimTrackId(), " EventID = ", linkiter->eventId().rawId(),
		   " TofBin = ", linkiter->TofBin(), " CFPos = ", linkiter->CFposition(),
		   " fraction = ", linkiter->fraction(), " amp = ", rawAmpl);

	  //
	  // Find simTracks associated with this cluster
	  //
	  unsigned int thisTrackID = linkiter->SimTrackId();
	  // Does at least one strip have >1 track?
	  if (stripIdx == prevIdx) ovlap = 1;
          // Have we seen this track yet?
	  bool newTrack = true;
	  if (std::find(trackID.begin(), trackID.end(), thisTrackID) != trackID.end()) newTrack = false;
	  if (newTrack) {
	    // This is the first time we've encountered this track (linked to this cluster)
	    trackID.push_back(thisTrackID);

	    trackCharge.push_back(linkiter->fraction()*rawAmpl);
	  } else {
	    for (unsigned int i=0; i<trackID.size(); ++i)
	      if (trackID[i] == thisTrackID)
		trackCharge[i] += linkiter->fraction()*rawAmpl;
	  } // if newTrack ... else
	  if (printOut_ > 2) {
	    std::cout << "    Track charge accumulator = ";
	    for (unsigned int it = 0; it < trackCharge.size(); ++it) printf("%7.1f  ", trackCharge[it]);
	    std::cout << std::endl;
	  }

	  //
	  // Find simHits associaed with this cluster
	  //
	  unsigned int currentCFPos = linkiter->CFposition();
	  unsigned int tofBin = linkiter->TofBin();
	  simHitCollectionID theSimHitCollID = std::make_pair(theDet.subdetId(), tofBin);
	  simhitAddr currentAddr = std::make_pair(theSimHitCollID, currentCFPos);
	  bool newHit = true;
	  if (std::find(CFaddr.begin(), CFaddr.end(), currentAddr) != CFaddr.end()) newHit = false;
	  if (newHit) {
	    simhit_collectionMap::const_iterator it = SimHitCollMap_.find(theSimHitCollID);
	    if (it!= SimHitCollMap_.end()) {
	      if (currentCFPos < (it->second).size()) {
		const PSimHit& theSimHit = (it->second)[currentCFPos];
		CFaddr.push_back(currentAddr);
		hitProcess.push_back(theSimHit.processType());
		hitPID.push_back(theSimHit.particleType());
		hitPmag.push_back(theSimHit.pabs());
		Local3DPoint entry = theSimHit.entryPoint();
		Local3DPoint exit = theSimHit.exitPoint();
		Local3DVector segment = exit - entry;
		hitPathLength.push_back(segment.mag());
		clusEloss += theSimHit.energyLoss();  // Add up the contributions of all simHits to this cluster
		if (printOut_ > 1 && theSimHit.pabs() >= 1.0) printf("    %s%3d%s%3d%s%5d%s%7.2f%s%10.6f%s%8.4f%s%8.4f\n",
								     "SimHit ", int(CFaddr.size()),
								     ", process = ", theSimHit.processType(),
								     ", PID = ", theSimHit.particleType(),
								     ", p = ", theSimHit.pabs(),
								     ", Eloss = ", theSimHit.energyLoss(),
								     ", segment = ", segment.mag(),
								     ", str segment = ", modPathLength);
	      } else {
		std::cout << "currentCFPos " << currentCFPos << " is out of range for " << (it->second).size() << std::endl;
	      }
	    }
	  }  // if (newHit)
	  prevIdx = stripIdx;
	}  // DigiSimLink belongs to this cluster
      }  // Traverse DigiSimLinks
      if (prevIdx == -1) continue;  // No truth information for this cluster; move on to the next one.

      if (ovlap && trackID[1] < trackID[0]) tkFlip = 1;

// RecoTracker/DeDx/python/dedxDiscriminator_Prod_cfi.py, line 12 -- MeVperADCStrip = cms.double(3.61e-06*250)

      // Fill the cluster Ntuple
      clusNtp_.subDet = subDet;
      clusNtp_.thickness = thickness;
      clusNtp_.width = amp.size();
      clusNtp_.NsimHits = CFaddr.size();
      clusNtp_.firstProcess = hitProcess.size() > 0 ? hitProcess[0] : -1;
      clusNtp_.secondProcess = hitProcess.size() > 1 ? hitProcess[1] : -1;
      clusNtp_.thirdProcess = hitProcess.size() > 2 ? hitProcess[2] : -1;
      clusNtp_.fourthProcess = hitProcess.size() > 3 ? hitProcess[3] : -1;
      clusNtp_.firstPID = hitPID.size() > 0 ? hitPID[0] : 0;
      clusNtp_.secondPID = hitPID.size() > 1 ? hitPID[1] : 0;
      clusNtp_.thirdPID = hitPID.size() > 2 ? hitPID[2] : 0;
      clusNtp_.fourthPID = hitPID.size() > 3 ? hitPID[3] : 0;
      clusNtp_.firstPmag = hitPmag.size() > 0 ? hitPmag[0] : 0;
      clusNtp_.secondPmag = hitPmag.size() > 1 ? hitPmag[1] : 0;
      clusNtp_.thirdPmag = hitPmag.size() > 2 ? hitPmag[2] : 0;
      clusNtp_.fourthPmag = hitPmag.size() > 3 ? hitPmag[3] : 0;
      clusNtp_.firstPathLength = hitPathLength.size() > 0 ? hitPathLength[0]: 0;
      clusNtp_.secondPathLength = hitPathLength.size() > 1 ? hitPathLength[1]: 0;
      clusNtp_.thirdPathLength = hitPathLength.size() > 2 ? hitPathLength[2]: 0;
      clusNtp_.fourthPathLength = hitPathLength.size() > 3 ? hitPathLength[3]: 0;
      clusNtp_.pathLstraight = modPathLength;
      float allHtPathLength = 0;
      for (unsigned int ih=0; ih<hitPathLength.size(); ++ih)
	  allHtPathLength += hitPathLength[ih];
      clusNtp_.allHtPathLength = allHtPathLength;
	clusNtp_.Ntp = trackID.size();
	if (printOut_ > 0 && trackCharge.size() == 2)
	  cout << "  charge 1st, 2nd, dE/dx 1st, 2nd, asymmetry = "
             << trackCharge[0] << "  "
             << trackCharge[1] << "  "
             << 3.36e-4*trackCharge[0]/modPathLength << "  "
             << 3.36e-4*trackCharge[1]/modPathLength << "  "
	       << (trackCharge[0]-trackCharge[1]) / (trackCharge[0]+trackCharge[1]) 
             << "  " << tkFlip << endl;
      clusNtp_.firstTkChg = trackCharge.size() > 0 ? trackCharge[0] : 0;
      clusNtp_.secondTkChg = trackCharge.size() > 1 ? trackCharge[1] : 0;
      clusNtp_.thirdTkChg = trackCharge.size() > 2 ? trackCharge[2] : 0;
      clusNtp_.fourthTkChg = trackCharge.size() > 3 ? trackCharge[3] : 0;
      clusNtp_.charge = charge;
      clusNtp_.Eloss = clusEloss;
      clusNtp_.sat = saturates ? 1 : 0;
      clusNtp_.tkFlip = tkFlip;
      clusNtp_.ovlap = ovlap;
      clusNtp_.layer = layer;
      clusNtp_.stereo = stereo;
      clusTree_->Fill();
    }  // traverse clusters in subdetector
  }  // traverse subdetectors
  if (printOut_ > 0) std::cout << clusterCount << " total clusters; " << clustersNoDigiSimLink << " without digiSimLinks" << std::endl;
}


// ------------ method called once each job just before starting event loop  ------------
void 
StripClusterMCanalysis::beginJob()
{
  int bufsize = 64000;
  edm::Service<TFileService> fs;
  // Create the cluster tree
  clusTree_ = fs->make<TTree>("ClusterNtuple", "Cluster analyzer ntuple");
  clusTree_->Branch("cluster", &clusNtp_, "subDet/I:thickness/F:width/I:NsimHits:firstProcess:secondProcess:thirdProcess:fourthProcess:firstPID:secondPID:thirdPID:fourthPID:Ntp:firstTkChg/F:secondTkChg:thirdTkChg/F:fourthTkChg:charge:firstPmag:secondPmag:thirdPmag:fourthPmag:firstPathLength:secondPathLength:thirdPathLength:fourthPathLength:pathLstraight:allHtPathLength:Eloss:sat/I:tkFlip:ovlap:layer:stereo", bufsize);
  // Create the vertex tree
  pvTree_ = fs->make<TTree>("pVertexNtuple", "Pixel vertex ntuple");
  pvTree_->Branch("pVertex", &pvNtp_, "isValid/I:isFake:Ntrks:nDoF:chisq/F:xV:yV:zV:xVsig:yVsig:zVsig", bufsize);

  // Book some histograms
  hNpv = fs->make<TH1F>("hNpv", "No. of pixel vertices", 40, 0, 40);
  hzV_Iev = fs->make<TH2F>("hzV_Iev", "Zvertex vs event index", 20, -10, 10, 100, 0, 100);
  hNtrk_zVerrPri = fs->make<TH2F>("hzVerr_NtrkPri", "Ntracks vs Zvertex error, primary", 50, 0, 0.025, 30, 0, 150);
  hNtrk_zVerrSec = fs->make<TH2F>("hzVerr_NtrkSec", "Ntracks vs Zvertex error, secondary", 50, 0, 0.025, 30, 0, 150);
  hZvPri = fs->make<TH1F>("hZvPri", "Zvertex, primary", 48, -15, 15);
  hZvSec = fs->make<TH1F>("hZvSec", "Zvertex, secondary", 48, -15, 15);
  evCnt_ = 0;

}

// ------------ method called once each job just after ending the event loop  ------------
void 
StripClusterMCanalysis::endJob() {
}

void StripClusterMCanalysis::makeMap(const edm::Event& theEvent) {
  //  The simHit collections are specified in trackerContainers, and can
  //  be either crossing frames (e.g., mix/g4SimHitsTrackerHitsTIBLowTof) 
  //  or just PSimHits (e.g., g4SimHits/TrackerHitsTIBLowTof)

  SimHitCollMap_.clear();
  for(auto const& cfToken : cfTokens_) {
    edm::Handle<CrossingFrame<PSimHit> > cf_simhit;
    int Nhits = 0;
    // if (theEvent.getByToken(cfToken, cf_simhit)) {
    if (theEvent.getByToken(cfToken, cf_simhit)) {
      std::unique_ptr<MixCollection<PSimHit> > thisContainerHits(new MixCollection<PSimHit>(cf_simhit.product()));
      for (auto const& isim : *thisContainerHits) {
        DetId theDet(isim.detUnitId());
	edm::EDConsumerBase::Labels labels;
	theEvent.labelsForToken(cfToken, labels);
	std::string trackerContainer(labels.productInstance);
	if (printOut_ && Nhits==0) std::cout << "  trackerContainer " << trackerContainer << std::endl;
	unsigned int tofBin = StripDigiSimLink::LowTof;
	if (trackerContainer.find(std::string("HighTof")) != std::string::npos) tofBin = StripDigiSimLink::HighTof;
	simHitCollectionID theSimHitCollID = std::make_pair(theDet.subdetId(), tofBin);
	SimHitCollMap_[theSimHitCollID].push_back(isim);
        ++Nhits;
      }
      if (printOut_ > 0) std::cout << "simHits from crossing frames; map size = " << SimHitCollMap_.size()
                                   << ", Hit count = " << Nhits << ", " << sizeof(SimHitCollMap_)
				   << " bytes" << std::endl;
    }
  }
  for(auto const& simHitToken : simHitTokens_) {
    edm::Handle<std::vector<PSimHit> > simHits;
    int Nhits = 0;
    if(theEvent.getByToken(simHitToken, simHits)) {
      for (auto const& isim : *simHits) {
        DetId theDet(isim.detUnitId());
	edm::EDConsumerBase::Labels labels;
	theEvent.labelsForToken(simHitToken, labels);
	std::string trackerContainer(labels.productInstance);
	if (printOut_ && Nhits==0) std::cout << "  trackerContainer " << trackerContainer << std::endl;
	unsigned int tofBin = StripDigiSimLink::LowTof;
	if (trackerContainer.find(std::string("HighTof")) != std::string::npos) tofBin = StripDigiSimLink::HighTof;
	simHitCollectionID theSimHitCollID = std::make_pair(theDet.subdetId(), tofBin);
	SimHitCollMap_[theSimHitCollID].push_back(isim);
        ++Nhits;
      }
      if (printOut_ > 0) std::cout << "simHits from hard-scatter collection; map size = " << SimHitCollMap_.size()
                                   << ", Hit count = " << Nhits << ", " << sizeof(SimHitCollMap_)
				   << " bytes" << std::endl;
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(StripClusterMCanalysis);

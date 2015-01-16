#include <memory>
#include <map>

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimTracker/SiPhase2Digitizer/interface/PixelClusterSimLink.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>
#include <THStack.h>
#include <TProfile.h>
#include <TMath.h>

#include "SimTracker/SiPhase2Digitizer/test/hits.h"

int verbose = 0;
const int nTypes = 18;
const int nPS=3;

using namespace std;

class SiPhase2ClustersValidation : public edm::EDAnalyzer {

    typedef vector< pair< PSimHit , vector< ValHit > > > V_HIT_CLUSTERS;
    typedef map< int , V_HIT_CLUSTERS > M_TRK_HIT_CLUSTERS;

public:
    explicit SiPhase2ClustersValidation(const edm::ParameterSet&);
    ~SiPhase2ClustersValidation();
    void beginJob();
    void analyze(const edm::Event&, const edm::EventSetup&);
    void endJob();

private:
    void calculataThor(ValHitsCollection* hitsCollection, edm::PSimHitContainer* simHits_B, edm::PSimHitContainer* simHits_E, edm::SimTrackContainer* simTracks, edm::SimVertexContainer* simVertices, const TrackerGeometry* tkGeom);
    void createLayerHistograms(unsigned int iLayer);
    void createHistograms(unsigned int nLayer);
    unsigned int getLayerNumber(const TrackerGeometry* tkgeom, unsigned int& detid);
    unsigned int getLayerNumber(unsigned int& detid);


    bool useRecHits_;

    TH2F* trackerLayout_;
    TH2F* trackerLayoutXY_;
    TH2F* trackerLayoutXYBar_;
    TH2F* trackerLayoutXYEC_;

    struct ClusterHistos {
        THStack* NumberOfClustersSource;
        TH1F* NumberOfClusterPixel;
        TH1F* NumberOfClusterStrip;

        TH1F* NumberOfClustersLink;

        TH1F* NumberOfMatchedHits[nPS][nTypes];
        TH1F* NumberOfMatchedClusters[nPS][nTypes];
        TH1F* hEfficiency[nPS][nTypes];
        TH1F* h_dx_Truth[nPS];
        TH1F* h_dy_Truth[nPS];

        THStack* ClustersSizeSource;
        TH1F* clusterSizePixel;
        TH1F* clusterSizeStrip;

        TH1F* clusterSize;
        TH1F* clusterSizeX;
        TH1F* clusterSizeY;

        TH1F* clusterShapeX;
        TH1F* clusterShapeY;
        TH2F* localPosXY;
        TH2F* globalPosXY;

        TH2F* localPosXYPixel;
        TH2F* localPosXYStrip;

        TH1F* digiType;
        TH2F* digiPosition;

        TH1F* dxCluRec;
        TH1F* dyCluRec;
        TH1F* dRCluRec;
    };

    map< unsigned int, ClusterHistos > layerHistoMap;

    string name_PS[nPS] = {"AllMod", "PixelMod", "StripMod"};

    string name_types[nTypes] = {"Undefined","Unknown","Primary","Hadronic",
                 "Decay","Compton","Annihilation","EIoni",
                 "HIoni","MuIoni","Photon","MuPairProd",
                 "Conversions","EBrem","SynchrotronRadiation",
                 "MuBrem","MuNucl","AllTypes"};

};

SiPhase2ClustersValidation::SiPhase2ClustersValidation(const edm::ParameterSet& iConfig) {
    useRecHits_ = iConfig.getParameter< bool >("useRecHits");

    // Use RecHits
    if (useRecHits_) cout << "INFO: Using RecHits" << endl;
    // Use Clusters
    else cout << "INFO: Using Clusters" << endl;
}

SiPhase2ClustersValidation::~SiPhase2ClustersValidation() { }

void SiPhase2ClustersValidation::beginJob() {
    createHistograms(19);
}

void SiPhase2ClustersValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

    // Get the clusters
    edm::Handle< SiPixelClusterCollectionNew > clustersHandle;
    iEvent.getByLabel("siPhase2Clusters", clustersHandle);
    // const edmNew::DetSetVector< SiPixelCluster >& clusters = *clustersHandle;

    // Get the cluster simlinks
    edm::Handle< edm::DetSetVector< PixelClusterSimLink > > clusterLinksHandle;
    iEvent.getByLabel("siPhase2Clusters", clusterLinksHandle);
    const edm::DetSetVector< PixelClusterSimLink >& clusterLinks = *clusterLinksHandle;

    // Get the geometry
    edm::ESHandle< TrackerGeometry > geomHandle;
    iSetup.get< TrackerDigiGeometryRecord >().get(geomHandle);
    const TrackerGeometry* tkGeom = &(*geomHandle);


    // Make selection on RecHits or Clusters
    ValHitsCollection hitsCollection;

    //Get the RecHits
    if (useRecHits_) {
        edm::Handle< SiPixelRecHitCollection > recHitsHandle;
        iEvent.getByLabel("siPixelRecHits", recHitsHandle);
        const edmNew::DetSetVector< SiPixelRecHit >& recHits = *recHitsHandle;

        hitsCollection = ValHitsBuilder(tkGeom, (edm::DetSetVector< PixelClusterSimLink >*) & clusterLinks, (edmNew::DetSetVector< SiPixelRecHit >*) & recHits);
    }
    // Use Clusters
    else hitsCollection = ValHitsBuilder(tkGeom, (edm::DetSetVector< PixelClusterSimLink >*) & clusterLinks);

    // SimHit
    edm::Handle< edm::PSimHitContainer > simHits_BHandle;
    iEvent.getByLabel("g4SimHits", "TrackerHitsPixelBarrelLowTof", simHits_BHandle);
    const edm::PSimHitContainer& simHits_B = *simHits_BHandle;

    edm::Handle< edm::PSimHitContainer > simHits_EHandle;
    iEvent.getByLabel("g4SimHits", "TrackerHitsPixelEndcapLowTof", simHits_EHandle);
    const edm::PSimHitContainer& simHits_E = *simHits_EHandle;

    // SimTrack
    edm::Handle< edm::SimTrackContainer > simTracksHandle;
    iEvent.getByLabel("g4SimHits", simTracksHandle);
    const edm::SimTrackContainer& simTracks = *simTracksHandle;

    // SimVertex
    edm::Handle< edm::SimVertexContainer > simVerticesHandle;
    iEvent.getByLabel("g4SimHits", simVerticesHandle);
    const edm::SimVertexContainer& simVertices = *simVerticesHandle;

    // Validation module
    calculataThor((ValHitsCollection*) & hitsCollection, (edm::PSimHitContainer*) & simHits_B, (edm::PSimHitContainer*) & simHits_E, (edm::SimTrackContainer*) & simTracks, (edm::SimVertexContainer*) & simVertices, tkGeom);
}

void SiPhase2ClustersValidation::endJob() { }


void SiPhase2ClustersValidation::calculataThor(ValHitsCollection* hitsCollection, edm::PSimHitContainer* simHits_B, edm::PSimHitContainer* simHits_E, edm::SimTrackContainer* simTracks, edm::SimVertexContainer* simVertices, const TrackerGeometry* tkGeom) {

    ////////////////////////////////
    // MAP SIM HITS TO SIM TRACKS //
    ////////////////////////////////
    vector< ValHit > matched_clusters;
    V_HIT_CLUSTERS matched_hits;
    M_TRK_HIT_CLUSTERS map_hits;

    // Fill the map
    int nHits = 0;
    for (edm::PSimHitContainer::const_iterator iHit = simHits_B->begin(); iHit != simHits_B->end(); ++iHit) {
        map_hits[iHit->trackId()].push_back(make_pair(*iHit , matched_clusters));
        nHits++;
    }
    for (edm::PSimHitContainer::const_iterator iHit = simHits_E->begin(); iHit != simHits_E->end(); ++iHit) {
        map_hits[iHit->trackId()].push_back(make_pair(*iHit , matched_clusters));
        nHits++;
    }

    if (verbose > 1) cout << endl << "-- Number of SimHits in the event : " << nHits << endl;

    //////////////////////////////////
    // LOOP OVER CLUSTER COLLECTION //
    //////////////////////////////////

    // Loop over the detector units
    for (ValHitsCollection::const_iterator vhCollectionIter = hitsCollection->begin(); vhCollectionIter != hitsCollection->end(); ++vhCollectionIter) {

        ValHitsVector hitsVector = vhCollectionIter->second;

        // Clusters
        unsigned int nClusters = 0;

        // Get the detector unit's id
        unsigned int rawid = vhCollectionIter->first;
        unsigned int layer = getLayerNumber(rawid);
        DetId detId(rawid);

        // Get the geometry of the tracker
        const GeomDetUnit* geomDetUnit = tkGeom->idToDetUnit(detId);
        const PixelGeomDetUnit* theGeomDet = dynamic_cast< const PixelGeomDetUnit* >(geomDetUnit);
        const PixelTopology& topol = theGeomDet->specificTopology();

        if (!geomDetUnit) break;

        // Create histograms for the layer if they do not yet exist
        map< unsigned int, ClusterHistos >::iterator iPos = layerHistoMap.find(layer);
        if (iPos == layerHistoMap.end()) {
            createLayerHistograms(layer);
            iPos = layerHistoMap.find(layer);
        }

        // Loop over the clusters in the detector unit
        for (ValHitsVector::const_iterator vhVectorIter = hitsVector.begin(); vhVectorIter != hitsVector.end(); ++vhVectorIter) {

            ValHit hit = *vhVectorIter;
            edm::Ref< edmNew::DetSetVector< SiPixelCluster >, SiPixelCluster > const& cluster = hit.cluster;

            iPos->second.clusterSize->Fill(cluster->size());
            iPos->second.clusterSizeX->Fill(cluster->sizeX());
            iPos->second.clusterSizeY->Fill(cluster->sizeY());

            // Fill the histograms
            MeasurementPoint mpClu(cluster->x(), cluster->y());
            Local3DPoint localPosClu = geomDetUnit->topology().localPosition(mpClu);

            iPos->second.dxCluRec->Fill(localPosClu.x() - hit.localPos.x());
            iPos->second.dyCluRec->Fill(localPosClu.y() - hit.localPos.y());
            iPos->second.dRCluRec->Fill(TMath::Sqrt((localPosClu.y() - hit.localPos.y()) * (localPosClu.y() - hit.localPos.y()) + (localPosClu.x() - hit.localPos.x()) * (localPosClu.x() - hit.localPos.x())));


            iPos->second.localPosXY->Fill(hit.localPos.x(), hit.localPos.y());
            iPos->second.globalPosXY->Fill(hit.globalPos.x(), hit.globalPos.y());

            trackerLayout_->Fill(hit.globalPos.z(), hit.globalPos.perp());
            trackerLayoutXY_->Fill(hit.globalPos.x(), hit.globalPos.y());
            if (layer < 100) trackerLayoutXYBar_->Fill(hit.globalPos.x(), hit.globalPos.y());
            else trackerLayoutXYEC_->Fill(hit.globalPos.x(), hit.globalPos.y());

            // Pixel module
            if (topol.ncolumns() == 32) {
                iPos->second.localPosXYPixel->Fill(hit.localPos.x(), hit.localPos.y());
                iPos->second.clusterSizePixel->Fill(cluster->size());
            }
            // Strip module
            else if (topol.ncolumns() == 2) {
                iPos->second.localPosXYStrip->Fill(hit.localPos.x(), hit.localPos.y());
                iPos->second.clusterSizeStrip->Fill(cluster->size());
            }

            // Get the pixels that form the Cluster
            const vector< SiPixelCluster::Pixel >& pixelsVec = cluster->pixels();

            // Loop over the pixels
            for (vector< SiPixelCluster::Pixel >::const_iterator pixelIt = pixelsVec.begin(); pixelIt != pixelsVec.end(); ++pixelIt) {
                SiPixelCluster::Pixel PDigi = (SiPixelCluster::Pixel) *pixelIt;

                iPos->second.digiPosition->Fill(PDigi.x, PDigi.y);

                //////////////////////////
                // NOT WORKING !!!!!!   //
                //////////////////////////
                iPos->second.clusterShapeX->Fill(hit.localPos.x() - PDigi.x);
                iPos->second.clusterShapeY->Fill(hit.localPos.y() - PDigi.y);
            }

            ++nClusters;
        }

        // Pixel module
        if (topol.ncolumns() == 32) iPos->second.NumberOfClusterPixel->Fill(nClusters);
        // Strip module
        else if (topol.ncolumns() == 2) iPos->second.NumberOfClusterStrip->Fill(nClusters);
    }

    /////////////////////////////
    // LOOP OVER CLUSTER LINKS //
    /////////////////////////////

    vector< unsigned int > simTrackID;

    // cluster and hit informations
    unsigned int trkID(-1);
    unsigned int sizeLink(0);
    unsigned int rawid(0);
    unsigned int layer(0);
    unsigned int simh_detid(0);
    unsigned int simh_layer(0);
    int simh_type(0);
    unsigned int nLinks(0);
    bool combinatoric(false);

    // matching quantities
    //
    // all hits ; type-2 hits ; primary hits ; secondary hits
    int nMatchedHits[nTypes] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    //
    Local3DPoint pos_hit;
    double x_hit(0), y_hit(0), z_hit(0), dx(0), dy(0);
    bool found_hits(false);
    bool fill_dtruth(false);

    if (verbose > 1) cout << endl << "-- Enter loop over links" << endl;

    // Loop over the Hits
    for (ValHitsCollection::const_iterator vhCollectionIter = hitsCollection->begin(); vhCollectionIter != hitsCollection->end(); ++vhCollectionIter) {

        ValHitsVector hitsVector = vhCollectionIter->second;

        trkID = -1;
        sizeLink = 0;
        combinatoric = false;

        // Get the detector unit's id
        rawid = vhCollectionIter->first;
        layer = getLayerNumber(rawid);
        DetId detId(rawid);

        // Get the geometry of the tracker
        const GeomDetUnit* geomDetUnit = tkGeom->idToDetUnit(detId);
	const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(geomDetUnit);
	const PixelTopology & topol = theGeomDet->specificTopology();

        if (!geomDetUnit) break;

        // Create histograms for the layer if they do not yet exist
        map< unsigned int, ClusterHistos >::iterator iPos = layerHistoMap.find(layer);
        if (iPos == layerHistoMap.end()) {
            createLayerHistograms(layer);
            iPos = layerHistoMap.find(layer);
        }

        // Loop over the links in the detector unit
        if (verbose>1) cout << endl << endl << "--- DetId=" << rawid << endl;

        for (ValHitsVector::const_iterator vhVectorIter = hitsVector.begin(); vhVectorIter != hitsVector.end(); ++vhVectorIter) {

            ValHit hit = *vhVectorIter;

            // Link informations
            combinatoric = false;
            nLinks++;
            simTrackID = hit.simTracks;
            sizeLink = simTrackID.size();

            // cluster matching quantities
            for (int iM = 0; iM < nTypes; iM++) nMatchedHits[iM] = 0;

            // Cluster informations
            edm::Ref< edmNew::DetSetVector< SiPixelCluster >, SiPixelCluster > const& cluster = hit.cluster;

            if (verbose > 1) cout << endl << "---- Cluster size="  << cluster->size() << " | " << sizeLink << " SimTracks: ids=(" ;

            for (unsigned int i = 0; i < sizeLink; i++) {
                if (verbose > 1) {
                    cout << simTrackID[i];
                    if (i < sizeLink - 1) cout << ",";
                }
                if (i == 0) trkID = simTrackID[i];
                else if (simTrackID[i] != trkID) combinatoric = true;
            }

            if (verbose > 1) {
                if (combinatoric) cout << ") COMBINATORIC !!! ";
                cout << ")" << endl << "     cluster local position = (" << hit.localPos.x() << " , " << hit.localPos.y() << " , " << "n/a"    << ")" << "   layer=" << layer << endl;
            }

            // Get matched SimHits from the map
            if (!combinatoric) {

                matched_hits = map_hits[trkID];

                if (verbose > 1) {
                    cout << "     number of hits matched to the SimTrack = " << matched_hits.size() ;
                    if (matched_hits.size() != 0) cout << " ids(" ;

                    // printout list of SimHits matched to the SimTrack
                    for (unsigned int iH = 0; iH < matched_hits.size(); ++iH) {
                        cout << matched_hits[iH].first.detUnitId();
                        if (iH < matched_hits.size() - 1) cout << ",";
                        else cout << ")";
                    }
                    cout << endl;
                }

                found_hits = false;
                fill_dtruth = false;

                // Loop over matched SimHits

                for (unsigned int iH = 0 ; iH < matched_hits.size(); iH++) {

                    // Consider only SimHits with same DetID as current cluster
                    simh_detid = matched_hits[iH].first.detUnitId();
                    if (simh_detid != rawid) continue;
                    else found_hits = true;


                    // Map current hit to current SimHit for efficiency study
                    map_hits[trkID][iH].second.push_back(hit);

                    simh_layer = getLayerNumber(simh_detid);
                    simh_type = matched_hits[iH].first.processType();
                    pos_hit = matched_hits[iH].first.localPosition();
                    x_hit = pos_hit.x();
                    y_hit = pos_hit.y();
                    z_hit = pos_hit.z();

                    if (simh_type >= 0 && simh_type < 17) nMatchedHits[simh_type]++;
                    nMatchedHits[17]++;

                    if (simh_type == 2) {
                        dx = x_hit - hit.localPos.x();
                        dy = y_hit - hit.localPos.y();
                        if (fill_dtruth == true) fill_dtruth = false; // eliminates cases with several type-2 hits
                        fill_dtruth = true; // toggle filling of the histo only when a type-2 hit is found
                    }

                    if (verbose > 1) cout << "----- SimHit #" << iH << " type="    << simh_type
                        //<< " s_id="    << simh_detid
                        << " s_lay="   << simh_layer << " c_lay="   << layer << " s("   << x_hit    << " , " << y_hit    << " , " << z_hit    << ")"
                        //<< " c_g(" << gPos.x() << " , " << gPos.y() << " , " << gPos.z() << ")"
                        << endl;

                } // end loop over matched SimHits

                if (!found_hits && verbose > 1) cout << "----- FOUND NO MATCHED HITS" << endl;

            } // endif !combinatoric

            // Number of matched hits (per type)
	    for(int iM=0 ; iM<nTypes ; iM++) {
	      iPos->second.NumberOfMatchedHits[0][iM]-> Fill(nMatchedHits[iM]);
	      if(verbose>1) cout << "------ type #" << iM << " " << name_types[nTypes] << " : " ;
	      if(topol.ncolumns() == 32)    {
		if(verbose>1) cout << "module pixel : nMatchedHits=" << nMatchedHits[iM] << endl;
		iPos->second.NumberOfMatchedHits[1][iM]-> Fill(nMatchedHits[iM]);
	      }
	      else if (topol.ncolumns()==2) {
		if(verbose>1) cout << "module strip : nMatchedHits=" << nMatchedHits[iM] << endl;
		iPos->second.NumberOfMatchedHits[2][iM]-> Fill(nMatchedHits[iM]);
	      }
	      else {
		if(verbose>1) cout << "module unknown : nMatchedHits=" << nMatchedHits[iM] << endl;
	      }
	    }

            // Position resolution
            if (fill_dtruth) {
	      iPos->second.h_dx_Truth[0]->Fill(dx);
	      iPos->second.h_dy_Truth[0]->Fill(dy);
	      if(topol.ncolumns() == 32)    {
		iPos->second.h_dx_Truth[1]->Fill(dx);
		iPos->second.h_dy_Truth[1]->Fill(dy);
	      }
	      else if (topol.ncolumns()==2) {
		iPos->second.h_dx_Truth[2]->Fill(dx);
		iPos->second.h_dy_Truth[2]->Fill(dy);
	      }
	    }

        } // end loop over links within a single DetID

        iPos->second.NumberOfClustersLink-> Fill(nLinks);

    } // end loop over all links


    ////////////////////////////////////
    // COMPUTE CLUSTERIZER EFFICIENCY //
    ////////////////////////////////////

    if (verbose > 1) cout << "- Enter efficiency computation" << endl;

    // Iterate over the map of hits & clusters
    M_TRK_HIT_CLUSTERS::const_iterator iMapHits;

    // Counters
    int nTrackHits(0);
    int countHit(0);
    int nMatchedClusters(0);
    int nTotalHits(0);
    int nMatchHits(0);
    float efficiency(0);

    // Hit informations
    unsigned int theHit_id(0);
    unsigned int theHit_layer(0);
    unsigned int theHit_type(0);

    // Prepare the map of counters for efficiency
    map< unsigned int, ClusterHistos >::iterator iPos;
    map< unsigned int , vector< vector< int > > > map_effi;
    map< unsigned int , vector< vector< int > > >::const_iterator iMapEffi;
    vector< int > init_counter;

    for (int iM = 0; iM < 2; iM++) init_counter.push_back(0);

    // Loop over the entries in the map of hits & clusters
    if (verbose > 1) cout << "- loop over map of hits & clusters (size=" << map_hits.size() << ")" << endl;

    for (iMapHits = map_hits.begin(); iMapHits != map_hits.end(); iMapHits++) {

        if (verbose > 1) cout << "-- SimTrack ID=" << iMapHits->first << endl;
        nTrackHits = (iMapHits->second).size();
        countHit += nTrackHits;

        // Loop over the hits matched to the current SimTrack ID
        for (int iH = 0; iH < nTrackHits; iH++) {

            // Current SimHit
            if (verbose > 1) cout << "--- SimHit #" << iH ;
            PSimHit theHit(((iMapHits->second)[iH]).first);
            theHit_id    = theHit.detUnitId();
            theHit_layer = getLayerNumber(theHit_id);
            theHit_type  = theHit.processType();

	    const GeomDetUnit* geomDetUnit = tkGeom->idToDetUnit(theHit_id);
	    const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(geomDetUnit);
	    const PixelTopology & topol = theGeomDet->specificTopology();

	    if(verbose>1) cout << " DetId=" << theHit_id
			       << " Layer=" << theHit_layer
			       << " Type="  << theHit_type
			       << " Topol nCol=" << topol.ncolumns()
			       << endl;

            // Check that the layer number makes sense
            if (theHit_layer == 0) {
                if (verbose > 1) cout << "---- !! layer=0 !!" << endl;
                continue;
            }

            // Clusters matched to the SimHit
            if (verbose > 2) cout << "--- Getting corresponding clusters" << endl;
            matched_clusters = ((iMapHits->second)[iH]).second;
            nMatchedClusters = matched_clusters.size();

            // Find layer in map of histograms
            if (verbose > 2) cout << "--- Find layer=" << theHit_layer << " in map of histograms" << endl;

            iPos = layerHistoMap.find(theHit_layer);
            if (iPos == layerHistoMap.end()) {
                if (verbose > 2) cout << "---- add layer in the map" << endl;
                createLayerHistograms(theHit_layer);
                iPos = layerHistoMap.find(theHit_layer);
            }

            // Fill Histograms
	    (iPos->second.NumberOfMatchedClusters[0][17])->Fill( nMatchedClusters );
	    if (topol.ncolumns() == 32)     (iPos->second.NumberOfMatchedClusters[1][17])->Fill( nMatchedClusters ); // ND
	    else if (topol.ncolumns() == 2) (iPos->second.NumberOfMatchedClusters[2][17])->Fill( nMatchedClusters ); // ND

	    if(theHit_type<17) {
	      (iPos->second.NumberOfMatchedClusters[0][theHit_type])->Fill( nMatchedClusters );
	      if (topol.ncolumns() == 32)     (iPos->second.NumberOfMatchedClusters[1][theHit_type])->Fill( nMatchedClusters );
	      else if (topol.ncolumns() == 2) (iPos->second.NumberOfMatchedClusters[2][theHit_type])->Fill( nMatchedClusters );
	    }

            if (nMatchedClusters == 0 && verbose > 1) cout << "---- No Cluster Matched" << endl;
            else if (verbose > 1) cout << "---- Yes Cluster Matched = " << nMatchedClusters << endl;

            if (map_effi.find(theHit_layer) == map_effi.end()) {
                for (int iT = 0; iT < nTypes; iT++) {
                    map_effi[theHit_layer].push_back(init_counter);
                    if (verbose > 2) cout << "----- type #" << iT << " layer=" << theHit_layer << " map size=" << map_effi.size() << endl;
                }
            }

            (map_effi[theHit_layer][theHit_type][0])++; // total number of hits of this type in this layer
            if (nMatchedClusters > 0) (map_effi[theHit_layer][theHit_type][1])++; // number of hits matched to >=1 cluster(s)

            (map_effi[theHit_layer][17][0])++; // total number of hits of this type in this layer
            if (nMatchedClusters > 0) (map_effi[theHit_layer][17][1])++; // number of hits matched to >=1 cluster(s)
        }

    }


    // Fill histograms from the map_effi
    if (verbose > 1) cout << "- fill [per layer] effi histo from effi map (size=" << map_effi.size() << ")" << endl;

    for (iMapEffi = map_effi.begin(); iMapEffi != map_effi.end(); iMapEffi++) {

        iPos = layerHistoMap.find(iMapEffi->first);
        if (verbose > 1) cout << "-- layer=" << iMapEffi->first << endl;

        for (int iT = 0; iT < nTypes; iT++) {
            nTotalHits = iMapEffi->second[iT][0];
            nMatchHits = iMapEffi->second[iT][1];
            efficiency = nTotalHits != 0 ? float(nMatchHits) / float(nTotalHits) : -1;

	    if(efficiency>=0) {
	      (iPos->second.hEfficiency[0][iT])->Fill( efficiency );
	      //if(topol.ncolumns() == 32)    (iPos->second.hEfficiency[1][iT])->Fill( efficiency ); // ND
	      //else if (topol.ncolumns()==2) (iPos->second.hEfficiency[2][iT])->Fill( efficiency ); // ND
	    }

	    if(verbose>1) cout << "--- type #"   << iT
			       << " nTotalHits=" << nTotalHits
			       << " nMatchHits=" << nMatchHits
			       << " efficiency=" << efficiency
			       << endl;
        }
    }


    // Check if all event's SimHits are mapped
    if (countHit != nHits && verbose > 1) cout << "---- Missing hits in the efficiency computation : " << countHit << " != " << nHits << endl;

    if (verbose > 999) cout << nTotalHits << nMatchHits << efficiency << endl;
}

// Create the histograms
void SiPhase2ClustersValidation::createLayerHistograms(unsigned int ival) {
    ostringstream fname1, fname2;

    edm::Service<TFileService> fs;
    fs->file().cd("/");

    string tag;
    unsigned int id;
    if (ival < 100) {
        id = ival;
        fname1 << "Barrel";
        fname2 << "Layer_" << id;
        tag = "_layer_";
    }
    else {
        int side = ival / 100;
        id = ival - side*100;
        fname1 << "EndCap_Side_" << side;
        fname2 << "Disc_" << id;
        tag = "_disc_";
    }

    TFileDirectory td1 = fs->mkdir(fname1.str().c_str());
    TFileDirectory td = td1.mkdir(fname2.str().c_str());

    ClusterHistos local_histos;

    ostringstream histoName;

    histoName.str("");
    histoName << "Number_of_Clusters_Pixel" << tag.c_str() <<  id;
    local_histos.NumberOfClusterPixel = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 21, 0., 20.);
    histoName.str("");
    histoName << "Number_of_Clusters_Strip" << tag.c_str() <<  id;
    local_histos.NumberOfClusterStrip = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 21, 0., 20.);

    local_histos.NumberOfClusterPixel->SetFillColor(kBlue);
    local_histos.NumberOfClusterStrip->SetFillColor(kRed);

    histoName.str("");
    histoName << "Number_of_Clusters" << tag.c_str() <<  id;
    local_histos.NumberOfClustersSource = td.make<THStack>(histoName.str().c_str(), histoName.str().c_str());
    local_histos.NumberOfClustersSource->Add(local_histos.NumberOfClusterPixel);
    local_histos.NumberOfClustersSource->Add(local_histos.NumberOfClusterStrip);


    histoName.str("");
    histoName << "Number_of_Clusters_Link" << tag.c_str() <<  id;
    local_histos.NumberOfClustersLink = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 21, 0., 20.);

    // Truth Matching
    for(int iPS=0 ; iPS<nPS ; iPS++) {

      for(int iM=0 ; iM<nTypes ; iM++) {
	histoName.str("");
	histoName << "NumberOfMatchedHits_" << name_PS[iPS] << "_" << name_types[iM] << tag.c_str() <<  id;
	local_histos.NumberOfMatchedHits[iPS][iM] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 21, 0., 20.);

	histoName.str("");
	histoName << "NumberOfMatchedClusters_" << name_PS[iPS] << "_" << name_types[iM] << tag.c_str() <<  id;
	local_histos.NumberOfMatchedClusters[iPS][iM] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 21, 0., 20.);

	histoName.str("");
	histoName << "Efficiency_" << name_PS[iPS] << "_" << name_types[iM] << tag.c_str() <<  id;
	local_histos.hEfficiency[iPS][iM] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 110, 0., 1.1);
      }

      histoName.str("");
      histoName << "DeltaX_simhit_cluster" << "_" << name_PS[iPS] << tag.c_str()  <<  id;
      local_histos.h_dx_Truth[iPS] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 1000, 0., 0.);

      histoName.str("");
      histoName << "DeltaY_simhit_cluster" << "_" << name_PS[iPS] << tag.c_str()  <<  id;
      local_histos.h_dy_Truth[iPS] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 1000, 0., 0.);

    }

    // Cluster topology

    histoName.str("");
    histoName << "ClusterSize" << tag.c_str() <<  id;
    local_histos.clusterSize = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 50, 0., 50.);
    histoName.str("");
    histoName << "ClusterSize_Pixel" << tag.c_str() <<  id;
    local_histos.clusterSizePixel = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 50, 0., 50.);
    histoName.str("");
    histoName << "ClusterSize_Strip" << tag.c_str() <<  id;
    local_histos.clusterSizeStrip = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 50, 0., 50.);

    local_histos.clusterSizePixel->SetFillColor(kBlue);
    local_histos.clusterSizeStrip->SetFillColor(kRed);

    histoName.str("");
    histoName << "Clusters_Size_Source" << tag.c_str() <<  id;
    local_histos.ClustersSizeSource = td.make<THStack>(histoName.str().c_str(), histoName.str().c_str());
    local_histos.ClustersSizeSource->Add(local_histos.clusterSizePixel);
    local_histos.ClustersSizeSource->Add(local_histos.clusterSizeStrip);

    histoName.str("");
    histoName << "ClusterSizeX" << tag.c_str() <<  id;
    local_histos.clusterSizeX = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 1000, 0., 0.);
    histoName.str("");
    histoName << "ClusterSizeY" << tag.c_str() <<  id;
    local_histos.clusterSizeY = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 1000, 0., 0.);

    histoName.str("");
    histoName << "ClusterShapeX" << tag.c_str() <<  id;
    local_histos.clusterShapeX = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 1000, 0., 0.);
    histoName.str("");
    histoName << "ClusterShapeY" << tag.c_str() <<  id;
    local_histos.clusterShapeY = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 1000, 0., 0.);

    histoName.str("");
    histoName << "LocalPositionXY" << tag.c_str() <<  id;
    local_histos.localPosXY = td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0., 2000, 0., 0.);

    histoName.str("");
    histoName << "GlobalPositionXY" << tag.c_str() <<  id;
    local_histos.globalPosXY = td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 2400, -120.0, 120.0, 2400, -120.0, 120.0);
    histoName.str("");
    histoName << "LocalPositionXY_Pixel" << tag.c_str() <<  id;
    local_histos.localPosXYPixel = td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0., 2000, 0., 0.);
    histoName.str("");
    histoName << "LocalPositionXY_Strip" << tag.c_str() <<  id;
    local_histos.localPosXYStrip = td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0., 2000, 0., 0.);

    histoName.str("");
    histoName << "Digi_type_" << tag.c_str() <<  id;
    local_histos.digiType = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 2, 0, 1);
    histoName.str("");
    histoName << "Digi_position_" << tag.c_str() <<  id;
    local_histos.digiPosition = td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0., 2000, 0., 0.);

    histoName.str("");
    histoName << "DeltaX_Cluster_RecHit_" << tag.c_str() <<  id;
    local_histos.dxCluRec = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 1000, 0., 0.);
    histoName.str("");
    histoName << "DeltaY_Cluster_RecHit_" << tag.c_str() <<  id;
    local_histos.dyCluRec = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 1000, 0., 0.);
    histoName.str("");
    histoName << "DeltaR_Cluster_RecHit_" << tag.c_str() <<  id;
    local_histos.dRCluRec = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 1000, 0., 0.);


    layerHistoMap.insert(make_pair(ival, local_histos));
    fs->file().cd("/");
}

void SiPhase2ClustersValidation::createHistograms(unsigned int nLayer) {
    edm::Service<TFileService> fs;
    fs->file().cd("/");
    TFileDirectory td = fs->mkdir("Common");

    trackerLayout_ = td.make<TH2F>("RVsZ", "R vs. z position", 6000, -300.0, 300.0, 1200, 0.0, 120.0);
    trackerLayoutXY_ = td.make<TH2F>("XVsY", "x vs. y position", 2400, -120.0, 120.0, 2400, -120.0, 120.0);
    trackerLayoutXYBar_ = td.make<TH2F>("XVsYBar", "x vs. y position", 2400, -120.0, 120.0, 2400, -120.0, 120.0);
    trackerLayoutXYEC_ = td.make<TH2F>("XVsYEC", "x vs. y position", 2400, -120.0, 120.0, 2400, -120.0, 120.0);
}

//
// -- Get Layer Number
//
unsigned int SiPhase2ClustersValidation::getLayerNumber(const TrackerGeometry* tkgeom,unsigned int& detid) {
  unsigned int layer = 999;
  DetId theDetId(detid);
  if (theDetId.subdetId() != 1)
    std::cout << ">>> Method1 : Det id " << theDetId.det() << " Subdet Id " << theDetId.subdetId() << std::endl;
  const PixelGeomDetUnit * theGeomDet =
      dynamic_cast<const PixelGeomDetUnit*> ( tkgeom->idToDet(theDetId) );

  const GeomDetUnit* it = tkgeom->idToDetUnit(DetId(theDetId));
  if (!it) std::cout << ">>> rawdetid " << detid
                     << " GeomDetUnit " << it
                     << " PixelGeomDetUnit " << theGeomDet
                     << " DetId " << theDetId.det()
                     << " Subdet Id " << theDetId.subdetId()
                     << std::endl;

  if (it && it->type().isTracker()) {
    if (it->type().isBarrel()) {
      PXBDetId pb_detId = PXBDetId(detid);
      layer = pb_detId.layer();
    } else if (it->type().isEndcap()) {
      PXFDetId pf_detId = PXFDetId(detid);
      layer = 100*pf_detId.side() + pf_detId.disk();
    }
  }
  return layer;
}
//
// -- Get Layer Number
//
unsigned int SiPhase2ClustersValidation::getLayerNumber(unsigned int& detid) {
  unsigned int layer = 999;
  DetId theDetId(detid);
  if (theDetId.det() == DetId::Tracker) {
    if (theDetId.subdetId() == PixelSubdetector::PixelBarrel) {
      PXBDetId pb_detId = PXBDetId(detid);
      layer = pb_detId.layer();
    } else if (theDetId.subdetId() == PixelSubdetector::PixelEndcap) {
      PXFDetId pf_detId = PXFDetId(detid);
      layer = 100*pf_detId.side() + pf_detId.disk();
    } else {
      std::cout << ">>> Invalid subdetId() = " << theDetId.subdetId() << std::endl;
    }
  }
  return layer;
}

DEFINE_FWK_MODULE(SiPhase2ClustersValidation);

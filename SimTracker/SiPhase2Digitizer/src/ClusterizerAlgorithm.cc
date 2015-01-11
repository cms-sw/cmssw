#include "SimTracker/SiPhase2Digitizer/interface/ClusterizerAlgorithm.h"
#include "SimTracker/SiPhase2Digitizer/interface/PixelClusterHitArray.h"
//#include "SimTracker/SiPhase2Digitizer/interface/PixelClusterSimLink.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"

#include <stack>
#include <vector>
#include <iostream>

ClusterizerAlgorithm::ClusterizerAlgorithm(edm::ParameterSet const& conf, int maxClusterSize, int maxNumberClusters) : conf_(conf), maxClusterSize_(maxClusterSize), maxNumberClusters_(maxNumberClusters), nrows_(0), ncols_(0), rawDetId_(0) {
    // Create a 2D matrix for this module that represents the hits
    hits.setSize(nrows_, ncols_);
}

// Change the size of the 2D matrix for this module (varies from pixel to strip modules)
void ClusterizerAlgorithm::setup(const PixelGeomDetUnit* pixDet) {
    const PixelTopology & topol = pixDet->specificTopology();
    hits.setSize(topol.nrows(), topol.ncolumns());
    nrows_ = topol.nrows();
    ncols_ = topol.ncolumns();
}

// Go over the Digis and create clusters
void ClusterizerAlgorithm::clusterizeDetUnit(const edm::DetSet<PixelDigi> & pixelDigis, const edm::Handle< edm::DetSetVector< PixelDigiSimLink > > & pixelSimLinks, edmNew::DetSetVector<SiPixelCluster>::FastFiller & clusters) {

    // Get the det ID
    rawDetId_ = pixelDigis.detId();

    // Fill the 2D matrix with the ADC values
    copy_to_buffer(pixelDigis.begin(), pixelDigis.end());

    // Number of clusters
    int numberOfClusters(0);

    // Loop over the Digis
    for (unsigned int row = 0; row < (unsigned int) nrows_; ++row) {
        for (unsigned int col = 0; col < (unsigned int)  ncols_; ++col) {

            // If the Digi is active
            if (hits(row, col)) {

                // Try to form a cluster

                // Create an entry for the pixelClusterLink
                std::vector< unsigned int > simTracks;

                // Add the simtrack of the Digi to the link
                unsigned int simTrackId = getSimTrackId(pixelSimLinks, PixelDigi::pixelToChannel(row, col));
                if (simTrackId) simTracks.push_back(simTrackId);

                // Set the value of this pixel to 0 as it is used to form a Digi
                hits.set(row, col, 0);

                // Create a temporary cluster (this allows to them easily form a "real" cluster with CMSSW data format)
                AccretionCluster acluster;

                // Create a pixel entry for the cluster
                SiPixelCluster::PixelPos firstpix(row, col);

                // Add the first pixel to the cluster
                acluster.add(firstpix, 255);

                // Go over all the pixels in the cluster
                while (!acluster.empty()) {

                    // Get the current pixel we are looking at
                    unsigned int curInd = acluster.top();
                    acluster.pop();

                    unsigned int from_r = acluster.x[curInd] - 1;
                    unsigned int to_r = acluster.x[curInd] + 1;

                    // Look left and right
                    for (unsigned int r = from_r; r <= to_r; ++r) {

                        // Look bottom and top
                        if (hits(r, col)) {

                            // Add it to the cluster
                            SiPixelCluster::PixelPos newpix(r, col);
                            if (!acluster.add(newpix, 255)) break;

                            // And change its value
                            hits.set(newpix, 0);

                            // Add the simtrack of the Digi to the link
                            unsigned int simTrackId2 = getSimTrackId(pixelSimLinks, PixelDigi::pixelToChannel(r, col));
                            if (simTrackId2) simTracks.push_back(simTrackId2);
                                        
                            if (acluster.isize == (unsigned int) maxClusterSize_) goto form_cluster;
                        }
                    }
                }

                form_cluster: 

                numberOfClusters++;

                // Check if we hit the maximym number of clusters per module
                if (maxNumberClusters_ != -1 and numberOfClusters > maxNumberClusters_) return;

                // Form a "real" CMSSW cluster
                SiPixelCluster cluster(acluster.isize, acluster.adc, acluster.x, acluster.y, acluster.xmin, acluster.ymin);

                // Add link
                tmpSimLinks.insert(std::pair< SiPixelCluster, std::vector< unsigned int > >(cluster, simTracks));

                // Add the cluster and the link to the list
                clusters.push_back(cluster);
            }
        }
    }

    // Reset the matrix
    clear_buffer(pixelDigis.begin(), pixelDigis.end());
}

void ClusterizerAlgorithm::copy_to_buffer(DigiIterator begin, DigiIterator end) {
    // Copy the value of the Digis' ADC to the 2D matrix. An ADC of 255 means the cell is hit (binary read-out)
    for (DigiIterator di = begin; di != end; ++di) hits.set(di->row(), di->column(), di->adc());
}

void ClusterizerAlgorithm::clear_buffer(DigiIterator begin, DigiIterator end) {
    // Resets the matrix
    for (DigiIterator di = begin; di != end; ++di) hits.set(di->row(), di->column(), 0);
}

// Returns the simtrack id of a digi
unsigned int ClusterizerAlgorithm::getSimTrackId(const edm::Handle< edm::DetSetVector< PixelDigiSimLink > > & pixelSimLinks, int channel) {
    edm::DetSetVector<PixelDigiSimLink>::const_iterator isearch = pixelSimLinks->find(rawDetId_);

    unsigned int simTrkId(0);
    if (isearch == pixelSimLinks->end()) return simTrkId;

    edm::DetSet<PixelDigiSimLink> link_detset = (*pixelSimLinks)[rawDetId_];
    int iSimLink = 0;
    for (edm::DetSet<PixelDigiSimLink>::const_iterator it = link_detset.data.begin(); it != link_detset.data.end(); it++,iSimLink++) {
        if (channel == (int) it->channel()) {
            simTrkId = it->SimTrackId();
            break;
        }
    }
    return simTrkId;
}

// Create the links between the clusters and the simTracks
/*void ClusterizerAlgorithm::makeLinks(edm::OrphanHandle< edmNew::DetSetVector<SiPixelCluster> > & clusters, std::vector<edm::DetSet<PixelClusterSimLink> > & linksByDet) {

    // Go over all the clusters
    for (edmNew::DetSetVector< SiPixelCluster >::const_iterator DSViter = clusters->begin(); DSViter != clusters->end(); ++DSViter) {

        unsigned int nclu = 0;

        edm::DetSet< PixelClusterSimLink > links(DSViter->id());

        for (edmNew::DetSet< SiPixelCluster >::const_iterator clustIt = DSViter->begin(); clustIt != DSViter->end(); ++clustIt) {

            // Add a cut of 10 links per module
    	    if (nclu++ > 10) continue;

            // Look for the cluster in the existing links
            std::map< SiPixelCluster, std::vector< unsigned int > >::iterator it = tmpSimLinks.find(*clustIt);

            if (it != tmpSimLinks.end()) {
                // Match the cluster with the simtrack
                std::vector< unsigned int > simTracks = it->second;

                PixelClusterSimLink simLink;

                simLink.setCluster(edmNew::makeRefTo(clusters, clustIt));
                simLink.setSimTracks(simTracks);
                links.data.push_back(simLink);
            }
        }

        linksByDet.push_back(links);
    }
}
*/

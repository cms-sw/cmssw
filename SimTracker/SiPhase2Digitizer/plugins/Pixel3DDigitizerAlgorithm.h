#ifndef _SimTracker_SiPhase2Digitizer_Pixel3DDigitizerAlgorithm_h
#define _SimTracker_SiPhase2Digitizer_Pixel3DDigitizerAlgorithm_h

//-------------------------------------------------------------
// class Pixel3DDigitizerAlgorithm
//
// Specialization of the tracker digitizer for the 3D pixel 
// sensors placed at PXB-Layer1 (and possibly Layer2), and
// at PXF-Disk1 and Disk2
//
// Authors: Jordi Duarte-Campderros (CERN/IFCA)
//          Clara Lasaosa Garcia (IFCA)
//--------------------------------------------------------------

#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerAlgorithm.h"

class Pixel3DDigitizerAlgorithm : public Phase2TrackerDigitizerAlgorithm 
{
    public:
        Pixel3DDigitizerAlgorithm(const edm::ParameterSet& conf);
        ~Pixel3DDigitizerAlgorithm() override;

        // initialization that cannot be done in the constructor
        void init(const edm::EventSetup& es) override;

        // run the algorithm to digitize a single det
        void accumulateSimHits(const std::vector<PSimHit>::const_iterator inputBegin,
                         const std::vector<PSimHit>::const_iterator inputEnd,
                         const size_t inputBeginGlobalIndex,
                         const unsigned int tofBin,
                         const Phase2TrackerGeomDetUnit* pixdet,
                         const GlobalVector& bfield) override;
};
#endif

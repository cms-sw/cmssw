#ifndef BDHadronTrackMonitoringHarvester_H
#define BDHadronTrackMonitoringHarvester_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

/** \class BDHadronTrackMonitoringHarvester
 *
 *  Top level steering routine for b tag performance harvesting.
 *
 */

class BDHadronTrackMonitoringHarvester : public DQMEDHarvester {
    public:
        explicit BDHadronTrackMonitoringHarvester(const edm::ParameterSet& pSet);
        ~BDHadronTrackMonitoringHarvester();

    private:
        void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;
        void beginJob();


        // Histograms
        // b jets
        MonitorElement *nTrk_absolute_bjet, *nTrk_relative_bjet, *nTrk_std_bjet; // number of selected tracks (absolute or in percents) in different track history categories for b jets

        // c jets
        MonitorElement *nTrk_absolute_cjet, *nTrk_relative_cjet, *nTrk_std_cjet; // number of selected tracks (absolute or in percents) in different track history categories for c jets


        // dusg jets
        MonitorElement *nTrk_absolute_dusgjet, *nTrk_relative_dusgjet, *nTrk_std_dusgjet; // number of selected tracks (absolute or in percents) in different track history categories for light jets


};

#endif

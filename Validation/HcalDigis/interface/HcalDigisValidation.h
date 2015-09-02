#ifndef HCALDIGISVALIDATION_H
#define	HCALDIGISVALIDATION_H

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"


#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

/*TP Code*/
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
/*~TP Code*/


#include <map>
#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>
#include <iostream>

class HcalDigisValidation : public DQMEDAnalyzer {
public:
    explicit HcalDigisValidation(const edm::ParameterSet&);

    ~HcalDigisValidation(); 

    virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &);

private:

    struct HistLim {

        HistLim(int nbin, double mini, double maxi)
        : n(nbin), min(mini), max(maxi) {
        }
        int n;
        double min;
        double max;
    };

    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    std::map<std::string, MonitorElement*> *msm_;

    void book1D(DQMStore::IBooker &ib, std::string name, int n, double min, double max);

    void book1D(DQMStore::IBooker &ib, std::string name, const HistLim& limX);

    void fill1D(std::string name, double X, double weight = 1);

    void book2D(DQMStore::IBooker &ib, std::string name, const HistLim& limX, const HistLim& limY);

    void fill2D(std::string name, double X, double Y, double weight = 1);

    void bookPf(DQMStore::IBooker &ib, std::string name, const HistLim& limX, const HistLim& limY);

    void bookPf(DQMStore::IBooker &ib, std::string name, const HistLim& limX, const HistLim& limY, const char *option);

    void fillPf(std::string name, double X, double Y);

    MonitorElement* monitor(std::string name);

    void booking(DQMStore::IBooker &ib, std::string subdetopt, int bnoise, int bmc);

    std::string str(int x);

    template<class Digi> void reco(const edm::Event& iEvent, const edm::EventSetup& iSetup, const edm::EDGetTokenT<edm::SortedCollection<Digi> > &tok);
    void eval_occupancy();

    std::string outputFile_;
    std::string subdet_;
    std::string zside_;
    std::string dirName_;
    edm::InputTag inputTag_;
    std::string mode_;
    std::string mc_;
    int noise_;

    edm::EDGetTokenT<edm::PCaloHitContainer> tok_mc_;
    edm::EDGetTokenT<edm::SortedCollection<HBHEDataFrame> > tok_hbhe_; 
    edm::EDGetTokenT<edm::SortedCollection<HODataFrame> > tok_ho_;
    edm::EDGetTokenT<edm::SortedCollection<HFDataFrame> > tok_hf_;
    edm::EDGetTokenT<HcalTrigPrimDigiCollection> tok_emulTPs_;
    edm::EDGetTokenT<HcalTrigPrimDigiCollection> tok_dataTPs_;

    edm::ESHandle<CaloGeometry> geometry;
    edm::ESHandle<HcalDbService> conditions;

    //TP Code
    edm::ESHandle<HcalTopology> htopo;
    //~TP Code

    int nevent1;
    int nevent2;
    int nevent3;
    int nevent4;
    int nevtot;

};

#endif



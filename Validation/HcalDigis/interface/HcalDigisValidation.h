#ifndef HCALDIGISVALIDATION_H
#define	HCALDIGISVALIDATION_H

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

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



#include <map>
#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>
#include <iostream>

class HcalDigisValidation : public edm::EDAnalyzer {
public:
    explicit HcalDigisValidation(const edm::ParameterSet&);

    ~HcalDigisValidation() {
    };

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

    virtual void beginJob();

    virtual void endJob();

    void beginRun();

    void endRun();

    DQMStore* dbe_;
    std::map<std::string, MonitorElement*> *msm_;

    void book1D(std::string name, int n, double min, double max);

    void book1D(std::string name, HistLim limX);

    void fill1D(std::string name, double X, double weight = 1);

    void book2D(std::string name, HistLim limX, HistLim limY);

    void fill2D(std::string name, double X, double Y, double weight = 1);

    void bookPf(std::string name, HistLim limX, HistLim limY);

    void fillPf(std::string name, double X, double Y);

    MonitorElement* monitor(std::string name);

    void booking(std::string subdetopt, int bnoise, int bmc);

    std::string str(int x);

    template<class Digi> void reco(const edm::Event& iEvent, const edm::EventSetup& iSetup);
    void eval_occupancy();

    std::string outputFile_;
    std::string subdet_;
    std::string zside_;
    std::string dirName_;
    edm::InputTag inputTag_;
    std::string mode_;
    std::string mc_;
    int noise_;
    bool doSLHC_;

    // specifically for SLHC    
    edm::InputTag inputTag_HBHE;  
    edm::InputTag inputTag_HF;


    edm::ESHandle<CaloGeometry> geometry;
    edm::ESHandle<HcalDbService> conditions;
    int nevent1;
    int nevent2;
    int nevent3;
    int nevent4;
    int nevtot;

};

#endif



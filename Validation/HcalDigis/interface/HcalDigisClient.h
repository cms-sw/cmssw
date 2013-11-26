/* 
 * File:   HcalDigisClient.h
 * Author: fahim
 *
 * Created on June 11, 2011, 6:38 PM
 */

#ifndef HCALDIGISCLIENT_H
#define	HCALDIGISCLIENT_H

#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class HcalDigisClient : public edm::EDAnalyzer {
public:
    explicit HcalDigisClient(const edm::ParameterSet&);

    ~HcalDigisClient() {
    };

private:

    virtual void beginJob() {
    };
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    virtual void endJob() {
        if (outputFile_.size() != 0 && dbe_) dbe_->save(outputFile_);

    };

    virtual void beginRun(edm::Run const&, edm::EventSetup const&) {
    };

    virtual void endRun(edm::Run const&, edm::EventSetup const&) {

        if (dbe_) dbe_->setCurrentFolder(dirName_);
        runClient();
    };

    struct HistLim {

        HistLim(int nbin, double mini, double maxi)
        : n(nbin), min(mini), max(maxi) {
        }
        int n;
        double min;
        double max;
    };

    bool doSLHC_;

    virtual void runClient();
    int HcalDigisEndjob(const std::vector<MonitorElement*> &hcalMEs, std::string subdet_);

    MonitorElement* monitor(std::string name);

    void book1D(std::string name, int n, double min, double max) {
        if (!msm_->count(name)) (*msm_)[name] = dbe_->book1D(name.c_str(), name.c_str(), n, min, max);
    }

    void book1D(std::string name, HistLim limX) {
        if (!msm_->count(name)) (*msm_)[name] = dbe_->book1D(name.c_str(), name.c_str(), limX.n, limX.min, limX.max);
    }

    void fill1D(std::string name, double X, double weight = 1) {
        msm_->find(name)->second->Fill(X, weight);
    }

    void book2D(std::string name, HistLim limX, HistLim limY) {
        if (!msm_->count(name)) (*msm_)[name] = dbe_->book2D(name.c_str(), name.c_str(), limX.n, limX.min, limX.max, limY.n, limY.min, limY.max);
    }

    void fill2D(std::string name, double X, double Y, double weight = 1) {
        msm_->find(name)->second->Fill(X, Y, weight);
    }

    void bookPf(std::string name, HistLim limX, HistLim limY) {
        if (!msm_->count(name)) (*msm_)[name] = dbe_->bookProfile(name.c_str(), name.c_str(), limX.n, limX.min, limX.max, limY.n, limY.min, limY.max);
    }

    void fillPf(std::string name, double X, double Y) {
        msm_->find(name)->second->Fill(X, Y);
    }

    void booking(std::string subdetopt);

    std::string str(int x);

    double integralMETH2D(MonitorElement* ME, int i0, int i1, int j0, int j1);
    void scaleMETH2D(MonitorElement* ME, double s);
    std::map<std::string, MonitorElement*> *msm_;
    DQMStore* dbe_;
    std::string outputFile_;
    std::string dirName_;
};



#endif	/* HCALDIGISCLIENT_H */


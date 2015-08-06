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
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class HcalDigisClient : public DQMEDHarvester {
public:
    explicit HcalDigisClient(const edm::ParameterSet&);

    ~HcalDigisClient();

private:

    virtual void dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter & igetter ) {
      igetter.setCurrentFolder("HcalDigisV/HcalDigiTask"); // moved this line from constructor

      // the following booking clas were moved from the constructor
      booking(ibooker, "HB");
      booking(ibooker, "HE");
      booking(ibooker, "HO");
      booking(ibooker, "HF");

      igetter.setCurrentFolder(dirName_); // This sets the DQMStore (should apply to ibooker as well
      runClient(ibooker, igetter);
    }

    struct HistLim {

        HistLim(int nbin, double mini, double maxi)
        : n(nbin), min(mini), max(maxi) {
        }
        int n;
        double min;
        double max;
    };

    virtual void runClient(DQMStore::IBooker &ib, DQMStore::IGetter &ig);
    int HcalDigisEndjob(const std::vector<MonitorElement*> &hcalMEs, std::string subdet_);

    MonitorElement* monitor(std::string name);

    void book1D(DQMStore::IBooker &ib, std::string name, int n, double min, double max) {
        if (!msm_->count(name)) (*msm_)[name] = ib.book1D(name.c_str(), name.c_str(), n, min, max);
    }

    void book1D(DQMStore::IBooker &ib, std::string name, const HistLim& limX) {
        if (!msm_->count(name)) (*msm_)[name] = ib.book1D(name.c_str(), name.c_str(), limX.n, limX.min, limX.max);
    }

    void fill1D(std::string name, double X, double weight = 1) {
        msm_->find(name)->second->Fill(X, weight);
    }

    void book2D(DQMStore::IBooker &ib, std::string name, const HistLim& limX, const HistLim& limY) {
        if (!msm_->count(name)) (*msm_)[name] = ib.book2D(name.c_str(), name.c_str(), limX.n, limX.min, limX.max, limY.n, limY.min, limY.max);
    }

    void fill2D(std::string name, double X, double Y, double weight = 1) {
        msm_->find(name)->second->Fill(X, Y, weight);
    }

    void bookPf(DQMStore::IBooker &ib, std::string name, const HistLim& limX, const HistLim& limY) {
        if (!msm_->count(name)) (*msm_)[name] = ib.bookProfile(name.c_str(), name.c_str(), limX.n, limX.min, limX.max, limY.n, limY.min, limY.max);
    }

    void bookPf(DQMStore::IBooker &ib, std::string name, const HistLim& limX, const HistLim& limY, const char *option) {
        if (!msm_->count(name)) (*msm_)[name] = ib.bookProfile(name.c_str(), name.c_str(), limX.n, limX.min, limX.max, limY.n, limY.min, limY.max, option);
    }

    void fillPf(std::string name, double X, double Y) {
        msm_->find(name)->second->Fill(X, Y);
    }

    void booking(DQMStore::IBooker &ib, std::string subdetopt);

    std::string str(int x);

    double integralMETH2D(MonitorElement* ME, int i0, int i1, int j0, int j1);
    void scaleMETH2D(MonitorElement* ME, double s);
    std::map<std::string, MonitorElement*> *msm_;
    std::string outputFile_;
    std::string dirName_;
};



#endif	/* HCALDIGISCLIENT_H */


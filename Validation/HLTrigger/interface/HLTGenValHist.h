#ifndef Validation_HLTrigger_HLTGenValHist_h
#define Validation_HLTrigger_HLTGenValHist_h

//********************************************************************************
//
// Description:
//   Histogram holder class for the GEN-level HLT validation
//   Handling and filling of 1D and 2D histograms is done by this class.
//
//
// Author : Finn Labe, UHH, Jul. 2022
//          (Strongly inspired by Sam Harpers HLTDQMHist class)
//
//***********************************************************************************

#include "DQMOffline/Trigger/interface/FunctionDefs.h"

#include "DQMOffline/Trigger/interface/VarRangeCutColl.h"

#include "FWCore/Framework/interface/Event.h"

#include "Validation/HLTrigger/interface/HLTGenValObject.h"

#include <TH1.h>
#include <TH2.h>

// base histogram class, with specific implementations following below
class HLTGenValHist {
public:
  HLTGenValHist() = default;
  virtual ~HLTGenValHist() = default;
  virtual void fill(const HLTGenValObject& objType) = 0;
};

// specific implimentation of a HLTGenValHist for 1D histograms
// it takes the histogram which it will fill
// it takes the variable to plot (func) and its name (varName)
// also, it takes additional cuts (rangeCuts) applied before filling
// to fill the histogram, an object is passed in the Fill function
class HLTGenValHist1D : public HLTGenValHist {
public:
  HLTGenValHist1D(TH1* hist,
                  std::string varName,
                  std::function<float(const HLTGenValObject&)> func,
                  VarRangeCutColl<HLTGenValObject> rangeCuts)
      : var_(std::move(func)), varName_(std::move(varName)), rangeCuts_(std::move(rangeCuts)), hist_(hist) {}

  void fill(const HLTGenValObject& obj) override {
    if (rangeCuts_(obj))
      hist_->Fill(var_(obj));
  }

private:
  std::function<float(const HLTGenValObject&)> var_;
  std::string varName_;
  VarRangeCutColl<HLTGenValObject> rangeCuts_;
  TH1* hist_;  //we do not own this
};

// specific implimentation of a HLTGenValHist for 2D histograms
// it takes the histogram which it will fill
// it takes the two variable to plot (func) and their name (varName)
// to fill the histogram, two objects are passed in the Fill function
class HLTGenValHist2D : public HLTGenValHist {
public:
  HLTGenValHist2D(TH2* hist,
                  std::string varNameX,
                  std::string varNameY,
                  std::function<float(const HLTGenValObject&)> funcX,
                  std::function<float(const HLTGenValObject&)> funcY)
      : varX_(std::move(funcX)),
        varY_(std::move(funcY)),
        varNameX_(std::move(varNameX)),
        varNameY_(std::move(varNameY)),
        hist_(hist) {}

  void fill(const HLTGenValObject& obj) override { hist_->Fill(varX_(obj), varY_(obj)); }

private:
  std::function<float(const HLTGenValObject&)> varX_;
  std::function<float(const HLTGenValObject&)> varY_;
  std::string varNameX_;
  std::string varNameY_;
  TH2* hist_;  //we do not own this
};

#endif

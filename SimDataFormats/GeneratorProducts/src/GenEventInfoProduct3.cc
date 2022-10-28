#include <functional>
#include <numeric>
using std::ptrdiff_t;

#include <HepMC3/GenEvent.h>
// #include <HepMC3/WeightContainer.h>
#include <HepMC3/GenPdfInfo.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct3.h"

using namespace edm;
using namespace std;

GenEventInfoProduct3::GenEventInfoProduct3()
    : signalProcessID_(0), qScale_(-1.), alphaQCD_(-1.), alphaQED_(-1.), nMEPartons_(-1), nMEPartonsFiltered_(-1) {}

GenEventInfoProduct3::GenEventInfoProduct3(const HepMC3::GenEvent *evt)
    : weights_(evt->weights().begin(), evt->weights().end()), nMEPartons_(-1), nMEPartonsFiltered_(-1) {
  std::shared_ptr<HepMC3::IntAttribute> A_signal_process_id = evt->attribute<HepMC3::IntAttribute>("signal_process_id");
  std::shared_ptr<HepMC3::DoubleAttribute> A_event_scale = evt->attribute<HepMC3::DoubleAttribute>("event_scale");
  std::shared_ptr<HepMC3::DoubleAttribute> A_alphaQCD = evt->attribute<HepMC3::DoubleAttribute>("alphaQCD");
  std::shared_ptr<HepMC3::DoubleAttribute> A_alphaQED = evt->attribute<HepMC3::DoubleAttribute>("alphaQED");
  //std::shared_ptr<HepMC3::IntAttribute> A_mpi = evt->attribute<HepMC3::IntAttribute>("mpi");

  signalProcessID_ = A_signal_process_id ? (A_signal_process_id->value()) : 0;
  qScale_ = A_event_scale ? (A_event_scale->value()) : 0.0;
  alphaQCD_ = A_alphaQCD ? (A_alphaQCD->value()) : 0.0;
  alphaQED_ = A_alphaQED ? (A_alphaQED->value()) : 0.0;

  std::shared_ptr<HepMC3::GenPdfInfo> A_pdf = evt->attribute<HepMC3::GenPdfInfo>("GenPdfInfo");
  if (A_pdf) {
    PDF pdf;
    pdf.id = std::make_pair(A_pdf->parton_id[0], A_pdf->parton_id[1]);
    pdf.x = std::make_pair(A_pdf->x[0], A_pdf->x[1]);
    pdf.xPDF = std::make_pair(A_pdf->xf[0], A_pdf->xf[1]);
    pdf.scalePDF = A_pdf->scale;
    setPDF(&pdf);
  }
}

GenEventInfoProduct3::GenEventInfoProduct3(GenEventInfoProduct3 const &other)
    : weights_(other.weights_),
      signalProcessID_(other.signalProcessID_),
      qScale_(other.qScale_),
      alphaQCD_(other.alphaQCD_),
      alphaQED_(other.alphaQED_),
      binningValues_(other.binningValues_),
      DJRValues_(other.DJRValues_),
      nMEPartons_(other.nMEPartons_),
      nMEPartonsFiltered_(other.nMEPartons_) {
  setPDF(other.pdf());
}

GenEventInfoProduct3::GenEventInfoProduct3(GenEventInfoProduct3 &&other)
    : weights_(std::move(other.weights_)),
      signalProcessID_(other.signalProcessID_),
      qScale_(other.qScale_),
      alphaQCD_(other.alphaQCD_),
      alphaQED_(other.alphaQED_),
      pdf_(other.pdf_.release()),
      binningValues_(std::move(other.binningValues_)),
      DJRValues_(std::move(other.DJRValues_)),
      nMEPartons_(other.nMEPartons_),
      nMEPartonsFiltered_(other.nMEPartons_) {}

GenEventInfoProduct3::~GenEventInfoProduct3() {}

GenEventInfoProduct3 &GenEventInfoProduct3::operator=(GenEventInfoProduct3 const &other) {
  weights_ = other.weights_;
  signalProcessID_ = other.signalProcessID_;
  qScale_ = other.qScale_;
  alphaQCD_ = other.alphaQCD_;
  alphaQED_ = other.alphaQED_;
  binningValues_ = other.binningValues_;
  DJRValues_ = other.DJRValues_;
  nMEPartons_ = other.nMEPartons_;
  nMEPartonsFiltered_ = other.nMEPartonsFiltered_;

  setPDF(other.pdf());

  return *this;
}

GenEventInfoProduct3 &GenEventInfoProduct3::operator=(GenEventInfoProduct3 &&other) {
  weights_ = std::move(other.weights_);
  signalProcessID_ = other.signalProcessID_;
  qScale_ = other.qScale_;
  alphaQCD_ = other.alphaQCD_;
  alphaQED_ = other.alphaQED_;
  binningValues_ = std::move(other.binningValues_);
  DJRValues_ = std::move(other.DJRValues_);
  nMEPartons_ = other.nMEPartons_;
  nMEPartonsFiltered_ = other.nMEPartonsFiltered_;
  pdf_ = std::move(other.pdf_);

  return *this;
}

double GenEventInfoProduct3::weightProduct() const {
  return std::accumulate(weights_.begin(), weights_.end(), 1., std::multiplies<double>());
}

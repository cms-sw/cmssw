#include <functional>
#include <numeric>

#include <HepMC/GenEvent.h>
#include <HepMC/WeightContainer.h>
#include <HepMC/PdfInfo.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

using namespace edm;
using namespace std;

GenEventInfoProduct::GenEventInfoProduct() :
	signalProcessID_(0), qScale_(-1.), alphaQCD_(-1.), alphaQED_(-1.)
{
}

GenEventInfoProduct::GenEventInfoProduct(const HepMC::GenEvent *evt) :
	weights_(evt->weights().begin(), evt->weights().end()),
	signalProcessID_(evt->signal_process_id()),
	qScale_(evt->event_scale()),
	alphaQCD_(evt->alphaQCD()),
	alphaQED_(evt->alphaQED())
{
	const HepMC::PdfInfo *hepPDF = evt->pdf_info();    
	if (hepPDF) {
		PDF pdf;

		pdf.id = std::make_pair(hepPDF->id1(), hepPDF->id2());
		pdf.x = std::make_pair(hepPDF->x1(), hepPDF->x2());
		pdf.xPDF = std::make_pair(hepPDF->pdf1(), hepPDF->pdf2());   
		pdf.scalePDF = hepPDF->scalePDF();

		setPDF(&pdf);
	}
}

GenEventInfoProduct::GenEventInfoProduct(GenEventInfoProduct const &other) :
	weights_(other.weights_),
	signalProcessID_(other.signalProcessID_),
	qScale_(other.qScale_),
	alphaQCD_(other.alphaQCD_),
	alphaQED_(other.alphaQED_),
	binningValues_(other.binningValues_)
{
	setPDF(other.pdf());
}

GenEventInfoProduct::~GenEventInfoProduct()
{
}

GenEventInfoProduct &GenEventInfoProduct::operator = (GenEventInfoProduct const &other)
{
	weights_ = other.weights_;
	signalProcessID_ = other.signalProcessID_;
	qScale_ = other.qScale_;
	alphaQCD_ = other.alphaQCD_;
	alphaQED_ = other.alphaQED_;
	binningValues_ = other.binningValues_;

	setPDF(other.pdf());

	return *this;
}

double GenEventInfoProduct::weight() const
{
	return std::accumulate(weights_.begin(), weights_.end(),
	                       1., std::multiplies<double>());
}

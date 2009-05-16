#ifndef SimDataFormats_GeneratorProducts_GenEventInfoProduct_h
#define SimDataFormats_GeneratorProducts_GenEventInfoProduct_h

#include <vector>
#include <memory>

#include "SimDataFormats/GeneratorProducts/interface/PdfInfo.h"

namespace HepMC { class GenEvent; } // forward declaration

/** \class GenEventInfoProduct
 *
 */

class GenEventInfoProduct {
    public:
	GenEventInfoProduct();
	GenEventInfoProduct(const HepMC::GenEvent *evt);
	GenEventInfoProduct(const GenEventInfoProduct &other);
	virtual ~GenEventInfoProduct();

	GenEventInfoProduct &operator = (const GenEventInfoProduct &other);

	typedef gen::PdfInfo PDF;

	// getters

	const std::vector<double> &weights() const { return weights_; }
	double weight() const;

	unsigned int signalProcessID() const { return signalProcessID_; }

	double qScale() const { return qScale_; }
	double alphaQCD() const { return alphaQCD_; }
	double alphaQED() const { return alphaQED_; }

	const PDF *pdf() const { return pdf_.get(); }
	bool hasPDF() const { return pdf() != 0; }

	const std::vector<double> &binningValues() const { return binningValues_; }
	bool hasBinningValues() const { return !binningValues_.empty(); }

	// setters

	void setWeights(const std::vector<double> &weights)
	{ weights_ = weights; }

	void setSignalProcessID(unsigned int procID)
	{ signalProcessID_  = procID; }

	void setScales(double q = -1., double qcd = -1., double qed = -1.)
	{ qScale_ = q, alphaQCD_ = qcd, alphaQED_ = qed; }

	void setPDF(const PDF *pdf) { pdf_.reset(pdf ? new PDF(*pdf) : 0); }

	void setBinningValues(const std::vector<double> &values)
	{ binningValues_ = values; }

    private:
	// HepMC::GenEvent provides a list of weights
	std::vector<double>	weights_;

	// generator-dependent process ID
	unsigned int		signalProcessID_;

	// information about scales
	double			qScale_;
	double			alphaQCD_, alphaQED_;

	// optional PDF info
	std::auto_ptr<PDF>	pdf_;

	// If event was produced in bis, this contains
	// the values that were used to define which
	// bin the event belongs in
	// This replaces the genEventScale, which only
	// corresponds to Pythia pthat.  The RunInfo
	// will contain the information what physical
	// quantity these values actually belong to
	std::vector<double>	binningValues_;
};

#endif // SimDataFormats_GeneratorProducts_GenEventInfoProduct_h

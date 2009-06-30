#ifndef SimDataFormats_GeneratorProducts_GenRunInfoProduct_h
#define SimDataFormats_GeneratorProducts_GenRunInfoProduct_h

/** \class GenRunInfoProduct
 *
 */

class GenRunInfoProduct {
    public:
	// a few forward declarations
	struct XSec;

	// constructors, destructors
	GenRunInfoProduct();
	GenRunInfoProduct(const GenRunInfoProduct &other);
	virtual ~GenRunInfoProduct();

	// getters

	const XSec &internalXSec() const { return internalXSec_; }
	const XSec &externalXSecLO() const { return externalXSecLO_; }
	const XSec &externalXSecNLO() const { return externalXSecNLO_; }
	double filterEfficiency() const { return externalFilterEfficiency_; }

	// setters

	void setInternalXSec(const XSec &xsec) { internalXSec_ = xsec; }
	void setExternalXSecLO(const XSec &xsec) { externalXSecLO_ = xsec; }
	void setExternalXSecNLO(const XSec &xsec) { externalXSecNLO_ = xsec; }
	void setFilterEfficiency(double effic) { externalFilterEfficiency_ = effic; }

#ifndef __GCCXML__
	// compatibility with GenInfoProduct
	// will go away with the removal of GenInfoProduct

	void set_cross_section(double xsec) { warn(__PRETTY_FUNCTION__); setInternalXSec(XSec(xsec)); } __attribute__ ((deprecated))
	void set_external_cross_section(double xsec) { warn(__PRETTY_FUNCTION__); setExternalXSecLO(XSec(xsec)); } __attribute__ ((deprecated))
	void set_filter_efficiency(double eff) { warn(__PRETTY_FUNCTION__); setFilterEfficiency(eff); } __attribute__ ((deprecated))

	double cross_section() const { warn(__PRETTY_FUNCTION__); return internalXSec().value(); } __attribute__ ((deprecated))
	double external_cross_section() const { warn(__PRETTY_FUNCTION__); return externalXSecLO().value() ; } __attribute__ ((deprecated))
	double filter_efficiency() const { warn(__PRETTY_FUNCTION__); return filterEfficiency(); } __attribute__ ((deprecated))
#endif

	// struct definitions
	struct XSec {
	    public:
		XSec() : value_(-1.), error_(-1.) {}
		XSec(double value, double error = -1.) :
			value_(value), error_(error) {}
		XSec(const XSec &other) :
			value_(other.value_), error_(other.error_) {}

		double value() const { return value_; }
		double error() const { return error_; }

		bool isSet() const { return value_ >= 0.; }
		bool hasError() const { return error_ >= 0.; }

		operator bool() const { return isSet(); }

		bool operator == (const XSec &other) const
		{ return value_ == other.value_ && error_ == other.error_; }
		bool operator != (const XSec &other) const { return !(*this == other); }

	    private:
		double value_, error_;
	};

	// convenience (return the value, prefer externally specified over internal one)
	double crossSection() const
	{ return externalXSecLO_ ? externalXSecLO_.value() : internalXSec_.value(); }

	// methods used by EDM
	virtual bool mergeProduct(const GenRunInfoProduct &other);
	virtual bool isProductEqual(const GenRunInfoProduct &other) const;

    private:
#ifndef __GCCXML__
	void warn(const char*) const;
#endif

	// cross sections
	XSec	internalXSec_;	// the one computed during cmsRun
	XSec	externalXSecLO_, externalXSecNLO_;	// from config file
	double	externalFilterEfficiency_;		// from config file
};

#endif // SimDataFormats_GeneratorProducts_GenRunInfoProduct_h

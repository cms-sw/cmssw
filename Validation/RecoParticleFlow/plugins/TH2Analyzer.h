#ifndef RecoParticleFlow_PFRootEvent_TH2Analyzer_h
#define RecoParticleFlow_PFRootEvent_TH2Analyzer_h

#include "TH1.h"
#include "TH2.h"
#include "TF1.h"

class TH2Analyzer: public TH2{
 public:

  TH2Analyzer();
  ~TH2Analyzer();
  void MeanSlice(TH1D*, const unsigned int binxmin, const unsigned int binxmax,
		 const unsigned int nbin, const std::string binning_option) const;
  void SigmaSlice(TH1D*, const unsigned int binxmin, const unsigned int binxmax,
		 const unsigned int nbin, const std::string binning_option) const;
  void MeanGaussSlice(TH1D*, const unsigned int binxmin, const unsigned int binxmax,
		      const unsigned int nbin, const std::string binning_option,
		      const int range, const double fitmin, const double fitmax,
		      const std::string fit_plot_name) const;
  void SigmaGaussSlice(TH1D*, const unsigned int binxmin, const unsigned int binxmax,
		       const unsigned int nbin, const std::string binning_option,
		       const int range, const double fitmin, const double fitmax,
		       const std::string fit_plot_name) const;
  void MeanXSlice(TH1D*, const unsigned int binxmin, const unsigned int binxmax,
		 const unsigned int nbin, const std::string binning_option) const;

  private:
  void checkBinning() const;
  // determine the array of low-edges for each bin (xlow)
  void binning_computation(double* xlow, const unsigned int nbin, const unsigned int binxmin,
			      const unsigned int binxmax, const std::string binning_option) const;
  void XSlice(const std::string Xvar, TH1D*, const unsigned int binxmin, const unsigned int binxmax,
	      const unsigned int nbin, const std::string binning_option, const std::string gauss,
	      const int range, const double fitmin, const double fitmax,
	      const std::string fit_plot_name) const;

  ClassDef(TH2Analyzer,1)
};

#endif

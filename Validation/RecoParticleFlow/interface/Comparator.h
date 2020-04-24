#ifndef __Validation_RecoParticleFlow_Comparator__
#define __Validation_RecoParticleFlow_Comparator__

#include <math.h>

#include <TLegend.h>
#include <TFile.h>
#include <TH1.h>
#include <TF1.h>

/* #include <string> */

class Style;

class Comparator {

public:

  enum Mode {
    NORMAL,
    SCALE,
    RATIO,
    GRAPH,
    EFF
  };

  Comparator() : rebin_(-1), xMin_(0), xMax_(0), resetAxis_(false), 
		 s0_(0), s1_(0), legend_(0,0,1,1) {}

  Comparator( const char* file0,
	      const char* dir0,
	      const char* file1,
	      const char* dir1 ) : 
    rebin_(-1), xMin_(0), xMax_(0), resetAxis_(false), 
    s0_(0), s1_(0), legend_(0,0,1,1) {
    
    SetDirs( file0, dir0, file1, dir1);
  }
  
  /// set the 2 files, and the directory within each file, in which the histograms will be compared
  void SetDirs( const char* file0,
		const char* dir0,
		const char* file1,
		const char* dir1  );

  // set the rebinning factor and the range
  void SetAxis( int rebin,
		float xmin, 
		float xmax) {
    rebin_ = rebin;
    xMin_ = xmin;
    xMax_ = xmax;
    resetAxis_ = true;
  }
  
  // set the rebinning factor, unset the range
  void SetAxis( int rebin ) {
    rebin_ = rebin;
    resetAxis_ = false;
  }

  // draws a Y projection of a slice along X
  void DrawSlice( const char* key, 
		  int binxmin, int binxmax, 
		  Mode mode );

  void DrawMeanSlice(const char* key, const int rebinFactor, Mode mode);
  void DrawSigmaSlice(const char* key, const int rebinFactor, Mode mode);
  void DrawGaussSigmaSlice(const char* key, const int rebinFactor, Mode mode);
  void DrawGaussSigmaSlice(const char* key, const int rebinFactor, const int binxmin,
			   const int binxmax, const bool cst_binning, Mode mode);
  void DrawGaussSigmaOverMeanXSlice(const char* key, const int rebinFactor, const int binxmin,
				    const int binxmax, const bool cst_binning, Mode mode);
  void DrawGaussSigmaOverMeanSlice(const char* key, const char* key2, const int rebinFactor, Mode mode);

  void Draw( const char* key, Mode mode);
  
  void Draw( const char* key0, const char* key1, Mode mode);

  // return the two temporary 1d histograms, that have just
  // been plotted
  TH1* h0() {return h0_;}
  TH1* h1() {return h1_;}

  TLegend& Legend() {return legend_;}
  const TLegend& Legend() const {return legend_;}
  
  // set the styles for further plots
  void SetStyles( Style* s0, 
		  Style* s1,
		  const char* leg0,
		  const char* leg1);
   
  TH1* Histo( const char* key, unsigned dirIndex);

  TDirectory* dir0(){ return dir0_;}
  TDirectory* dir1(){ return dir1_;}

private:

  // retrieve an histogram in one of the two directories

  // draw 2 1D histograms.
  // the histograms can be normalized to the same number of entries, 
  // or plotted as a ratio.
  void Draw( TH1* h0, TH1* h1, Mode mode );

  int rebin_;
  float xMin_;
  float xMax_;
  bool resetAxis_;

  TFile*      file0_;
  TDirectory* dir0_;
  TFile*      file1_;
  TDirectory* dir1_;
  
  TH1* h0_;
  TH1* h1_;
  
  Style* s0_;
  Style* s1_;
  
  TLegend legend_;

};

#endif

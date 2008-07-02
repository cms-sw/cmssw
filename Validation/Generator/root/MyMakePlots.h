#ifndef MakePlots_h
#define MakePlots_h
/** \class MakePltos
 *
 * Analyze ROOT files produced by analyzer and create plots
 *
 * \author Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
 *
 * \version $Id: MyMakePlots.h,v 1.2 2008/05/23 17:02:43 ksmith Exp $
 *
 */

#include "TString.h"

class MyMakePlots {

  public:
	MyMakePlots();
	//MakePlots(TString root_filename, TString webpath, TString extension="png", bool compare = false, TString compare_filename="", bool logaxis = false);
	void Draw();
	void SetFilename(TString name) { root_filename = name; }
	void SetWebPath(TString name) { webpath = name; }
	void SetExtension(TString name) { extension = name; }
	void SetCompare( bool option ) { compare = option; }
	void SetCompareFilename(TString name) { compare_filename = name; }
	void SetLogAxis( bool option) { logaxis = option; }
	void SetFilePrefix(TString name) {fileprefix = name;}
	void SetDirectory(TString name) {directory = name;}
	void SetRelease(TString name) {_release = name;}
	void SetReference(TString name) {_reference = name;}
	void SetDataset(TString name) {dataset = name;}
	
	
  private:
	TString root_filename;
	TString webpath;
	TString extension;
	bool compare;
	TString compare_filename;
	bool logaxis;
	TString fileprefix;
	TString directory;
	TString _release;
	TString _reference;
	TString dataset;
};

#endif

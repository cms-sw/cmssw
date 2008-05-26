#ifndef SimDataFormats/GeneratorProducts_LHECommonBlocks_h
#define SimDataFormats/GeneratorProducts_LHECommonBlocks_h

extern "C" {
	extern struct HEPRUP_ {
		int idbmup[2];
		double ebmup[2];
		int pdfgup[2];
		int pdfsup[2];
		int idwtup;
		int nprup;
		double xsecup[100];
		double xerrup[100];
		double xmaxup[100];
		int lprup[100];
	} heprup_;

	extern struct HEPEUP_ {
		int nup;
		int idprup;
		double xwgtup;
		double scalup;
		double aqedup;
		double aqcdup;
		int idup[500];
		int istup[500];
		int mothup[500][2];
		int icolup[500][2];
		double pup[500][5];
		double vtimup[500];
		double spinup[500];
	} hepeup_;
} // extern "C"

namespace lhef {
	class HEPRUP;
	class HEPEUP;

	class CommonBlocks {
	    public:
		static void fillHEPRUP(const HEPRUP *heprup);
		static void fillHEPEUP(const HEPEUP *hepeup);

		static void readHEPRUP(HEPRUP *heprup);
		static void readHEPEUP(HEPEUP *hepeup);

	    private:
		CommonBlocks();
		~CommonBlocks();
	};
}

#endif // SimDataFormats/GeneratorProducts_LHECommonBlocks_h

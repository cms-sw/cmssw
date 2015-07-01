from ROOT import gROOT,gSystem

def load_libs():
    print 'loading FWLite.'
    #load the libaries needed
    gSystem.Load("libFWCoreFWLite")
    gROOT.ProcessLine('AutoLibraryLoader::enable();')
    gSystem.Load("libFWCoreFWLite")
    gSystem.Load("libCintex")
    gROOT.ProcessLine('ROOT::Cintex::Cintex::Enable();')
        
    #now the SVfit stuff
    gSystem.Load("libTauAnalysisSVfitStandalone")
    gSystem.Load("pluginTauAnalysisSVfitStandaloneCapabilities")

load_libs()


import ROOT
import os

debug_btagSF = False

# load the BTagCalibrationStandalone.cc macro from https://twiki.cern.ch/twiki/bin/view/CMS/BTagCalibration
csvpath = os.environ['CMSSW_BASE']+"/src/VHbbAnalysis/Heppy/data/csv"
ROOT.gSystem.Load(csvpath+'/BTagCalibrationStandalone.so')

# CSVv2
calib_csv = ROOT.BTagCalibration("csvv2", csvpath+"/CSVv2.csv")

# cMVAv2
calib_cmva = ROOT.BTagCalibration("cmvav2", csvpath+"/cMVAv2.csv")

# map of calibrators. E.g. btag_calibrators["CSVM_nominal_bc"], btag_calibrators["CSVM_up_l"], ...
btag_calibrators = {}
for wp in [ [0, "L"],[1, "M"], [2,"T"] ]:
    for syst in ["central", "up", "down"]:
        btag_calibrators["CSV"+wp[1]+"_"+syst+"_bc"] = ROOT.BTagCalibrationReader(calib_csv, wp[0], "mujets", syst)
        btag_calibrators["CSV"+wp[1]+"_"+syst+"_l"] = ROOT.BTagCalibrationReader(calib_csv, wp[0], "incl", syst)
        btag_calibrators["CMVAV2"+wp[1]+"_"+syst+"_bc"] = ROOT.BTagCalibrationReader(calib_cmva, wp[0], "ttbar", syst)
        btag_calibrators["CMVAV2"+wp[1]+"_"+syst+"_l"] = ROOT.BTagCalibrationReader(calib_cmva, wp[0], "incl", syst)

# function that reads the SF
def get_csv_SF(pt=30., eta=0.0, fl=5, syst="central", wp="CSVM", btag_calibrators=btag_calibrators):

    # no SF for pT<20 GeV or pt>1000 or abs(eta)>2.4
    if abs(eta)>2.4 or pt>1000. or pt<20.:
        return 1.0

    # pt ranges for bc SF
    pt_range_high_bc = 670.-1e-02
    if "CMVAV2" in wp:
        pt_range_high_bc = 320.-1e-02
    pt_range_low_bc = 30.+1e-02

    # b or c jets
    if fl>=4:
        # use end_of_range values for pt in [20,30] or pt in [670,1000], with double error
        out_of_range = False
        if pt>pt_range_high_bc or pt<pt_range_low_bc:
            out_of_range = True        
        pt = min(pt, pt_range_high_bc)
        pt = max(pt, pt_range_low_bc)
        sf = btag_calibrators[wp+"_"+syst+"_bc"].eval(-fl+5 ,eta, pt)
        if out_of_range and syst in ["up","down"]:
            sf = max(2*sf - btag_calibrators[wp+"_central_bc"].eval(-fl+5 ,eta, pt), 0.)
        return sf

    # light jets
    else:
        return btag_calibrators[wp+"_"+syst+"_l"].eval( 2 ,eta, pt)

def applies( flavour, syst ):
    if flavour==5 and syst not in ["central", "up_jes", "down_jes",  "up_lf", "down_lf",  "up_hfstats1", "down_hfstats1", "up_hfstats2", "down_hfstats2"]:
        return False
    elif flavour==4 and syst not in ["central", "up_cferr1", "down_cferr1", "up_cferr2", "down_cferr2" ]:
        return False
    elif flavour==0 and syst not in ["central", "up_jes", "down_jes", "up_hf", "down_hf",  "up_lfstats1", "down_lfstats1", "up_lfstats2", "down_lfstats2" ]:
        return False
    return True

if debug_btagSF:
    print "POG WP:"
    for wp in [ "CSVL", "CSVM", "CSVT", "CMVAV2L", "CMVAV2M", "CMVAV2T" ]:
        print wp+":"
        for syst in ["central", "up", "down"]:
            print "\t"+syst+":"
            print "\t\tB(pt=30, eta=0.0):", get_csv_SF(30., 0.0, 5, syst, wp)
            print "\t\tC(pt=30, eta=0.0):", get_csv_SF(30., 0.0, 4, syst, wp)
            print "\t\tL(pt=30, eta=0.0):", get_csv_SF(30., 0.0, 0, syst, wp)
            
    print "Iterative:"
    for syst in ["central", "up_jes", "down_jes", "up_lf", "down_lf", "up_hf", "down_hf", "up_hfstats1", "down_hfstats1", "up_hfstats2", "down_hfstats2", "up_lfstats1", "down_lfstats1", "up_lfstats2", "down_lfstats2", "up_cferr1", "down_cferr1", "up_cferr2", "down_cferr2"]:
        print "\t"+syst
        csv_calib_iterative = ROOT.BTagCalibrationReader(calib_csv, 3 , "iterativefit", syst)
        weight = lambda pt, eta, fl, csv, csv_calib=csv_calib_iterative, check=applies, syst=syst : csv_calib.eval( -fl+5, eta, pt, csv ) if abs(eta)<2.4 and pt>20. and pt<10000. and check(fl,syst) else 1.0
        # usage: weight( pt, eta, flavour, csv )
        print "\t\tB:", weight(50., 0.5, 5, 0.89)
        print "\t\tC:", weight(50., 0.5, 4, 0.89)
        print "\t\tL:", weight(50., 0.5, 3, 0.89)



from ROOT import TStyle, kWhite, kTRUE
from ROOT import kGray, kAzure, kMagenta, kOrange, kWhite
from ROOT import kRed, kBlue, kGreen, kPink, kYellow
from ROOT import TLine, TLatex

from collections import namedtuple, OrderedDict
from math import sin, cos, tan, atan, exp, pi

Plot_params = namedtuple('Params',
                         ['plotNumber',
                          'abscissa', 'ordinate',
                          'ymin', 'ymax',
                          'xmin', 'xmax',
                          'quotaName', 'iDrawEta',
                          'histoMin', 'histoMax',
                          'zLog', 'iRebin'])
plots = {}
plots.setdefault('x_vs_eta', Plot_params(10, '#eta', 'x/X_{0}', 0.0, 2.575, -4.0, 4.0, '', 0, 0., 0., 0, 1))
plots.setdefault('x_vs_phi', Plot_params(20, '#varphi [rad]', 'x/X_{0}', 0.0, 6.2, -4.0, 4.0, '', 0, 0., 0., 0, 1))
plots.setdefault('x_vs_R',   Plot_params(40, 'R [cm]', 'x/X_{0}', 0.0, 70.0, 0.0, 1200.0, '', 0, 0., 0., 0, 1))
plots.setdefault('l_vs_eta', Plot_params(1010, '#eta', 'x/#lambda_{0}', 0.0, 0.73, -4.0, 4.0, '', 0, 0., 0., 0, 1))
plots.setdefault('l_vs_phi', Plot_params(1020, '#varphi [rad]', 'x/#lambda_{0}', 0.0, 1.2, -4.0, 4.0, '', 0, 0., 0., 0, 1))
plots.setdefault('l_vs_R',   Plot_params(1040, '#R [cm]', 'x/#lambda_{0}', 0.0, 7.5, 0.0, 1200.0, '', 0, 0., 0., 0, 1))
plots.setdefault('x_vs_eta_vs_phi', Plot_params(30, '#eta', '#varphi', -3.2, 3.2, -5.0, 5.0, 'x/X_{0}', 0, -1., -1., 0, 1))
plots.setdefault('l_vs_eta_vs_phi', Plot_params(1030, '#eta', '#varphi', -3.2, 3.2, -5.0, 5.0, 'x/#lambda_{0}', 0, -1, -1, 0, 1))
plots.setdefault('x_vs_z_vs_Rsum', Plot_params(50, 'z [mm]', 'R [mm]', 0., 0., 0., 0., '#Sigmax/X_{0}', 1, 0., 2.5, 0, 0))
plots.setdefault('x_vs_z_vs_R', Plot_params(60, 'z [mm]', 'R [mm]', 0., 0., 0., 0., '1/X_{0}', 1, 0.00001, 0.01, 1, 0))
plots.setdefault('l_vs_z_vs_Rsum', Plot_params(1050, 'z [mm]', 'R [mm]', 0., 0., 0., 0., '#Sigmax/#lambda_{0}', 1, 0., 1., 0, 0))
plots.setdefault('l_vs_z_vs_R', Plot_params(1060, 'z [mm]', 'R [mm]', 0., 1400., -3500., 3500., '1/#lambda_{0}', 1, 0.001, 0.9, 1, 0))
plots.setdefault('l_vs_z_vs_R_geocomp', Plot_params(1060, 'z [mm]', 'R [mm]', 0., 1400., -3500., 3500., '1/#lambda_{0}', 1, 0.001, 0.9, 0, 0))
# x_over_l_vs_eta is not extracted from the ROOT file 
# but generated from 10 and 1010 in MaterialBudget.py
plots.setdefault('x_over_l_vs_eta', Plot_params(10, '#eta', '(#frac{x}{X_{0}})/(#frac{x}{#lambda_{0}})', 0., 0., 0., 0., '', 0, -1, -1, 0, 0))
# same goes with x_over_l_vs_phi 20/1020
plots.setdefault('x_over_l_vs_phi', Plot_params(20, '#varphi [rad]', '(#frac{x}{X_{0}})/(#frac{x}{#lambda_{0}})', 0., 0., 0., 0., '', 0, -1, -1, 0, 0))

# Conversion name from the label (key) to the components in CMSSW/Geometry
_LABELS2COMPS = {'BeamPipe': 'BEAM',
                 'Tracker': 'Tracker',
                 'Pixel':   ['PixelBarrel', 'PixelForwardZplus', 'PixelForwardZminus'],
                 'PixBar':  'PixelBarrel',
                 'PixFwd':  ['PixelForwardZplus', 'PixelForwardZminus', 'PixelForward'],
                 'PixFwdMinus': 'PixelForwardZminus',
                 'PixFwdPlus':  'PixelForwardZplus',
                 'TIB':         'TIB',
                 'TOB':         'TOB',
                 'TIDB':        'TIDB',
                 'TIDF':        'TIDF',
                 'TEC':         'TEC',
                 'InnerServices': ['TIBTIDServicesF', 'TIBTIDServicesB'],
                 'TkStrct': ['TrackerOuterCylinder', 'TrackerBulkhead'],
                 'Phase2PixelBarrel': 'Phase2PixelBarrel',
                 'Phase2OTBarrel': 'Phase2OTBarrel',
                 'Phase2PixelEndcap': 'Phase2PixelEndcap',
                 'Phase2OTForward': 'Phase2OTForward'}

# Compounds are used to stick together different part of the Tracker
# detector, so that their cumulative material description can be
# derived. The key name can be generic, while the names in the
# associated list must be such that an appropriate material
# description file, in ROOT format, is present while producing the
# cumulative plot. A missing element will invalidate the full
# procedure.
COMPOUNDS = OrderedDict()
COMPOUNDS["Tracker"] = ["Tracker"]
COMPOUNDS["TrackerSum"] = ["TIB", "TIDF", "TIDB",
                           "BeamPipe", "InnerServices",
                           "TOB", "TEC",
                           "TkStruct",
                           "PixBar", "PixFwdPlus", "PixFwdMinus"]
COMPOUNDS["TrackerSumPhaseII"] = ["BeamPipe",
                                  "Phase2PixelBarrel",
                                  "Phase2OTBarrel", "Phase2OTForward",
                                  "Phase2PixelEndcap"]
COMPOUNDS["Pixel"] = ["PixBar", "PixFwdMinus", "PixFwdPlus"]
COMPOUNDS["Strip"] = ["TIB", "TIDF", "TIDB", "InnerServices", "TOB", "TEC"]
COMPOUNDS["InnerTracker"] = ["TIB", "TIDF", "TIDB", "InnerServices"]

# The DETECTORS must be the single component of the tracker for which
# the user can ask for the corresponding material description.
DETECTORS = OrderedDict()
DETECTORS["BeamPipe"] = kGray+2
DETECTORS["InnerServices"] = kGreen+2
DETECTORS["PixBar"] = kAzure-5
DETECTORS["Phase1PixelBarrel"] = kAzure-5
DETECTORS["Phase2PixelBarrel"] = kAzure-5
DETECTORS["PixFwdPlus"] = kAzure-9
DETECTORS["PixFwdMinus"] = kAzure-9
DETECTORS["Phase2PixelEndcap"] = kAzure-9
DETECTORS["Phase2OTBarrel"] = kMagenta-2
DETECTORS["Phase2OTForward"] = kOrange-2
DETECTORS["TIB"] = kMagenta-6
DETECTORS["TIDF"] = kMagenta+2
DETECTORS["TIDB"] = kMagenta+2
DETECTORS["TOB"] = kOrange+10
DETECTORS["TEC"] = kOrange-2

# sDETS are the label of the Tracker elements in the Reconstruction
# geometry. They are all used to derive the reconstruction material
# profile to be compared to the one obtained directly from the
# simulation. A missing key in the real reconstruction geometry is not
# a problem, since this will imply that the corresponding plotting
# routine will skip that missing part. For this reason this map can be
# made as inclusive as possible with respect to the many
# reconstruction geometries in CMSSW.
sDETS = OrderedDict()
sDETS["PXB"] = kRed
sDETS["PXF"] = kBlue
sDETS["TIB"] = kGreen
sDETS["TID"] = kYellow
sDETS["TOB"] = kOrange
sDETS["TEC"] = kPink

# hist_label_to_num contains the logical names of the Tracker detector
# that holds material. They are therefore not aware of which detector
# they belong to, but they are stored in specific plots in all the
# mat*root files produced. The numbering of the plots is identical
# across all files.
hist_label_to_num = OrderedDict()
hist_label_to_num['SUM'] = [0, kGreen+1, 'Total']
hist_label_to_num['SUP'] = [100, 13, 'Support'] # [Index , color, legend label]
hist_label_to_num['SEN'] = [200, 27, 'Sensitive']
hist_label_to_num['CAB'] = [300, 46, 'Cables']
hist_label_to_num['COL'] = [400, 38, 'Cooling']
hist_label_to_num['ELE'] = [500, 30, 'Electronics']
hist_label_to_num['OTH'] = [600, 42, 'Other']
hist_label_to_num['AIR'] = [700, 29, 'Air']

def setTDRStyle():
    """Function to setup a TDR-like style"""

    tdrStyle = TStyle("tdrStyle","Style for P-TDR")

    # For the canvas:
    tdrStyle.SetCanvasBorderMode(0)
    tdrStyle.SetCanvasColor(kWhite)
    tdrStyle.SetCanvasDefH(600) #Height of canvas
    tdrStyle.SetCanvasDefW(600) #Width of canvas
    tdrStyle.SetCanvasDefX(0)   #Position on screen
    tdrStyle.SetCanvasDefY(0)

    # For the Pad:
    tdrStyle.SetPadBorderMode(0)
    tdrStyle.SetPadColor(kWhite)
    tdrStyle.SetPadGridX(False)
    tdrStyle.SetPadGridY(False)
    tdrStyle.SetGridColor(kWhite)
    tdrStyle.SetGridStyle(3)
    tdrStyle.SetGridWidth(1)
    tdrStyle.SetPadTickX(True)
    tdrStyle.SetPadTickY(True)

    # For the frame:
    tdrStyle.SetFrameBorderMode(0)
    tdrStyle.SetFrameBorderSize(1)
    tdrStyle.SetFrameFillColor(0)
    tdrStyle.SetFrameFillStyle(0)
    tdrStyle.SetFrameLineColor(1)
    tdrStyle.SetFrameLineStyle(1)
    tdrStyle.SetFrameLineWidth(0)

    # For the histo:
    tdrStyle.SetHistLineColor(1)
    tdrStyle.SetHistLineStyle(0)
    tdrStyle.SetHistLineWidth(1)
    tdrStyle.SetEndErrorSize(1)
    #tdrStyle.SetErrorX(0.)
    tdrStyle.SetMarkerStyle(20)

    #For the fit/function:
    tdrStyle.SetOptFit(0)
    tdrStyle.SetFitFormat("5.4g")
    tdrStyle.SetFuncColor(2)
    tdrStyle.SetFuncStyle(1)
    tdrStyle.SetFuncWidth(1)

    #For the date:
    tdrStyle.SetOptDate(0)

    # For the statistics box:
    tdrStyle.SetOptFile(0)
    tdrStyle.SetOptStat(0); # To display the mean and RMS:   SetOptStat("mr")
    tdrStyle.SetStatColor(kWhite)
    tdrStyle.SetStatFont(42)
    tdrStyle.SetStatFontSize(0.025)
    tdrStyle.SetStatTextColor(1)
    tdrStyle.SetStatFormat("6.4g")
    tdrStyle.SetStatBorderSize(1)
    tdrStyle.SetStatH(0.1)
    tdrStyle.SetStatW(0.15)

    # Margins:
    tdrStyle.SetPadTopMargin(0.1)
    tdrStyle.SetPadBottomMargin(0.1)
    tdrStyle.SetPadLeftMargin(0.1)
    tdrStyle.SetPadRightMargin(0.1)

    # For the Global title:
    tdrStyle.SetOptTitle(1)
    tdrStyle.SetTitleFont(42)
    tdrStyle.SetTitleColor(1)
    tdrStyle.SetTitleTextColor(1)
    tdrStyle.SetTitleFillColor(10)
    tdrStyle.SetTitleFontSize(0.0525)
    tdrStyle.SetTitleH(0); # Set the height of the title box
    tdrStyle.SetTitleW(0); # Set the width of the title box
    tdrStyle.SetTitleX(0.5); # Set the position of the title box
    tdrStyle.SetTitleY(1.0); # Set the position of the title box
    tdrStyle.SetTitleStyle(1001);
    tdrStyle.SetTitleBorderSize(0);
    tdrStyle.SetTitleAlign(23)

    # For the axis titles:
    tdrStyle.SetTitleColor(1, "XYZ")
    tdrStyle.SetTitleFont(42, "XYZ")
    tdrStyle.SetTitleSize(0.05, "XY")
    tdrStyle.SetTitleSize(0.035, "Z")
    tdrStyle.SetTitleXOffset(1.0)
    tdrStyle.SetTitleYOffset(1.0)

    # For the axis labels:
    tdrStyle.SetLabelColor(1, "XYZ")
    tdrStyle.SetLabelFont(42, "XYZ")
    tdrStyle.SetLabelOffset(5e-3, "XYZ")
    tdrStyle.SetLabelSize(0.03, "XYZ")

    # For the axis:
    tdrStyle.SetAxisColor(1, "XYZ")
    tdrStyle.SetStripDecimals(kTRUE)
    tdrStyle.SetTickLength(0.03, "XYZ")
    tdrStyle.SetNdivisions(510, "XYZ")
    tdrStyle.SetPadTickX(1)  # To get tick marks on the opposite side of the frame
    tdrStyle.SetPadTickY(1)

    # Change for log plots:
    tdrStyle.SetOptLogx(0)
    tdrStyle.SetOptLogy(0)
    tdrStyle.SetOptLogz(0)

    # Miscellaneous
    tdrStyle.SetLegendBorderSize(0)

    # Postscript options:
    tdrStyle.SetPaperSize(20.,20.)

    tdrStyle.cd()

def drawEtaValues():
    """Function to draw the eta.

    Function to draw the eta references on top of an already existing
    TCanvas. The lines and labels drawn are collected inside a list and
    the list is returned to the user to extend the live of the objects
    contained, otherwise no lines and labels will be drawn, since they
    will be garbage-collected as soon as this function returns.
    """

    # Add eta labels
    keep_alive = []
    etas = [ 0.2*i for i in range(-17,18) ]

    etax = 2850.
    etay = 1240.
    lineL = 110.
    offT = 10.

    for ieta in etas:
        th = 2*atan(exp(-ieta))
        talign = 21

        #IP
        lineh = TLine(-20.,0.,20.,0.)
        lineh.Draw()
        linev = TLine(0.,-10.,0.,10.)
        linev.Draw()
        keep_alive.append(lineh)
        keep_alive.append(linev)

        x1 = 0
        y1 = 0
        if ieta>-1.6 and ieta<1.6:
            x1 = etay/tan(th)
            y1 = etay
        elif ieta <=-1.6:
            x1 = -etax
            y1 = -etax*tan(th)
            talign = 11
        elif ieta>=1.6:
            x1 = etax
            y1 = etax*tan(th)
            talign = 31
        x2 = x1+lineL*cos(th)
        y2 = y1+lineL*sin(th)
        xt = x2
        yt = y2+offT

        line1 = TLine(x1,y1,x2,y2)
        line1.Draw()
        keep_alive.append(line1)

        text = "%3.1f" % ieta
        t1 = TLatex(xt, yt, '%s' % ('#eta = 0' if ieta == 0 else text))
        t1.SetTextSize(0.03)
        t1.SetTextAlign(talign)
        t1.Draw()
        keep_alive.append(t1)
    return keep_alive

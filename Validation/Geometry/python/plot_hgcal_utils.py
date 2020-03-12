from __future__ import print_function
from ROOT import TStyle, kWhite, kTRUE
from ROOT import gROOT, gStyle
from ROOT import kGray, kAzure, kMagenta, kOrange, kWhite
from ROOT import kRed, kBlue, kGreen, kPink, kYellow
from ROOT import TLine, TLatex, TColor

from collections import namedtuple, OrderedDict
from math import sin, cos, tan, atan, exp, pi
from array import array

from Validation.Geometry.plot_utils import Plot_params

plots = {}
plots.setdefault('x_vs_eta', Plot_params(10, '#eta', 'x/X_{0}', 0.0, 145., -4.0, 4.0, '', 0, 0., 0., 0, 1))
plots.setdefault('x_vs_phi', Plot_params(20, '#varphi [rad]', 'x/X_{0}', 0.0, 6.2, -4.0, 4.0, '', 0, 0., 0., 0, 1))
plots.setdefault('x_vs_R',   Plot_params(40, 'R [cm]', 'x/X_{0}', 0.0, 70.0, 0.0, 1200.0, '', 0, 0., 0., 0, 1))
plots.setdefault('l_vs_eta', Plot_params(10010, '#eta', 'x/#lambda_{I}', 0.0, 22.8, -4.0, 4.0, '', 0, 0., 0., 0, 1))
plots.setdefault('l_vs_phi', Plot_params(10020, '#varphi [rad]', 'x/#lambda_{I}', 0.0, 1.2, -4.0, 4.0, '', 0, 0., 0., 0, 1))
plots.setdefault('l_vs_R',   Plot_params(10040, 'R [cm]', 'x/#lambda_{I}', 0.0, 7.5, 0.0, 1200.0, '', 0, 0., 0., 0, 1))
plots.setdefault('x_vs_eta_vs_phi', Plot_params(30, '#eta', '#varphi', 0., 0., 0., 0., 'x/X_{0}', 0, -1., -1., 0, 1))
plots.setdefault('l_vs_eta_vs_phi', Plot_params(10030, '#eta', '#varphi', 0., 0., 0., 0., 'x/#lambda_{I}', 0, -1, -1, 0, 1))
plots.setdefault('x_vs_z_vs_Rsum', Plot_params(50, 'z [mm]', 'R [mm]', 0., 0., 0., 0., '#Sigmax/X_{0}', 1, -1., -1., 0, 0))
plots.setdefault('x_vs_z_vs_Rsumcos', Plot_params(52, 'z [mm]', 'R [mm]', 0., 0., 0., 0., '#Sigmax/X_{0}', 1, -1., -1., 0, 0))
#plots.setdefault('x_vs_z_vs_R', Plot_params(60, 'z [mm]', 'R [mm]', 0., 0., 0., 0., '1/X_{0}', 1, -1., -1., 0, 0))
plots.setdefault('x_vs_z_vs_Rloc', Plot_params(70, 'z [mm]', 'R [mm]', 0., 0., 0., 0., 'x/X_{0}', 1, -1., -1., 0, 0))
plots.setdefault('x_vs_z_vs_Rloccos', Plot_params(72, 'z [mm]', 'R [mm]', 0., 0., 0., 0., 'x/X_{0}', 1, -1., -1., 0, 0))
plots.setdefault('l_vs_z_vs_Rsum', Plot_params(10050, 'z [mm]', 'R [mm]', 0., 0., 0., 0., '#Sigmax/#lambda_{I}', 1, -1., -1., 0, 0))
plots.setdefault('l_vs_z_vs_Rsumcos', Plot_params(10052, 'z [mm]', 'R [mm]', 0., 0., 0., 0., '#Sigmax/#lambda_{I}', 1, -1., -1., 0, 0))
plots.setdefault('l_vs_z_vs_R', Plot_params(10060, 'z [mm]', 'R [mm]', 0., 0., 0., 0., '1/#lambda_{I}', 1, -1., -1., 0, 0))
plots.setdefault('l_vs_z_vs_Rloc', Plot_params(10070, 'z [mm]', 'R [mm]', 0., 0., 0., 0., 'x/#lambda_{I}', 1, -1., -1., 0, 0))
plots.setdefault('l_vs_z_vs_Rloccos', Plot_params(10072, 'z [mm]', 'R [mm]', 0., 0., 0., 0., 'x/#lambda_{I}', 1, -1., -1., 0, 0))
plots.setdefault('x_over_l_vs_eta', Plot_params(10, '#eta', '(x/X_{0})/(x/#lambda_{I})', 0., 0., 0., 0., '', 0, -1, -1, 0, 0))
plots.setdefault('x_over_l_vs_phi', Plot_params(20, '#varphi [rad]', '(x/X_{0})/(x/#lambda_{I})', 0., 0., 0., 0., '', 0, -1, -1, 0, 0))

# Conversion name from the label (key) to the components in CMSSW/Geometry
_LABELS2COMPS = {'BeamPipe': 'BEAM',
                 'Tracker': 'Tracker', 
                 'EndcapTimingLayer + Thermal Screen': 'CALOECTSFront',
                 'Neutron Moderator + Thermal Screen' : 'CALOECTSMiddle',
                 'HGCal + HGCal Service + Thermal Screen' : 'CALOECTSRear',
                 'Solenoid Magnet' : 'MGNT',
                 'Muon Wheels and Cables' : 'MB',
                 'ECAL': 'ECAL',
                 'HCal': 'HCal',
                 'FromVertexToBackOfHGCal' : ['BEAM','Tracker','ECAL','HCal','CALOECTSFront','CALOECTSMiddle','CALOECTSRear','MGNT','MB'],
                 'HGCal': 'HGCal',
                 'HGCalEE': 'HGCalEE',
                 'HGCalHE': ['HGCalHEsil', 'HGCalHEmix']
                 }

# Compounds are used to stick together different part of the HGCal
# detector, so that their cumulative material description can be
# derived. The key name can be generic, while the names in the
# associated list must be such that an appropriate material
# description file, in ROOT format, is present while producing the
# cumulative plot. A missing element will invalidate the full
# procedure.
COMPOUNDS = OrderedDict()
COMPOUNDS["HGCal"] = ["HGCal"]
COMPOUNDS["HGCalEE"] = ["HGCalEE"]
COMPOUNDS["HGCalHE"] = ["HGCalHEsil", "HGCalHEmix"]
COMPOUNDS["FromVertexToBackOfHGCal"] = ["BeamPipe","Tracker","ECAL","HCal","EndcapTimingLayer + Thermal Screen","Neutron Moderator + Thermal Screen","HGCal + HGCal Service + Thermal Screen","Solenoid Magnet","Muon Wheels and Cables"]

# The DETECTORS must be the single component of the HGCal for which
# the user can ask for the corresponding material description.
# BE CAREFUL: When running on a single detector and not through all 
# only uncomment that one below. e.g. when running on HGCal only you should 
# have only: 
# DETECTORS["HGCal"] = kAzure-5
# and others should be commented out.
DETECTORS = OrderedDict()
DETECTORS["BeamPipe"] = kGray+2
DETECTORS["Tracker"]  = 9 #kAzure-5
DETECTORS["ECAL"] = 2 #kOrange+10
DETECTORS["HCal"] = 6  #kMagenta-2
DETECTORS["EndcapTimingLayer + Thermal Screen"] = 7#kAzure-9 
DETECTORS["Neutron Moderator + Thermal Screen"] = 46#kOrange+5
DETECTORS["HGCal + HGCal Service + Thermal Screen"] = 5#kOrange-2
DETECTORS["Solenoid Magnet"] = 4#kGray+5
DETECTORS["Muon Wheels and Cables"] = 28
#When running to get the R vs z sum of all subdetectors comment out above and 
#uncomment the next line
#DETECTORS["FromVertexToBackOfHGCal"] = 30


# sDETS are the label of the HGCal elements in the Reconstruction
# geometry. They are all used to derive the reconstruction material
# profile to be compared to the one obtained directly from the
# simulation. A missing key in the real reconstruction geometry is not
# a problem, since this will imply that the corresponding plotting
# routine will skip that missing part. For this reason this map can be
# made as inclusive as possible with respect to the many
# reconstruction geometries in CMSSW.
sDETS = OrderedDict()
#sDETS["HGCalEE"] = kRed
#sDETS["HGCalHEsil"] = kBlue
#sDETS["HGCalHEmix"] = kGreen
sDETS["ECAL"] = kBlue
sDETS["HCal"] = kOrange
#sDETS[""] = kYellow
#sDETS[""] = kOrange
#sDETS[""] = kPink

# hist_label_to_num contains the logical names of the HGCal detector
# that holds material. They are therefore not aware of which detector
# they belong to, but they are stored in specific plots in all the
# mat*root files produced. The numbering of the plots is identical
# across all files.
hist_label_to_num = OrderedDict()
hist_label_to_num['COP'] = [100, 2, 'Copper'] # Index first, color second, legend label third
hist_label_to_num['SCI'] = [200, 3, 'Scintillator']
hist_label_to_num['CAB'] = [300, 4, 'Cables']
hist_label_to_num['MNE'] = [400, 5, 'HGC_G10-FR4']
hist_label_to_num['SIL'] = [500, 6, 'Silicon']
hist_label_to_num['OTH'] = [600, 7, 'Other']
hist_label_to_num['AIR'] = [700, 8, 'Air']
hist_label_to_num['SST'] = [800, 9, 'Stainless Steel']
hist_label_to_num['WCU'] = [900, 28, 'WCu']
hist_label_to_num['LEA'] = [1000, 12, 'Lead']
hist_label_to_num['EPX'] = [1100, 46, 'Epoxy']
hist_label_to_num['KAP'] = [1200, 49, 'Kapton']
hist_label_to_num['ALU'] = [1300, 33, 'Aluminium']

def TwikiPrintout(plotname, label, zoom): 
    """The plots in the twiki are already too much and to avoid mistakes 
       we will try to automatize the procedure
    """

    #Twiki will strip out spaces
    label = label.replace(" ", "_")

    zoomstring = ""

    if zoom == "all":
        zoomstring = ""
        zoomtitle = "in all HGCal"
        zoomdir = "%s/" % label
    elif zoom == "zplus":
        zoomstring = "_ZplusZoom"
        zoomtitle = "in Z+ endcap of HGCal"
        zoomdir = "%s/ZPlusZoom/" % label
    elif zoom == "zminus":
        zoomstring = "_ZminusZoom"
        zoomtitle = "in Z- endcap of HGCal"
        zoomdir = "%s/ZMinusZoom/" % label
    else :
        print("WRONG OPTION")


    #Here for the hide button
    if plotname == "x_vs_z_vs_Rsum":
        print("%%TWISTY{ mode=\"div\" showlink=\"Click to see the %s plots %s \" hidelink=\"Hide %s %s\" showimgright=\"%%ICONURLPATH{toggleopen-small}%%\" hideimgright=\"%%ICONURLPATH{toggleclose-small}%%\"}%%" % (label,zoomtitle, label, zoomtitle))

    if "Rsum" in plotname and "x_vs" in plotname and not "cos" in plotname: 
        print("| <img alt=\"HGCal_%s%s%s.png\" height=\"300\" width=\"550\" src=\"http://apsallid.web.cern.ch/apsallid/HGCalMaterial/%sHGCal_%s%s%s.png\" /> | The plot on the left shows the 2D profile histogram for *%s* %s that displays the mean value of the material budget in units of radiation length in each R-z cell. R-z cell is 1 cm x 1 mm. The plot depicts the accumulated material budget as seen by the track, as the track travels throughout the detector.[[http://apsallid.web.cern.ch/apsallid/HGCalMaterial/%sHGCal_%s%s%s.pdf][Click to enlarge plot]] |" % (plotname,label,zoomstring,zoomdir,plotname,label,zoomstring, label, zoomtitle,zoomdir,plotname,label,zoomstring))

    if "Rsum" in plotname and "l_vs" in plotname and not "cos" in plotname: 
        print("| <img alt=\"HGCal_%s%s%s.png\" height=\"300\" width=\"550\" src=\"http://apsallid.web.cern.ch/apsallid/HGCalMaterial/%sHGCal_%s%s%s.png\" /> | The plot on the left shows the 2D profile histogram for *%s* %s that displays the mean value of the material budget in units of interaction length in each R-z cell. R-z cell is 1 cm x 1 mm. The plot depicts the accumulated material budget as seen by the track, as the track travels throughout the detector.[[http://apsallid.web.cern.ch/apsallid/HGCalMaterial/%sHGCal_%s%s%s.pdf][Click to enlarge plot]] |" % (plotname,label,zoomstring,zoomdir,plotname,label,zoomstring, label, zoomtitle,zoomdir,plotname,label,zoomstring))

    if "Rsumcos" in plotname and "x_vs" in plotname: 
        print("| <img alt=\"HGCal_%s%s%s.png\" height=\"300\" width=\"550\" src=\"http://apsallid.web.cern.ch/apsallid/HGCalMaterial/%sHGCal_%s%s%s.png\" /> | The plot on the left shows the 2D profile histogram for *%s* %s that displays the mean value of the material budget in units of radiation length in each R-z cell. R-z cell is 1 cm x 1 mm. The plot depicts the orthogonal accumulated material budget, that is cos(theta) what the track sees.[[http://apsallid.web.cern.ch/apsallid/HGCalMaterial/%sHGCal_%s%s%s.pdf][Click to enlarge plot]] |" % (plotname,label,zoomstring,zoomdir,plotname,label,zoomstring, label, zoomtitle,zoomdir,plotname,label,zoomstring))

    if "Rsumcos" in plotname and "l_vs" in plotname: 
        print("| <img alt=\"HGCal_%s%s%s.png\" height=\"300\" width=\"550\" src=\"http://apsallid.web.cern.ch/apsallid/HGCalMaterial/%sHGCal_%s%s%s.png\" /> | The plot on the left shows the 2D profile histogram for *%s* %s that displays the mean value of the material budget in units of interaction length in each R-z cell. R-z cell is 1 cm x 1 mm. The plot depicts the orthogonal accumulated material budget, that is cos(theta) what the track sees.[[http://apsallid.web.cern.ch/apsallid/HGCalMaterial/%sHGCal_%s%s%s.pdf][Click to enlarge plot]] |" % (plotname,label,zoomstring,zoomdir,plotname,label,zoomstring, label, zoomtitle,zoomdir,plotname,label,zoomstring))

    if "Rloc" in plotname and "x_vs" in plotname and not "cos" in plotname: 
        print("| <img alt=\"HGCal_%s%s%s.png\" height=\"300\" width=\"550\" src=\"http://apsallid.web.cern.ch/apsallid/HGCalMaterial/%sHGCal_%s%s%s.png\" /> | The plot on the left shows the 2D profile histogram for *%s* %s that displays the local mean value of the material budget in units of radiation length in each R-z cell. R-z cell is 1 cm x 1 mm. The plot depicts the local material budget as seen by the track, as the track travels throughout the detector.[[http://apsallid.web.cern.ch/apsallid/HGCalMaterial/%sHGCal_%s%s%s.pdf][Click to enlarge plot]] |" % (plotname,label,zoomstring,zoomdir,plotname,label,zoomstring, label, zoomtitle,zoomdir,plotname,label,zoomstring))

    if "Rloc" in plotname and "l_vs" in plotname and not "cos" in plotname: 
        print("| <img alt=\"HGCal_%s%s%s.png\" height=\"300\" width=\"550\" src=\"http://apsallid.web.cern.ch/apsallid/HGCalMaterial/%sHGCal_%s%s%s.png\" /> | The plot on the left shows the 2D profile histogram for *%s* %s that displays the local mean value of the material budget in units of interaction length in each R-z cell. R-z cell is 1 cm x 1 mm. The plot depicts the local material budget as seen by the track, as the track travels throughout the detector.[[http://apsallid.web.cern.ch/apsallid/HGCalMaterial/%sHGCal_%s%s%s.pdf][Click to enlarge plot]] |" % (plotname,label,zoomstring,zoomdir,plotname,label,zoomstring, label, zoomtitle,zoomdir,plotname,label,zoomstring))

    #Here again for the closing of the hide button
    if plotname == "l_vs_z_vs_Rloc":
        print("%ENDTWISTY%")

    """ 
    I won't put the local cos plots for now, only the sum cos above
    if "Rloccos" in plotname and "x_vs" in plotname: 
         print "| <img alt=\"HGCal_%s%s%s.png\" height=\"300\" width=\"550\" src=\"http://apsallid.web.cern.ch/apsallid/HGCalMaterial/%sHGCal_%s%s%s.png\" /> | The plot on the left shows the 2D profile histogram for *%s* %s that displays the local mean value of the material budget in units of radiation length in each R-z cell. R-z cell is 1 cm x 1 mm. The plot depicts the orthogonal accumulated material budget, that is cos(theta) what the track sees.[[http://apsallid.web.cern.ch/apsallid/HGCalMaterial/%sHGCal_%s%s%s.pdf][Click to enlarge plot]] |" % (plotname,label,zoomstring,zoomdir,plotname,label,zoomstring, label, zoomtitle,zoomdir,plotname,label,zoomstring)

    if "Rloccos" in plotname and "l_vs" in plotname: 
         print "| <img alt=\"HGCal_%s%s%s.png\" height=\"300\" width=\"550\" src=\"http://apsallid.web.cern.ch/apsallid/HGCalMaterial/%sHGCal_%s%s%s.png\" /> | The plot on the left shows the 2D profile histogram for *%s* %s that displays the local mean value of the material budget in units of interaction length in each R-z cell. R-z cell is 1 cm x 1 mm. The plot depicts the orthogonal accumulated material budget, that is cos(theta) what the track sees.[[http://apsallid.web.cern.ch/apsallid/HGCalMaterial/%sHGCal_%s%s%s.pdf][Click to enlarge plot]] |" % (plotname,label,zoomstring,zoomdir,plotname,label,zoomstring, label, zoomtitle,zoomdir,plotname,label,zoomstring)
    """

def acustompalette(): 
    NRGBs = 7
    NCont = 100
    ncolors = array('i', [])
    gStyle.SetNumberContours(NCont);
    stops   = [ 0.00, 0.10, 0.25, 0.45, 0.60, 0.75, 1.00 ]
    red     = [ 1.00, 0.00, 0.00, 0.00, 0.97, 0.97, 0.10 ]
    green   = [ 1.00, 0.97, 0.30, 0.40, 0.97, 0.00, 0.00 ]
    blue    = [ 1.00, 0.97, 0.97, 0.00, 0.00, 0.00, 0.00 ]
    stopsArray = array('d', stops)
    redArray = array('d', red)
    greenArray = array('d', green)
    blueArray = array('d', blue)
    first_color_number = TColor.CreateGradientColorTable(NRGBs, stopsArray, redArray, greenArray, blueArray, NCont);
    gStyle.SetNumberContours(NCont)


    palsize = NCont
    palette = []
    for i in range(palsize):
        palette.append(first_color_number+i)
        palarray = array('i',palette)

    gStyle.SetPalette(palsize,palarray)


#In MeV/mm
dEdx = OrderedDict()
#--------
#Some elements necessary to build our materials
dEdx['Fe'] = 1.143
dEdx['Mn'] = 1.062
dEdx['Cr'] = 1.046
dEdx['Ni'] = 1.307
dEdx['C']  = 0.3952
dEdx['0']  = 0. # 2.398E-04 -> essentially zero
dEdx['H']  = 0. #3.437E-05 -> essentially zero
dEdx['Br'] = 0. #9.814E-04 -> essentially zero
dEdx['W'] = 2.210
dEdx['Al'] = 0.4358
#-------- 

dEdx['Copper'] = 1.257
#http://cmslxr.fnal.gov/source/Geometry/CMSCommonData/data/materials.xml#1996
dEdx['H_Scintillator'] = 0.91512109*dEdx['C'] + 0.084878906*dEdx['H']
dEdx['Silicon'] = 0.3876
#http://cmslxr.fnal.gov/source/Geometry/CMSCommonData/data/materials.xml#2730
dEdx['HGC_G10-FR4'] = 0.18077359*dEdx['Silicon'] + 0.4056325*dEdx['0'] + 0.27804208*dEdx['C'] + 0.068442752*dEdx['H'] + 0.067109079*dEdx['Br']
dEdx['Other'] = 0.
#http://cmslxr.fnal.gov/source/Geometry/CMSCommonData/data/materials.xml#0290
dEdx['Air'] = 0.
#http://cmslxr.fnal.gov/source/Geometry/CMSCommonData/data/materials.xml#3692
dEdx['StainlessSteel'] = 0.6996*dEdx['Fe']+0.01*dEdx['Mn']+0.19*dEdx['Cr']+0.1*dEdx['Ni']+0.0004*dEdx['C'];
#http://cmslxr.fnal.gov/source/Geometry/CMSCommonData/data/materials.xml#0568
dEdx['WCu'] = 0.75*dEdx['W']+0.25*dEdx['Copper']
#--------
dEdx['Lead'] = 1.274 #Pb
dEdx['Epoxy'] = 0.53539691*dEdx['C'] + 0.13179314*dEdx['H'] + 0.33280996*dEdx['0']
dEdx['Kapton'] = 0.59985105*dEdx['C'] + 0.080541353*dEdx['H'] + 0.31960759*dEdx['0']
#Composition of cable as Sunanda uses them is here: 
#http://cmslxr.fnal.gov/source/Geometry/CMSCommonData/data/materials.xml#2841
dEdx['Cables'] = 0.586*dEdx['Copper'] + 0.259*dEdx['C'] + 0.138*dEdx['0'] + 0.017*dEdx['H']

#In mm
MatXo = OrderedDict()
MatXo['Copper'] = 14.3559
MatXo['H_Scintillator'] = 425.393
MatXo['Cables'] = 66.722
MatXo['HGC_G10-FR4'] = 175.056
MatXo['Silicon'] = 93.6762
MatXo['Other'] = 0.
MatXo['Air'] = 301522.
MatXo['StainlessSteel'] = 17.3555
MatXo['WCu'] = 5.1225
MatXo['Lead'] = 5.6118
MatXo['Epoxy'] = 315.901
MatXo['Kapton'] = 365.309

def drawHalfEtaValues():
    """Function to draw the eta.
    Function to draw the eta references on top of an already existing
    TCanvas. The lines and labels drawn are collected inside a list and
    the list is returned to the user to extend the live of the objects
    contained, otherwise no lines and labels will be drawn, since they
    will be garbage-collected as soon as this function returns.
    """

    # Add eta labels
    keep_alive = []
    etas = [ 0.2*i for i in range(0,18) ]

    etax = 2850.#6850.
    etay = 1240.#5200.
    lineL = 110#8600.
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

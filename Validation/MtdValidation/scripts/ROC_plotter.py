import os
import numpy as np
import uproot as up
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

"""
This is a sample code for ROC curve plotting (Feel free to change it to Your needs!)
Code is ran from "Validation/MtdValidation/test" directory
Before running the code initialize cmsenv
Run the code by using python3 -> 'python3 ROC_ploter.py'
Change the DQM File and ROC plot output directories accordingly to Your workspace

"""


print("All libraries has been read in!")

directory_path = os.getenv('PWD') # DQM file location directory, to be adjusted according need
ROC_plots_directory = directory_path + '/ROC' # ROC plots location directory

class MTD_Ele_Iso: # Using class, so code would be a bit shorter

    def __init__(self,filename_Sig: str, filename_Bkg: str,dz_cut: str,dtSignif_cut: bool):

        self.filename_sig = directory_path + filename_Sig
        self.filename_bkg = directory_path + filename_Bkg

        self.cut_type = dtSignif_cut # True if dt_significance is used, false if absolute dt cut is used.
        self.dz_cut_description = dz_cut

        self.Tree_Sig = up.open(self.filename_sig)["DQMData/Run 1/MTD/Run summary/ElectronIso;1"]
        self.Tree_Bkg = up.open(self.filename_bkg)["DQMData/Run 1/MTD/Run summary/ElectronIso;1"]

        self.Sig_hists = {} # Dictionary that will hold all Signal histograms
        self.Bkg_hists = {} # Dictionary that will hold all Bakcground histograms

        self.Sig_iso_eff = {} # Dictionary that will hold isolation efficiency values for all iso cut values in different timing cuts (Signal)
        self.Bkg_iso_eff = {} # Dictionary that will hold isolation efficiency values for all iso cut values in different timing cuts (Bakcground)

    def Read_hists(self):

        self.Sig_hists['Sig_noMTD_EB']  = self.Tree_Sig['Ele_chIso_sum_Sig_EB;1'].to_numpy()
        self.Sig_hists['Sig_noMTD_EE']  = self.Tree_Sig['Ele_chIso_sum_Sig_EE;1'].to_numpy()
        self.Bkg_hists['Bkg_noMTD_EB']  = self.Tree_Bkg['Ele_chIso_sum_Bkg_EB;1'].to_numpy()
        self.Bkg_hists['Bkg_noMTD_EE']  = self.Tree_Bkg['Ele_chIso_sum_Bkg_EE;1'].to_numpy()

        if self.cut_type == True: # dt significance cut case

            self.Sig_hists['Sig_4sigma_EB'] = self.Tree_Sig['Ele_chIso_sum_MTD_4sigma_Sig_EB;1'].to_numpy()
            self.Sig_hists['Sig_3sigma_EB'] = self.Tree_Sig['Ele_chIso_sum_MTD_3sigma_Sig_EB;1'].to_numpy()
            self.Sig_hists['Sig_2sigma_EB'] = self.Tree_Sig['Ele_chIso_sum_MTD_2sigma_Sig_EB;1'].to_numpy()

            self.Sig_hists['Sig_4sigma_EE'] = self.Tree_Sig['Ele_chIso_sum_MTD_4sigma_Sig_EE;1'].to_numpy()
            self.Sig_hists['Sig_3sigma_EE'] = self.Tree_Sig['Ele_chIso_sum_MTD_3sigma_Sig_EE;1'].to_numpy()
            self.Sig_hists['Sig_2sigma_EE'] = self.Tree_Sig['Ele_chIso_sum_MTD_2sigma_Sig_EE;1'].to_numpy()

            self.Bkg_hists['Bkg_4sigma_EB'] = self.Tree_Bkg['Ele_chIso_sum_MTD_4sigma_Bkg_EB;1'].to_numpy()
            self.Bkg_hists['Bkg_3sigma_EB'] = self.Tree_Bkg['Ele_chIso_sum_MTD_3sigma_Bkg_EB;1'].to_numpy()
            self.Bkg_hists['Bkg_2sigma_EB'] = self.Tree_Bkg['Ele_chIso_sum_MTD_2sigma_Bkg_EB;1'].to_numpy()

            self.Bkg_hists['Bkg_4sigma_EE'] = self.Tree_Bkg['Ele_chIso_sum_MTD_4sigma_Bkg_EE;1'].to_numpy()
            self.Bkg_hists['Bkg_3sigma_EE'] = self.Tree_Bkg['Ele_chIso_sum_MTD_3sigma_Bkg_EE;1'].to_numpy()
            self.Bkg_hists['Bkg_2sigma_EE'] = self.Tree_Bkg['Ele_chIso_sum_MTD_2sigma_Bkg_EE;1'].to_numpy()

        else: # absolute dt cut case #optional

            # Can redifine how many of the 7 cuts defined in the EleIsoValidation code to plot. (Here they are numbered in the star position -> 'Ele_chIso_sum_MTD_*_Sig_EB;1')
            # The cut values themselves are defined in the Validaton code, here we just plot the ROC curves.

            self.Sig_hists['Sig_cut1_EB'] = self.Tree_Sig['Ele_chIso_sum_MTD_1_Sig_EB;1'].to_numpy()
            self.Sig_hists['Sig_cut2_EB'] = self.Tree_Sig['Ele_chIso_sum_MTD_2_Sig_EB;1'].to_numpy()
            self.Sig_hists['Sig_cut3_EB'] = self.Tree_Sig['Ele_chIso_sum_MTD_3_Sig_EB;1'].to_numpy()
            self.Sig_hists['Sig_cut5_EB'] = self.Tree_Sig['Ele_chIso_sum_MTD_5_Sig_EB;1'].to_numpy()

            self.Sig_hists['Sig_cut1_EE'] = self.Tree_Sig['Ele_chIso_sum_MTD_1_Sig_EE;1'].to_numpy()
            self.Sig_hists['Sig_cut2_EE'] = self.Tree_Sig['Ele_chIso_sum_MTD_2_Sig_EE;1'].to_numpy()
            self.Sig_hists['Sig_cut3_EE'] = self.Tree_Sig['Ele_chIso_sum_MTD_3_Sig_EE;1'].to_numpy()
            self.Sig_hists['Sig_cut5_EE'] = self.Tree_Sig['Ele_chIso_sum_MTD_5_Sig_EE;1'].to_numpy()

            self.Bkg_hists['Bkg_cut1_EB'] = self.Tree_Bkg['Ele_chIso_sum_MTD_1_Bkg_EB;1'].to_numpy()
            self.Bkg_hists['Bkg_cut2_EB'] = self.Tree_Bkg['Ele_chIso_sum_MTD_2_Bkg_EB;1'].to_numpy()
            self.Bkg_hists['Bkg_cut3_EB'] = self.Tree_Bkg['Ele_chIso_sum_MTD_3_Bkg_EB;1'].to_numpy()
            self.Bkg_hists['Bkg_cut5_EB'] = self.Tree_Bkg['Ele_chIso_sum_MTD_5_Bkg_EB;1'].to_numpy()

            self.Bkg_hists['Bkg_cut1_EE'] = self.Tree_Bkg['Ele_chIso_sum_MTD_1_Bkg_EE;1'].to_numpy()
            self.Bkg_hists['Bkg_cut2_EE'] = self.Tree_Bkg['Ele_chIso_sum_MTD_2_Bkg_EE;1'].to_numpy()
            self.Bkg_hists['Bkg_cut3_EE'] = self.Tree_Bkg['Ele_chIso_sum_MTD_3_Bkg_EE;1'].to_numpy()
            self.Bkg_hists['Bkg_cut5_EE'] = self.Tree_Bkg['Ele_chIso_sum_MTD_5_Bkg_EE;1'].to_numpy()

    def Calculate_efficiencies(self): # Scan through ch_iso_sum bin values (iso cuts), to calculate the efficiency for full cut range.

        efficiency_Sig_noMTD_EB  , efficiency_Bkg_noMTD_EB ,  efficiency_Sig_noMTD_EE  , efficiency_Bkg_noMTD_EE  = [],[],[],[]

        if self.cut_type == True:

            efficiency_Sig_4sigma_EB , efficiency_Bkg_4sigma_EB , efficiency_Sig_4sigma_EE , efficiency_Bkg_4sigma_EE = [],[],[],[]
            efficiency_Sig_3sigma_EB , efficiency_Bkg_3sigma_EB , efficiency_Sig_3sigma_EE , efficiency_Bkg_3sigma_EE = [],[],[],[]
            efficiency_Sig_2sigma_EB , efficiency_Bkg_2sigma_EB , efficiency_Sig_2sigma_EE , efficiency_Bkg_2sigma_EE = [],[],[],[]

            for i in range( len(self.Sig_hists['Sig_noMTD_EB'][0]) ):

                efficiency_Sig_noMTD_EB.append( sum(self.Sig_hists['Sig_noMTD_EB'][0][0:i+1])/sum(self.Sig_hists['Sig_noMTD_EB'][0]) )
                efficiency_Sig_4sigma_EB.append( sum(self.Sig_hists['Sig_4sigma_EB'][0][0:i+1])/sum(self.Sig_hists['Sig_4sigma_EB'][0]) )
                efficiency_Sig_3sigma_EB.append( sum(self.Sig_hists['Sig_3sigma_EB'][0][0:i+1])/sum(self.Sig_hists['Sig_3sigma_EB'][0]) )
                efficiency_Sig_2sigma_EB.append( sum(self.Sig_hists['Sig_2sigma_EB'][0][0:i+1])/sum(self.Sig_hists['Sig_2sigma_EB'][0]) )

                efficiency_Bkg_noMTD_EB.append( sum(self.Bkg_hists['Bkg_noMTD_EB'][0][0:i+1])/sum(self.Bkg_hists['Bkg_noMTD_EB'][0]) )
                efficiency_Bkg_4sigma_EB.append( sum(self.Bkg_hists['Bkg_4sigma_EB'][0][0:i+1])/sum(self.Bkg_hists['Bkg_4sigma_EB'][0]) )
                efficiency_Bkg_3sigma_EB.append( sum(self.Bkg_hists['Bkg_3sigma_EB'][0][0:i+1])/sum(self.Bkg_hists['Bkg_3sigma_EB'][0]) )
                efficiency_Bkg_2sigma_EB.append( sum(self.Bkg_hists['Bkg_2sigma_EB'][0][0:i+1])/sum(self.Bkg_hists['Bkg_2sigma_EB'][0]) )


                efficiency_Sig_noMTD_EE.append( sum(self.Sig_hists['Sig_noMTD_EE'][0][0:i+1])/sum(self.Sig_hists['Sig_noMTD_EE'][0]) )
                efficiency_Sig_4sigma_EE.append( sum(self.Sig_hists['Sig_4sigma_EE'][0][0:i+1])/sum(self.Sig_hists['Sig_4sigma_EE'][0]) )
                efficiency_Sig_3sigma_EE.append( sum(self.Sig_hists['Sig_3sigma_EE'][0][0:i+1])/sum(self.Sig_hists['Sig_3sigma_EE'][0]) )
                efficiency_Sig_2sigma_EE.append( sum(self.Sig_hists['Sig_2sigma_EE'][0][0:i+1])/sum(self.Sig_hists['Sig_2sigma_EE'][0]) )

                efficiency_Bkg_noMTD_EE.append( sum(self.Bkg_hists['Bkg_noMTD_EE'][0][0:i+1])/sum(self.Bkg_hists['Bkg_noMTD_EE'][0]) )
                efficiency_Bkg_4sigma_EE.append( sum(self.Bkg_hists['Bkg_4sigma_EE'][0][0:i+1])/sum(self.Bkg_hists['Bkg_4sigma_EE'][0]) )
                efficiency_Bkg_3sigma_EE.append( sum(self.Bkg_hists['Bkg_3sigma_EE'][0][0:i+1])/sum(self.Bkg_hists['Bkg_3sigma_EE'][0]) )
                efficiency_Bkg_2sigma_EE.append( sum(self.Bkg_hists['Bkg_2sigma_EE'][0][0:i+1])/sum(self.Bkg_hists['Bkg_2sigma_EE'][0]) )


            self.Sig_iso_eff['Sig_noMTD_EB'] , self.Bkg_iso_eff['Bkg_noMTD_EB'] = efficiency_Sig_noMTD_EB , efficiency_Bkg_noMTD_EB
            self.Sig_iso_eff['Sig_4sigma_EB'] , self.Bkg_iso_eff['Bkg_4sigma_EB'] = efficiency_Sig_4sigma_EB , efficiency_Bkg_4sigma_EB
            self.Sig_iso_eff['Sig_3sigma_EB'] , self.Bkg_iso_eff['Bkg_3sigma_EB'] = efficiency_Sig_3sigma_EB , efficiency_Bkg_3sigma_EB
            self.Sig_iso_eff['Sig_2sigma_EB'] , self.Bkg_iso_eff['Bkg_2sigma_EB'] = efficiency_Sig_2sigma_EB , efficiency_Bkg_2sigma_EB

            self.Sig_iso_eff['Sig_noMTD_EE'] , self.Bkg_iso_eff['Bkg_noMTD_EE'] = efficiency_Sig_noMTD_EE , efficiency_Bkg_noMTD_EE
            self.Sig_iso_eff['Sig_4sigma_EE'] , self.Bkg_iso_eff['Bkg_4sigma_EE'] = efficiency_Sig_4sigma_EE , efficiency_Bkg_4sigma_EE
            self.Sig_iso_eff['Sig_3sigma_EE'] , self.Bkg_iso_eff['Bkg_3sigma_EE'] = efficiency_Sig_3sigma_EE , efficiency_Bkg_3sigma_EE
            self.Sig_iso_eff['Sig_2sigma_EE'] , self.Bkg_iso_eff['Bkg_2sigma_EE'] = efficiency_Sig_2sigma_EE , efficiency_Bkg_2sigma_EE

        else:

            efficiency_Sig_cut1_EB , efficiency_Bkg_cut1_EB , efficiency_Sig_cut1_EE , efficiency_Bkg_cut1_EE = [],[],[],[]
            efficiency_Sig_cut2_EB , efficiency_Bkg_cut2_EB , efficiency_Sig_cut2_EE , efficiency_Bkg_cut2_EE = [],[],[],[]
            efficiency_Sig_cut3_EB , efficiency_Bkg_cut3_EB , efficiency_Sig_cut3_EE , efficiency_Bkg_cut3_EE = [],[],[],[]
            efficiency_Sig_cut5_EB , efficiency_Bkg_cut5_EB , efficiency_Sig_cut5_EE , efficiency_Bkg_cut5_EE = [],[],[],[]

            for i in range( len(self.Sig_hists['Sig_noMTD_EB'][0]) ):

                efficiency_Sig_noMTD_EB.append( sum(self.Sig_hists['Sig_noMTD_EB'][0][0:i+1])/sum(self.Sig_hists['Sig_noMTD_EB'][0]) )
                efficiency_Sig_cut1_EB.append( sum(self.Sig_hists['Sig_cut1_EB'][0][0:i+1])/sum(self.Sig_hists['Sig_cut1_EB'][0]) )
                efficiency_Sig_cut2_EB.append( sum(self.Sig_hists['Sig_cut2_EB'][0][0:i+1])/sum(self.Sig_hists['Sig_cut2_EB'][0]) )
                efficiency_Sig_cut3_EB.append( sum(self.Sig_hists['Sig_cut3_EB'][0][0:i+1])/sum(self.Sig_hists['Sig_cut3_EB'][0]) )
                efficiency_Sig_cut5_EB.append( sum(self.Sig_hists['Sig_cut5_EB'][0][0:i+1])/sum(self.Sig_hists['Sig_cut5_EB'][0]) )

                efficiency_Bkg_noMTD_EB.append( sum(self.Bkg_hists['Bkg_noMTD_EB'][0][0:i+1])/sum(self.Bkg_hists['Bkg_noMTD_EB'][0]) )
                efficiency_Bkg_cut1_EB.append( sum(self.Bkg_hists['Bkg_cut1_EB'][0][0:i+1])/sum(self.Bkg_hists['Bkg_cut1_EB'][0]) )
                efficiency_Bkg_cut2_EB.append( sum(self.Bkg_hists['Bkg_cut2_EB'][0][0:i+1])/sum(self.Bkg_hists['Bkg_cut2_EB'][0]) )
                efficiency_Bkg_cut3_EB.append( sum(self.Bkg_hists['Bkg_cut3_EB'][0][0:i+1])/sum(self.Bkg_hists['Bkg_cut3_EB'][0]) )
                efficiency_Bkg_cut5_EB.append( sum(self.Bkg_hists['Bkg_cut5_EB'][0][0:i+1])/sum(self.Bkg_hists['Bkg_cut5_EB'][0]) )


                efficiency_Sig_noMTD_EE.append( sum(self.Sig_hists['Sig_noMTD_EE'][0][0:i+1])/sum(self.Sig_hists['Sig_noMTD_EE'][0]) )
                efficiency_Sig_cut1_EE.append( sum(self.Sig_hists['Sig_cut1_EE'][0][0:i+1])/sum(self.Sig_hists['Sig_cut1_EE'][0]) )
                efficiency_Sig_cut2_EE.append( sum(self.Sig_hists['Sig_cut2_EE'][0][0:i+1])/sum(self.Sig_hists['Sig_cut2_EE'][0]) )
                efficiency_Sig_cut3_EE.append( sum(self.Sig_hists['Sig_cut3_EE'][0][0:i+1])/sum(self.Sig_hists['Sig_cut3_EE'][0]) )
                efficiency_Sig_cut5_EE.append( sum(self.Sig_hists['Sig_cut5_EE'][0][0:i+1])/sum(self.Sig_hists['Sig_cut5_EE'][0]) )

                efficiency_Bkg_noMTD_EE.append( sum(self.Bkg_hists['Bkg_noMTD_EE'][0][0:i+1])/sum(self.Bkg_hists['Bkg_noMTD_EE'][0]) )
                efficiency_Bkg_cut1_EE.append( sum(self.Bkg_hists['Bkg_cut1_EE'][0][0:i+1])/sum(self.Bkg_hists['Bkg_cut1_EE'][0]) )
                efficiency_Bkg_cut2_EE.append( sum(self.Bkg_hists['Bkg_cut2_EE'][0][0:i+1])/sum(self.Bkg_hists['Bkg_cut2_EE'][0]) )
                efficiency_Bkg_cut3_EE.append( sum(self.Bkg_hists['Bkg_cut3_EE'][0][0:i+1])/sum(self.Bkg_hists['Bkg_cut3_EE'][0]) )
                efficiency_Bkg_cut5_EE.append( sum(self.Bkg_hists['Bkg_cut5_EE'][0][0:i+1])/sum(self.Bkg_hists['Bkg_cut5_EE'][0]) )


            self.Sig_iso_eff['Sig_noMTD_EB'] , self.Bkg_iso_eff['Bkg_noMTD_EB'] = efficiency_Sig_noMTD_EB , efficiency_Bkg_noMTD_EB
            self.Sig_iso_eff['Sig_cut1_EB'] , self.Bkg_iso_eff['Bkg_cut1_EB'] = efficiency_Sig_cut1_EB , efficiency_Bkg_cut1_EB
            self.Sig_iso_eff['Sig_cut2_EB'] , self.Bkg_iso_eff['Bkg_cut2_EB'] = efficiency_Sig_cut2_EB , efficiency_Bkg_cut2_EB
            self.Sig_iso_eff['Sig_cut3_EB'] , self.Bkg_iso_eff['Bkg_cut3_EB'] = efficiency_Sig_cut3_EB , efficiency_Bkg_cut3_EB
            self.Sig_iso_eff['Sig_cut5_EB'] , self.Bkg_iso_eff['Bkg_cut5_EB'] = efficiency_Sig_cut5_EB , efficiency_Bkg_cut5_EB

            self.Sig_iso_eff['Sig_noMTD_EE'] , self.Bkg_iso_eff['Bkg_noMTD_EE'] = efficiency_Sig_noMTD_EE , efficiency_Bkg_noMTD_EE
            self.Sig_iso_eff['Sig_cut1_EE'] , self.Bkg_iso_eff['Bkg_cut1_EE'] = efficiency_Sig_cut1_EE , efficiency_Bkg_cut1_EE
            self.Sig_iso_eff['Sig_cut2_EE'] , self.Bkg_iso_eff['Bkg_cut2_EE'] = efficiency_Sig_cut2_EE , efficiency_Bkg_cut2_EE
            self.Sig_iso_eff['Sig_cut3_EE'] , self.Bkg_iso_eff['Bkg_cut3_EE'] = efficiency_Sig_cut3_EE , efficiency_Bkg_cut3_EE
            self.Sig_iso_eff['Sig_cut5_EE'] , self.Bkg_iso_eff['Bkg_cut5_EE'] = efficiency_Sig_cut5_EE , efficiency_Bkg_cut5_EE

    def Plot_ROC_curves(self,xmin,xmax,ymin,ymax,save: bool):

        if self.cut_type == True:

            plt.plot(self.Sig_iso_eff['Sig_noMTD_EB'],self.Bkg_iso_eff['Bkg_noMTD_EB'], label = 'noMTD')
            plt.plot(self.Sig_iso_eff['Sig_4sigma_EB'],self.Bkg_iso_eff['Bkg_4sigma_EB'], label = '4sigma cut')
            plt.plot(self.Sig_iso_eff['Sig_3sigma_EB'],self.Bkg_iso_eff['Bkg_3sigma_EB'], label = '3sigma cut')
            plt.plot(self.Sig_iso_eff['Sig_2sigma_EB'],self.Bkg_iso_eff['Bkg_2sigma_EB'], label = '2sigma cut')
            plt.legend(loc='best')
            plt.ylim(ymin,ymax)
            plt.xlim(xmin,xmax)
            plt.grid()
            plt.xlabel("Signal efficiency")
            plt.ylabel("Background efficiency")
            plt.title(f'ROC curves for BTL, {self.dz_cut_description}')
            if(save):
                plt.savefig(ROC_plots_directory+f'ROC_curve_BTL_dtsignif_{self.dz_cut_description}')
            plt.show()

            plt.plot(self.Sig_iso_eff['Sig_noMTD_EE'],self.Bkg_iso_eff['Bkg_noMTD_EE'], label = 'noMTD')
            plt.plot(self.Sig_iso_eff['Sig_4sigma_EE'],self.Bkg_iso_eff['Bkg_4sigma_EE'], label = '4sigma cut')
            plt.plot(self.Sig_iso_eff['Sig_3sigma_EE'],self.Bkg_iso_eff['Bkg_3sigma_EE'], label = '3sigma cut')
            plt.plot(self.Sig_iso_eff['Sig_2sigma_EE'],self.Bkg_iso_eff['Bkg_2sigma_EE'], label = '2sigma cut')
            plt.legend(loc='best')
            plt.ylim(ymin,ymax)
            plt.xlim(xmin,xmax)
            plt.grid()
            plt.xlabel("Signal efficiency")
            plt.ylabel("Background efficiency")
            plt.title(f'ROC curves for ETL, {self.dz_cut_description}')
            if(save):
                plt.savefig(ROC_plots_directory+f'ROC_curve_ETL_dtsignif_{self.dz_cut_description}')
            plt.show()

        else:
            # Check the cut definitions in mtdEleIsoValidation.cc file (These are default ones)

            plt.plot(self.Sig_iso_eff['Sig_noMTD_EB'],self.Bkg_iso_eff['Bkg_noMTD_EB'], label = 'noMTD')
            plt.plot(self.Sig_iso_eff['Sig_cut1_EB'],self.Bkg_iso_eff['Bkg_cut1_EB'], label = 'cut1 - 300ps')
            plt.plot(self.Sig_iso_eff['Sig_cut2_EB'],self.Bkg_iso_eff['Bkg_cut2_EB'], label = 'cut2 - 270ps')
            plt.plot(self.Sig_iso_eff['Sig_cut3_EB'],self.Bkg_iso_eff['Bkg_cut3_EB'], label = 'cut3 - 240ps')
            plt.plot(self.Sig_iso_eff['Sig_cut5_EB'],self.Bkg_iso_eff['Bkg_cut5_EB'], label = 'cut5 - 180ps')
            plt.legend(loc='best')
            plt.ylim(ymin,ymax)
            plt.xlim(xmin,xmax)
            plt.grid()
            plt.xlabel("Signal efficiency")
            plt.ylabel("Background efficiency")
            plt.title(f'ROC curves for BTL, {self.dz_cut_description}')
            if(save):
                plt.savefig(ROC_plots_directory+f'ROC_curve_BTL_abs_dt_{self.dz_cut_description}')
            plt.show()

            plt.plot(self.Sig_iso_eff['Sig_noMTD_EE'],self.Bkg_iso_eff['Bkg_noMTD_EE'], label = 'noMTD')
            plt.plot(self.Sig_iso_eff['Sig_cut1_EE'],self.Bkg_iso_eff['Bkg_cut1_EE'], label = 'cut1 - 300ps')
            plt.plot(self.Sig_iso_eff['Sig_cut2_EE'],self.Bkg_iso_eff['Bkg_cut2_EE'], label = 'cut2 - 270ps')
            plt.plot(self.Sig_iso_eff['Sig_cut3_EE'],self.Bkg_iso_eff['Bkg_cut3_EE'], label = 'cut3 - 240ps')
            plt.plot(self.Sig_iso_eff['Sig_cut5_EE'],self.Bkg_iso_eff['Bkg_cut5_EE'], label = 'cut5 - 180ps')
            plt.legend(loc='best')
            plt.ylim(ymin,ymax)
            plt.xlim(xmin,xmax)
            plt.grid()
            plt.xlabel("Signal efficiency")
            plt.ylabel("Background efficiency")
            plt.title(f'ROC curves for ETL, {self.dz_cut_description}')
            if(save):
                plt.savefig(ROC_plots_directory+f'ROC_curve_ETL_abs_dt_{self.dz_cut_description}')
            plt.show()



def Plot_ROC_curves_vs_dz_noMTD(xmin,xmax,ymin,ymax,save:bool,*objects):

    for i in objects:
        plt.plot(i.Sig_iso_eff['Sig_noMTD_EB'],i.Bkg_iso_eff['Bkg_noMTD_EB'],label=f'noMTD {i.dz_cut_description}')
    plt.legend(loc='best')
    plt.ylim(ymin,ymax)
    plt.xlim(xmin,xmax)
    plt.grid()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background efficiency")
    plt.title('ROC curves for BTL, no MTD case')
    if(save):
        plt.savefig(ROC_plots_directory+'ROC_curve_BTL_noMTD_dzCuts')
    plt.show()

    for i in objects:
        plt.plot(i.Sig_iso_eff['Sig_noMTD_EE'],i.Bkg_iso_eff['Bkg_noMTD_EE'],label=f'noMTD {i.dz_cut_description}')
    plt.legend(loc='best')
    plt.ylim(ymin,ymax)
    plt.xlim(xmin,xmax)
    plt.grid()
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background efficiency")
    plt.title('ROC curves for ETL, no MTD case')
    if(save):
        plt.savefig(ROC_plots_directory+'ROC_curve_ETL_noMTD_dzCuts')
    plt.show()




# object.Sig_hists['Sig_noMTD_EB'][1] # -> Array of ch_iso_sum bin values. (cuts for iso efficiency)

def main():

    #obj = MTD_Ele_Iso(Signal_DQM_file,Bakcground_DQM_file,dz_cut description,dt_significance_check(False if abs(dt) check))
    dz010_cut_obj = MTD_Ele_Iso('DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_Sig_dz010_v5.root','DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_Bkg_dz010_v5.root','dz_1mm',True)
    dz020_cut_obj = MTD_Ele_Iso('DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_Sig_dz020_v5.root','DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_Bkg_dz020_v5.root','dz_2mm',True)
    dz030_cut_obj = MTD_Ele_Iso('DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_Sig_dz030_v5.root','DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_Bkg_dz030_v5.root','dz_3mm',True)
    dz040_cut_obj = MTD_Ele_Iso('DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_Sig_dz040_v5.root','DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_Bkg_dz040_v5.root','dz_4mm',True)
    dz050_cut_obj = MTD_Ele_Iso('DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_Sig_dz050_v5.root','DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_Bkg_dz050_v5.root','dz_5mm',True)

    dz010_cut_obj.Read_hists()
    dz020_cut_obj.Read_hists()
    dz030_cut_obj.Read_hists()
    dz040_cut_obj.Read_hists()
    dz050_cut_obj.Read_hists()

    dz010_cut_obj.Calculate_efficiencies()
    dz020_cut_obj.Calculate_efficiencies()
    dz030_cut_obj.Calculate_efficiencies()
    dz040_cut_obj.Calculate_efficiencies()
    dz050_cut_obj.Calculate_efficiencies()

    # .Plot_ROC_curves(xmin,xmax,ymix,ymax,savePlot) -> Saves plot for BTL and ETL parts
    dz010_cut_obj.Plot_ROC_curves(0.75,1.0,0.05,0.4,True)
    dz020_cut_obj.Plot_ROC_curves(0.75,1.0,0.05,0.4,True)
    dz030_cut_obj.Plot_ROC_curves(0.75,1.0,0.05,0.4,True)
    dz040_cut_obj.Plot_ROC_curves(0.75,1.0,0.05,0.4,True)
    dz050_cut_obj.Plot_ROC_curves(0.75,1.0,0.05,0.4,True)


    # This function usually works, but sometimes??? it gives errors -> no idea why. (python version issue?)
    Plot_ROC_curves_vs_dz_noMTD(0.75,1.0,0.05,0.4,True,dz010_cut_obj,dz020_cut_obj,dz030_cut_obj,dz040_cut_obj,dz050_cut_obj) 


if __name__ == "__main__":
   main()

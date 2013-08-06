Usage of the plotting scripts
=============================

General stuff
-------------

1. Keep your input ROOT files in a directory "files/" relative to the location of the scripts
2. The plots will be produced in a directory "plots/" relative to the location of the scripts - for each plotting script a specific output directory was added. 

ROOT
----

###drawplot_eff.C

Produces matching efficiency plots (simtrack to LCT, gem pad vs track pt, pt eta, lct eta, lct number) for even and odd chambers

###drawplot_etastep.C

###drawplot_frankenstein.C	

<pre><code>root -l -b drawplot_frankenstein.C
</code></pre>

Produces a lot of rate plots

<pre><code>Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s-2s1b__gem-3s-2s1b__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s-2s1b__gem-3s-2s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s__gem-3s-2s1b__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s__gem-3s-2s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s-3s1b__gem-3s-2s1b__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s-3s1b__gem-3s-2s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__100-240_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__100-240_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__100-240_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__100-240_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s-2s1b__gem-3s-2s1b__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s-2s1b__gem-3s-2s1b__Frankenstein_pat8__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s__gem-3s-2s1b__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s__gem-3s-2s1b__Frankenstein_pat8__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s-3s1b__gem-3s-2s1b__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__def-3s-3s1b__gem-3s-2s1b__Frankenstein_pat8__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__100-240_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__100-240_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__100-240_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__100-240_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__120-214_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__120-214_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__120-214_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__120-214_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__3s__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__3s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__3s__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__3s_GMT__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__3s_GMT__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__3s_tightGEM__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__3s_tightGEM__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__3s_xtightGEM__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__3s_xtightGEM__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__3s__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__3s__Frankenstein.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__2s__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__2s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__2s_GMT__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__2s_GMT__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__2s_tightGEM__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__2s_tightGEM__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__2s_xtightGEM__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__2s_xtightGEM__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__2s__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__2s__Frankenstein.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__GMT__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__2s3s__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__2s3s123__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__2s3s123__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__164-214_PU100__sequential__3s2s13__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__120-214_PU100__sequential__GMT2s1b__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt/rates__120-214_PU100__sequential__GMT2s1b__Frankenstein_pat2__ratio.png has been created
</code></pre>


###drawplot_frankenstein_ptshift.C

<pre><code>root -l -b drawplot_frankenstein_ptshift.C
</code></pre>

Produces a lot of rate plots - what is the shift?

<pre><code>Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s-2s1b__gem-3s-2s1b__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s-2s1b__gem-3s-2s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s__gem-3s-2s1b__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s__gem-3s-2s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s-3s1b__gem-3s-2s1b__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s-3s1b__gem-3s-2s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__100-240_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__100-240_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__100-240_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__100-240_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__100-214_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__100-214_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__100-214_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__100-214_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s-2s1b__gem-3s-2s1b__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s-2s1b__gem-3s-2s1b__Frankenstein_pat8__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s__gem-3s-2s1b__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s__gem-3s-2s1b__Frankenstein_pat8__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s-3s1b__gem-3s-2s1b__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__def-3s-3s1b__gem-3s-2s1b__Frankenstein_pat8__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__100-240_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__100-240_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__100-240_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__100-240_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__100-214_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__100-214_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__100-214_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__100-214_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__sequential__3s__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__sequential__3s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__sequential__3s__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__sequential__3s_GMT__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__sequential__3s_GMT__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__sequential__3s__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__sequential__3s__Frankenstein.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__sequential__2s__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__sequential__2s1b__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__sequential__2s_GMT__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__sequential__2s_GMT__Frankenstein_pat2__ratio.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__sequential__2s__Frankenstein_pat8.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__sequential__2s__Frankenstein.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__sequential__GMT__Frankenstein_pat2.png has been created
Info in <TCanvas::Print>: png file plots/rate_vs_pt_shift/rates__164-214_PU100__sequential__2s3s__Frankenstein_pat2.png has been created
</code></pre>


###drawplot_frankenstein_ptshiftX.C

Produces a lot of rate plots - what is the shiftX?

<pre><code>root -l -b drawplot_frankenstein_ptshiftX.C
</code></pre>

###drawplot_gmtrt.C	

Used by the frankenstein plotting scripts

###effFunctions.C

Contains definitions of efficiencies

###getPTHistos.C

Auxiliary file

###produceRatePlotsForApproval.C

Produces trigger summary plots for approval

Based on drawplot_frankenstein.C. Produces 4 plots comparing the trigger rate of GMT with CSCTF (n stubs, n stubs with ME1/b, n stubs with ME1/b and GEM); n=2,3; CSCTF track patterns > 2 and > 8

###rootlogon.C	

Some useful definitions for ROOT 

###tdrstyle.C

Contains the recommended style for plots for the TDR 

PyROOT
------

###cuts.py

Contains definitions of cuts used throughout the other PyROOT scripts

###drawplot_eff.py  

Produces a lot of trigger efficiency plots. 

<pre><code>python drawplot_eff.py
</code></pre>

###plotGEMCSCdPhi.py  

Produces plots comparing the bending angle values for high and low pt muons

<pre><code>python plotGEMCSCdPhi.py  
</code></pre>

###produceDphiDict.py	

Uses a set of input files from the GEMCSCAnalyzer for different pt values and builds a python libary file (GEMCSCdPhiDict.py) which can be used in other scripts.

<pre><code>python produceDphiDict.py  
</code></pre>


###tdrStyle.py

Contains the recommended style for plots for the TDR 

###effFunctions.py

Contains definitions of efficiencies





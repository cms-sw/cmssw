Usage of the plotting scripts
=============================

ROOT
----

###drawplot_eff.C

Produces matching efficiency plots (simtrack to LCT, gem pad vs track pt, pt eta, lct eta, lct number) for even and odd chambers

###drawplot_etastep.C

###drawplot_frankenstein.C	

Produces a lot of rate plots. 

<pre><code>root -l -b drawplot_frankenstein.C
</code></pre>

###drawplot_frankenstein_ptshift.C

Produces a lot of rate plots - what is the shift?

<pre><code>root -l -b drawplot_frankenstein_ptshift.C
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





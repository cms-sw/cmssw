# Produce summary plots from the tables generated with writeSummaryTable.r.
#
# Requires gnuplot 4.6. The font path can be set with the environment variable GDFONTPATH.
# 

reset
set macro

#Set the input file here
myfile =  '"<sort -gs -k 2 RelValZMM5312_BP7X_noDRR_summary.txt "'


#Set this to 1 to print plots to png files
print=1

# Output to png viewer
set terminal png enhanced font "Arial,14" size 800,600
set output '| display png:-'


######################################################################
set pointsize 1
set grid

set key reverse Left font ",14" opaque

theta=0
phi=1

set arrow 2 from 180-18,graph 0 to 180-18,graph 1 nohead  back lt 1 lc -1 lw 1
set arrow 3 from 360-18,graph 0 to 360-18,graph 1 nohead  back lt 1 lc -1 lw 1
set arrow 4 from 540-18,graph 0 to 540-18,graph 1 nohead  back lt 1 lc -1 lw 1

#set arrow 2 from 180,graph 0 to 180,graph 1 nohead  back lt 22 lw 1
#set arrow 3 from 360,graph 0 to 360,graph 1 nohead  back lt 22 lw 1
#set arrow 4 from 540,graph 0 to 540,graph 0.85 nohead  back lt 22 lw 1

set xtics ("W-2 MB1" 0, "W-1" 36, "W0" 72, "W1" 108, "W2" 144, \
           "W-2 MB2" 180, "W-1" 216, "W0" 252, "W1" 288, "W2" 324, \
           "W-2 MB3" 360, "W-1" 396, "W0" 432, "W1" 468, "W2" 504, \
           "W-2 MB4" 540, "W-1" 568, "W0" 596, "W1" 624, "W2" 652) ;\
#set xrange[0:]
set xrange[-18:668]
set xtics rotate by -45


set label 1 "MB1" at  72,graph 0.05 center
set label 2 "MB2" at 252,graph 0.05 center
set label 3 "MB3" at 432,graph 0.05 center
set label 4 "MB4" at 596,graph 0.05 center


######################################################################

sl=phi
#sl=theta

if (sl==theta) unset arrow 4



# Fields:
# 1:W 2:St 3:sec 4:SL 5:effS1RPhi 6:effS3RPhi 7:effSeg 8:resHit 9:pullHit 10:meanAngle  11:sigmaAngle 12:meanAngle_pull 13:sigmaAngle_pull 14:meanPos_pull 15:sigmaPos_pull


# We use "set label 99" instead of ylabel, so that it can be right-aligned. 
set ylabel " " 

#### Hit Resolution
if (print) {set output "TrueHitReso.png"}
set yrange [100:900]
set label 99 "True Hit Resolution [{/Symbol m}m]" rotate right at screen 0.02,0.97
plot \
@myfile using (($3)==1&&int($4)==1?$8*10000:1/0) title '{/Symbol f} SLs' w p pt  5 ps 1.4 lc 4, \
@myfile using (($3)==1&&int($4)==2?$8*10000:1/0) title '{/Symbol q} SLs' w p pt 13 ps 1.5 lc 13


#horizontal line for pull and eff plots
set arrow 1 from -18,1 to 668,1 nohead back lt 0 lw 2 lc 0

#### Hit Pull
if (print) {set output "TrueHitPull.png"}
set yrange [0:2]
set label 99 "True Hit Pull" rotate right at screen 0.02,0.97
plot \
@myfile using (($3)==1&&int($4)==1?$9:1/0) title '{/Symbol f} SLs' w p pt  5 ps 1.4 lc 4, \
@myfile using (($3)==1&&int($4)==2?$9:1/0) title '{/Symbol q} SLs' w p pt 13 ps 1.5 lc 13


#### Hit Eff
if (print) {set output "TrueHitEff.png"}
set yrange [0.8:1.05]
set label 99 "True Hit Efficiency" rotate right at screen 0.02,0.97
plot \
@myfile using (($3)==1&&int($4)==1?$5:1/0) title '{/Symbol f} SLs, S1' w p pt  5 ps 1.4 lc 4, \
@myfile using (($3)==1&&int($4)==1?$6:1/0) title '{/Symbol f} SLs, S3' w p pt  4 ps 1.4 lc 4, \
@myfile using (($3)==1&&int($4)==2?$5:1/0) title '{/Symbol q} SLs, S1' w p pt 13 ps 1.5 lc 13, \
@myfile using (($3)==1&&int($4)==2?$6:1/0) title '{/Symbol q} SLs, S3' w p pt 12 ps 1.5 lc 13


### Segment eff
if (print) {set output "TrueSegmentEff.png"}
set yrange [0.8:1.05]
set label 99 "True Segment Efficiency" rotate right at screen 0.02,0.97
plot \
@myfile using (($3)==1&&int($4)==1?$7:1/0) title '4D segment' w p pt  5 ps 1.4 lc 4


### Segment position resolution
unset arrow 1
if (print) {set output "TrueSegmentPosReso.png"}
set yrange [0:500]
set label 99 "Segment Position Resolution [{/Symbol m}m]" rotate right at screen 0.02,0.97
plot \
@myfile using (($3)==1&&int($4)==1?$17*10000:1/0) title '{/Symbol f} SLs' w p pt  5 ps 1.4 lc 4, \
@myfile using (($3)==1&&int($4)==2?$17*10000:1/0) title '{/Symbol q} SLs' w p pt 13 ps 1.5 lc 13


### Segment angular resolution
#unset arrow 1
if (print) {set output "TrueSegmentAngleReso.png"}
set yrange [0:20]
set label 99 "Segment Angle Resolution [mrad]" rotate right at screen 0.02,0.97
plot \
@myfile using (($3)==1&&int($4)==1?$11*1000.:1/0) title '{/Symbol f} SLs' w p pt  5 ps 1.4 lc 4, \
@myfile using (($3)==1&&int($4)==1?$11*10000.:1/0) title '{/Symbol f} SLs (10x)' w p pt 4 ps 1.4 lc 4, \
@myfile using (($3)==1&&int($4)==2?$11*1000.:1/0) title '{/Symbol q} SLs' w p pt 13 ps 1.5 lc 13


### Segment average angle bias
if (print) {set output "TrueSegmentAngleBias.png"}
set arrow 1 from -18,0 to 668,0 nohead back lt 0 lw 2 lc 0
set yrange [-3:3]
set label 99 "Segment Average Angle bias [mrad]" rotate right at screen 0.02,0.97
plot \
@myfile using (($3)==1&&int($4)==1?$10*1000.:1/0) title '{/Symbol f} SLs' w p pt  5 ps 1.4 lc 4, \
@myfile using (($3)==1&&int($4)==2?$10*1000.:1/0) title '{/Symbol q} SLs' w p pt 13 ps 1.5 lc 13

### Segment angular pull width
set arrow 1 from -18,1 to 668,1 nohead back lt 0 lw 2 lc 0
if (print) {set output "TrueSegmentAnglePull.png"}
set yrange [0:2]
set label 99 "Segment Angle Pull Width" rotate right at screen 0.02,0.97
plot \
@myfile using (($3)==1&&int($4)==1?$13:1/0) title '{/Symbol f} SLs' w p pt  5 ps 1.4 lc 4, \
@myfile using (($3)==1&&int($4)==2?$13:1/0) title '{/Symbol q} SLs' w p pt 13 ps 1.5 lc 13

### Segment position pull width
if (print) {set output "TrueSegmentPosPull.png"}
set yrange [0:2]
set label 99 "Segment Position Pull Width" rotate right at screen 0.02,0.97
plot \
@myfile using (($3)==1&&int($4)==1?$15:1/0) title '{/Symbol f} SLs' w p pt  5 ps 1.4 lc 4, \
@myfile using (($3)==1&&int($4)==2?$15:1/0) title '{/Symbol q} SLs' w p pt 13 ps 1.5 lc 13

### Number of segment ratio
if (print) {set output "TrueNSegRatio.png"}
set yrange [0:0.15]
set label 99 "(ev with more than 2 seg)/(ev with more than 1 seg) " rotate right at screen 0.02,0.97
plot \
@myfile using (($3)==1&&int($4)==1?$18:1/0) title '4D segment' w p pt  5 ps 1.4 lc 4


### p0
if (print) {set output "TrueP0.png"}
set yrange [0:0.8]
set label 99 "p0" rotate right at screen 0.02,0.97
plot \
@myfile using (($3)==1&&int($4)==1?$19:1/0) title '{/Symbol f} SLs' w p pt  5 ps 1.4 lc 4, \
@myfile using (($3)==1&&int($4)==2?$19:1/0) title '{/Symbol q} SLs' w p pt 13 ps 1.5 lc 13

### p1
if (print) {set output "TrueP1.png"}
set yrange [-1:1]
set label 99 "p0" rotate right at screen 0.02,0.97
plot \
@myfile using (($3)==1&&int($4)==1?$20:1/0) title '{/Symbol f} SLs' w p pt  5 ps 1.4 lc 4, \
@myfile using (($3)==1&&int($4)==2?$20:1/0) title '{/Symbol q} SLs' w p pt 13 ps 1.5 lc 13
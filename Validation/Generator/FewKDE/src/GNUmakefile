#############################################################
# Makefile for FewKDE package
# FewKDE/GNUmakefile
############################################################

OBJS =   Kernel.o SimpleKDE.o AdaptiveKDE.o FewKDE.o

all:   	lib

lib:	../../lib/libFewKDE.a

../../lib/libFewKDE.a:	${OBJS}
	echo '' > libFewKDE.a; rm libFewKDE.a;
	${AR} rcv libFewKDE.a ${OBJS}
	mv libFewKDE.a ../../lib/

.cc.o:; ${CC} ${FLAGS} -c $*.cc

clean:;	rm *.o


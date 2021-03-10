all:
	make -C TRAF
	make -C GOL
	make -C STUT
	make -C GEN
	make -C COLI
	make -C NBD

clean:
	make -C TRAF clean 
	make -C GOL clean 
	make -C STUT clean 
	make -C GEN clean 
	make -C COLI clean 
	make -C NBD clean 

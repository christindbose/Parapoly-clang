all:
	make -C TRAF
	make -C GOL
	make -C STUT
	make -C GEN
	make -C COLI
	make -C NBD
	make -C BFS-vE
	make -C CC-vE
	make -C PR-vE
	make -C BFS-vEN
	make -C CC-vEN
	make -C PR-vEN
	make -C RAY

clean:
	make -C TRAF clean 
	make -C GOL clean 
	make -C STUT clean 
	make -C GEN clean 
	make -C COLI clean 
	make -C NBD clean 
	make -C BFS-vE clean
	make -C CC-vE clean
	make -C PR-vE clean
	make -C BFS-vEN clean
	make -C CC-vEN clean
	make -C PR-vEN clean
	make -C RAY clean

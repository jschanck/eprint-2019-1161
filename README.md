The .qasm files in this directory can be compiled into either png of pdf files by running ./qasm2png file.qasm or ./qasm2pdf file.qasm respectively.
However qasm2pdf seems quite shell specific and qasm2png requires some non standard commands, so the pdfs required to compile main.tex have been added.

The popcnt.py file, using some standard assumptions regarding vectors sampled i.i.d. from the sphere, tries to estimate how many tested pairs would pass a popcnt (as described in Ducas18).

Run as:

python popcnt.py lattice_dim number_of_lattice_points number_of_popcnt_vectors popcnt_threshod

(it is very much a work in progress.)

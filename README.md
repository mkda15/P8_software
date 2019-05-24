# P8_software
This repository includes software that has been developed and used at the 8th semester project of Mathematical Engineering at Aalborg University. The software is written in Python, and the overall goal has been to estimate the channel matrix <strong>G</strong> in the equation <strong>Y</strong> = <strong>S</strong> &middot; <strong>G</strong> + <strong>W</strong>.

Scripts included in the repository are:
<ul>
  <li>Simulation.py, which is used to simulate a narrowband channel with various possible settings. The outputs are Y, S, G and W.</li>
  <li>CISTA.py, which is an implementation of the Iterative Soft-Threshold Algorithm that has been further developed to handle complex numbers as well.</li>
  <li>SCISTA.py, which is an implementation of a neural network that uses B-splines to learn the parameters in CISTA.</li>
  <li>GCISTA.py, which is similar to SCISTA.py but uses the Gaussian Radial Basis Functions instead of B-splines.
</ul>

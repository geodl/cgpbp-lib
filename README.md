# cgpbp-lib
Modified Version of the CGP Library and Fast Artificial Neural Network Library
======

Hybridization of Cartesian Genetic Programming and Backpropagation for Generating Classifiers based on Artificial Neural Networks.   
It includes the CGPBP-IN and CGPBP-OUT methods.

Author: Johnathan M Melo Neto   
Email: jmmn.mg@gmail.com

Credits of the original libraries are placed below.

CGP Library
======

A cross platform Cartesian Genetic Programming Library written in C.

Author: Andrew James Turner    
Webpage: http://www.cgplibrary.co.uk/     
Email: andrew.turner@york.ac.uk    
License: Lesser General Public License (LGPL) 

If this library is used in published work I would greatly appreciate a citation to the following:  

A. J. Turner and J. F. Miller. [**Introducing A Cross Platform Open Source Cartesian Genetic Programming Library**](http://andrewjamesturner.co.uk/files/GPEM2014.pdf). The Journal of Genetic Programming and Evolvable Machines, 2014, 16, 83-91.


Fast Artificial Neural Network Library (FANN)
======

**Fast Artificial Neural Network (FANN) Library** is a free open source neural network library, which implements multilayer artificial neural networks in C with support for both fully connected and sparsely connected networks.

## To Install

### On Linux

#### From Source


First you'll want to clone the repository:

`git clone https://github.com/johnathamelo/cgpbp-lib.git`

Once that's finished, navigate to the Root directory. In this case it would be ./cgpbp-lib:

`cd ./cgpbp-lib`

Then run CMake:

`cmake .`

After that, you'll need to use elevated privileges to install the library:

`sudo make install`

Then run:

`sudo ldconfig`

Navigate to the ./cgpbp:

`cd ./cgpbp`

Then run Makefile:

`make main`

Now you can run CGPBP algorithms by running:

`./main`

## To Learn More

For more information about FANN, please refer to the [FANN website](http://leenissen.dk/fann/wp/)
For more information about CGP-Library, please refer to the [CGP-Library website](http://www.cgplibrary.co.uk/)

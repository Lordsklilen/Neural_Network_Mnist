# Neural network Implementation

The purpose of the application is to recognize hand-written numbers from the database available here:
http://yann.lecun.com/exdb/mnist/

I based on the project orginaly made by louisjc:
https://github.com/louisjc/mnist-neural-network?fbclid=IwAR1YwEq2yZJHgMTsnPbHXkhz5DLTJTF6JLmgPc5CQQk4ZcR1iaAVkmlmMgo

This project was created for academic purposes

### Installation
To install you need download third party library - numpy:
```sh
pip install scipy numpy
```
About numpy:
http://www.numpy.org/

Run by executing neural_net.py
```sh
python neural_net.py
```
### Todos
 - finish SGD method
 - Add more training methods(Momentum, ADAM)

License
----

MIT

Yann LeCun and Corinna Cortes hold the copyright of MNIST database, which is a derivative work from original NIST datasets. MNIST database is made available under the terms of the Creative Commons Attribution-Share Alike 3.0 license.

The files data_testing and data_training contain the MNIST database saved using Python pickle. They are licensed under Creative Commons Attribution-Share Alike 3.0.

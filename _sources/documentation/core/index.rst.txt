Core
====





.. toctree::
   :maxdepth: 1
   :caption: Contents:

   generics
   generators
   mappers
   processors
   devices
   filters
   channels
   impairments 
   compensators 
   monitors 
   metrics
   visualizers

Processor Vs Compensator
------------------------

The comnumpy library distinguishes between two types of signal processing components: Processors and Compensators. 

This separation is based on the nature of their operations and how they interact with the input signals.

- **Processors**: These components apply fixed transformations to the input signals, regardless of the signal's content. Examples include amplifiers and clippers. 

- **Compensators**: These components, on the other hand, adapt their behavior based on the input signal to achieve a desired output characteristic. Examples include normalizers and DC offset correctors. 

   


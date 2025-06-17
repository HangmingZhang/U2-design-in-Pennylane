#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pennylane as qml
from pennylane import numpy as np
import itertools


# In[2]:


class PauliTwirling:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
    
    def twirlinglayer(self, measurements_str: str, pauli_op_n_qubits: list,
                      noise_channel: "qml.QubitChannel function", channel_paras, channel_wires,
                      circuit: "function", *args, **kwargs):
        # channel_paras: it could be a float when the noise_channel is defined by qml.AmplitudeDamping
        # or a list when the noise_channel is defined by qml.QubitChannel.
        # *args, **kwargs: parameters of circuit.
        
        measurements_split = list(measurements_str)
        
        measurements = self.__save_measurements(measurements_split)
        
        circuit(*args, **kwargs)
        
        # Here’s the direct implementation of the Pauli twirling method
        # —it’s based on the mathematical formulation itself
        # and may differ slightly from practical approaches.
        # However, our main goal here is to use the code
        # to verify the mathematical principles behind Pauli twirling.
        
        self.__add_twirling_layer(pauli_op_n_qubits)
        
        noise_channel(channel_paras, channel_wires)
        
        self.__add_twirling_layer(pauli_op_n_qubits)
        
        return qml.expval(self.__create_measurement_ops(measurements))
    
    def paulitwirling(self, dev: "qml.device for paulitwirling", measurements_str: str,
                     noise_channel: "qml.QubitChannel function", channel_paras, channel_wires,
                      circuit: "function", *args, **kwargs):
        twirled_expectation_value_lst = []
        pauli_ops_n_qubits = self.__get_pauli_ops_n_qubits()
        
        for pauli_op_n_qubits in pauli_ops_n_qubits:
            e = qml.QNode(self.twirlinglayer, dev)(measurements_str, pauli_op_n_qubits,
                                            noise_channel, channel_paras, channel_wires,
                                            circuit, *args, **kwargs)
            twirled_expectation_value_lst.append(e)
            
        e = np.mean(twirled_expectation_value_lst)
        return e
    
       
    def __save_measurements(self, measurements_split):
        measurements = []
        for i, measurement_str in enumerate(measurements_split):
            if measurement_str == "I":
                continue
            elif measurement_str == "X":
                measurements.append(qml.PauliX(i))
            elif measurement_str == "Y":
                measurements.append(qml.PauliY(i))
            elif measurement_str == "Z":
                measurements.append(qml.PauliZ(i))
            else:
                raise ValueError("measurements_str can only be 'I', 'X', 'Y' or 'Z'.")
        return measurements
    
    def __create_measurement_ops(self, measurements):
        measurement_ops = measurements[0]
        for i in range(1, len(measurements)):
            measurement_ops = measurement_ops @ measurements[i]
        return measurement_ops
    
    
    def __get_pauli_ops_n_qubits(self):
        pauli_ops = ["I", "X", "Y", "Z"]
        pauli_ops_n_qubits = list(itertools.product(pauli_ops, repeat=self.n_qubits))
        return pauli_ops_n_qubits
    
    def __add_twirling_layer(self, pauli_op_n_qubits):
        for i, pauli_op in enumerate(pauli_op_n_qubits):
            if pauli_op == "I":
                qml.Identity(wires=i)
                
            elif pauli_op == "X":
                qml.PauliX(wires=i)
                    
            elif pauli_op == "Y":
                qml.PauliY(wires=i)
                    
            elif pauli_op == "Z":
                qml.PauliZ(wires=i)
                
            else:
                print("Unexpected error!")        


# In[3]:


class U2Desgin:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
    
    def u2dlayer(self, measurements_str: str, u2_op_n_qubits: list,
                      noise_channel: "qml.QubitChannel function", channel_paras, channel_wires,
                      circuit: "function", *args, **kwargs):
        # channel_paras: it could be a float when the noise_channel is defined by qml.AmplitudeDamping
        # or a list when the noise_channel is defined by qml.QubitChannel.
        # *args, **kwargs: parameters of circuit.
        # u2_op_n_qubits: one ("clifford_gate_10" \otimes "clifford_gate_4")
        
        measurements_split = list(measurements_str)
        
        measurements = self.__save_measurements(measurements_split)
        
        circuit(*args, **kwargs)
        
        # Here’s the direct implementation of the Pauli twirling method
        # —it’s based on the mathematical formulation itself
        # and may differ slightly from practical approaches.
        # However, our main goal here is to use the code
        # to verify the mathematical principles behind Pauli twirling.
        
        self.__add_u2desgin_layer(u2_op_n_qubits)
        
        noise_channel(channel_paras, channel_wires)
        
        self.__add_u2desgin_dagger_layer(u2_op_n_qubits)
        
        return qml.expval(self.__create_measurement_ops(measurements))
    
    def u2desgin(self, dev: "qml.device for u2 desgin", measurements_str: str,
                     noise_channel: "qml.QubitChannel function", channel_paras, channel_wires,
                      circuit: "function", *args, **kwargs):
        u2d_expectation_value_lst = []
        u2d_ops_n_qubits = self.__get_u2d_ops_n_qubits()  
        # 2 qubits ("clifford_gate_10" \otimes "clifford_gate_4") ... (...)
        
        for u2d_op_n_qubits in u2d_ops_n_qubits:
            e = qml.QNode(self.u2dlayer, dev)(measurements_str, u2d_op_n_qubits,
                                            noise_channel, channel_paras, channel_wires,
                                            circuit, *args, **kwargs)
            u2d_expectation_value_lst.append(e)
        e = np.mean(u2d_expectation_value_lst)
        
        return e
    
       
    def __save_measurements(self, measurements_split):
        measurements = []
        for i, measurement_str in enumerate(measurements_split):
            if measurement_str == "I":
                continue
            elif measurement_str == "X":
                measurements.append(qml.PauliX(i))
            elif measurement_str == "Y":
                measurements.append(qml.PauliY(i))
            elif measurement_str == "Z":
                measurements.append(qml.PauliZ(i))
            else:
                raise ValueError("measurements_str can only be 'I', 'X', 'Y' or 'Z'.")
        return measurements
    
    def __create_measurement_ops(self, measurements):
        measurement_ops = measurements[0]
        for i in range(1, len(measurements)):
            measurement_ops = measurement_ops @ measurements[i]
        return measurement_ops
    
    
    def __get_u2d_ops_n_qubits(self):

        u2d_ops = ["clifford_gate_1", "clifford_gate_2", "clifford_gate_3", "clifford_gate_4",
                  "clifford_gate_5", "clifford_gate_6", "clifford_gate_7", "clifford_gate_8",
                  "clifford_gate_9", "clifford_gate_10", "clifford_gate_11", "clifford_gate_12",
                  "clifford_gate_13", "clifford_gate_14", "clifford_gate_15", "clifford_gate_16",
                  "clifford_gate_17", "clifford_gate_18", "clifford_gate_19", "clifford_gate_20",
                  "clifford_gate_21", "clifford_gate_22", "clifford_gate_23", "clifford_gate_24",]
        u2d_ops_n_qubits = list(itertools.product(u2d_ops, repeat=self.n_qubits))
        return u2d_ops_n_qubits
    
    def __add_u2desgin_layer(self, u2d_ops_n_qubits):
        # u2d_ops_n_qubits: one clifford gate on n qubits ("clifford_gate_10" \otimes "clifford_gate_4")
        for i, u2d_op in enumerate(u2d_ops_n_qubits):  # u2d_op： clifford_gate_10
            if u2d_op == "clifford_gate_1":
                qml.Identity(wires=i)
                
            elif u2d_op == "clifford_gate_2":
                qml.PauliX(wires=i)
                    
            elif u2d_op == "clifford_gate_3":
                qml.PauliY(wires=i)
                    
            elif u2d_op == "clifford_gate_4":
                qml.PauliZ(wires=i)
                
            elif u2d_op == "clifford_gate_5":
                qml.Hadamard(wires=i)
                
            elif u2d_op == "clifford_gate_6":
                qml.S(wires=i)
            
            elif u2d_op == "clifford_gate_7":
                qml.PauliX(wires=i)
                qml.Hadamard(wires=i)
            
            elif u2d_op == "clifford_gate_8":
                qml.PauliX(wires=i)
                qml.S(wires=i)
            
            elif u2d_op == "clifford_gate_9":
                qml.PauliY(wires=i)
                qml.Hadamard(wires=i)
            
            elif u2d_op == "clifford_gate_10":
                qml.PauliY(wires=i)
                qml.S(wires=i)
                
            elif u2d_op == "clifford_gate_11":
                qml.PauliZ(wires=i)
                qml.Hadamard(wires=i)
                
            elif u2d_op == "clifford_gate_12":
                qml.PauliZ(wires=i)
                qml.S(wires=i)
                
            elif u2d_op == "clifford_gate_13":
                qml.Hadamard(wires=i)
                qml.S(wires=i)
                
            elif u2d_op == "clifford_gate_14":
                qml.S(wires=i)
                qml.Hadamard(wires=i)
                
            elif u2d_op == "clifford_gate_15":
                qml.PauliX(wires=i)
                qml.Hadamard(wires=i)
                qml.S(wires=i)
                
            elif u2d_op == "clifford_gate_16":
                qml.PauliX(wires=i)
                qml.S(wires=i)
                qml.Hadamard(wires=i)
                
            elif u2d_op == "clifford_gate_17":
                qml.PauliY(wires=i)
                qml.Hadamard(wires=i)
                qml.S(wires=i)
                
            elif u2d_op == "clifford_gate_18":
                qml.PauliY(wires=i)
                qml.S(wires=i)
                qml.Hadamard(wires=i)
                
            elif u2d_op == "clifford_gate_19":
                qml.PauliZ(wires=i)
                qml.Hadamard(wires=i)
                qml.S(wires=i)
                
            elif u2d_op == "clifford_gate_20":
                qml.PauliZ(wires=i)
                qml.S(wires=i)
                qml.Hadamard(wires=i)
                
            elif u2d_op == "clifford_gate_21":
                qml.Hadamard(wires=i)
                qml.S(wires=i)
                qml.Hadamard(wires=i)
                
            elif u2d_op == "clifford_gate_22":
                qml.PauliX(wires=i)
                qml.Hadamard(wires=i)
                qml.S(wires=i)
                qml.Hadamard(wires=i)
            
            elif u2d_op == "clifford_gate_23":
                qml.PauliY(wires=i)
                qml.Hadamard(wires=i)
                qml.S(wires=i)
                qml.Hadamard(wires=i)
            
            elif u2d_op == "clifford_gate_24":
                qml.PauliZ(wires=i)
                qml.Hadamard(wires=i)
                qml.S(wires=i)
                qml.Hadamard(wires=i)
                
            else:
                print("Unexpected error!")
                
    def __add_u2desgin_dagger_layer(self, u2d_ops_n_qubits):
        # u2d_ops_n_qubits: one clifford gate on n qubits ("clifford_gate_10" \otimes "clifford_gate_4")
        for i, u2d_op in enumerate(u2d_ops_n_qubits):  # u2d_op： clifford_gate_10
            if u2d_op == "clifford_gate_1":
                qml.Identity(wires=i)
                
            elif u2d_op == "clifford_gate_2":
                qml.PauliX(wires=i)
                    
            elif u2d_op == "clifford_gate_3":
                qml.PauliY(wires=i)
                    
            elif u2d_op == "clifford_gate_4":
                qml.PauliZ(wires=i)
                
            elif u2d_op == "clifford_gate_5":
                qml.Hadamard(wires=i)
                
            elif u2d_op == "clifford_gate_6":
                qml.adjoint(qml.S)(wires=i)
            
            elif u2d_op == "clifford_gate_7":
                qml.Hadamard(wires=i)
                qml.PauliX(wires=i)
            
            elif u2d_op == "clifford_gate_8":
                qml.adjoint(qml.S)(wires=i)
                qml.PauliX(wires=i)
            
            elif u2d_op == "clifford_gate_9":
                qml.Hadamard(wires=i)
                qml.PauliY(wires=i)
            
            elif u2d_op == "clifford_gate_10":
                qml.adjoint(qml.S)(wires=i)
                qml.PauliY(wires=i)
                
            elif u2d_op == "clifford_gate_11":
                qml.Hadamard(wires=i)
                qml.PauliZ(wires=i)
                
            elif u2d_op == "clifford_gate_12":
                qml.adjoint(qml.S)(wires=i)
                qml.PauliZ(wires=i)
                
            elif u2d_op == "clifford_gate_13":
                qml.adjoint(qml.S)(wires=i)
                qml.Hadamard(wires=i)
                
            elif u2d_op == "clifford_gate_14":
                qml.Hadamard(wires=i)
                qml.adjoint(qml.S)(wires=i)
                
            elif u2d_op == "clifford_gate_15":
                qml.adjoint(qml.S)(wires=i)
                qml.Hadamard(wires=i)
                qml.PauliX(wires=i)
                
            elif u2d_op == "clifford_gate_16":
                qml.Hadamard(wires=i)
                qml.adjoint(qml.S)(wires=i)
                qml.PauliX(wires=i)
                
            elif u2d_op == "clifford_gate_17":
                qml.adjoint(qml.S)(wires=i)
                qml.Hadamard(wires=i)
                qml.PauliY(wires=i)
                
            elif u2d_op == "clifford_gate_18":
                qml.Hadamard(wires=i)
                qml.adjoint(qml.S)(wires=i)
                qml.PauliY(wires=i)
                
            elif u2d_op == "clifford_gate_19":
                qml.adjoint(qml.S)(wires=i)
                qml.Hadamard(wires=i)
                qml.PauliZ(wires=i)
                
            elif u2d_op == "clifford_gate_20":
                qml.Hadamard(wires=i)
                qml.adjoint(qml.S)(wires=i)
                qml.PauliZ(wires=i)
                
            elif u2d_op == "clifford_gate_21":
                qml.Hadamard(wires=i)
                qml.adjoint(qml.S)(wires=i)
                qml.Hadamard(wires=i)
                
            elif u2d_op == "clifford_gate_22":
                qml.Hadamard(wires=i)
                qml.adjoint(qml.S)(wires=i)
                qml.Hadamard(wires=i)
                qml.PauliX(wires=i)
            
            elif u2d_op == "clifford_gate_23":
                qml.Hadamard(wires=i)
                qml.adjoint(qml.S)(wires=i)
                qml.Hadamard(wires=i)
                qml.PauliY(wires=i)
            
            elif u2d_op == "clifford_gate_24":
                qml.Hadamard(wires=i)
                qml.adjoint(qml.S)(wires=i)
                qml.Hadamard(wires=i)
                qml.PauliZ(wires=i)
                
            else:
                print("Unexpected error!")
            
        


# In[ ]:





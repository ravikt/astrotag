import numpy as np
import galois as gl
from . import basereedsolomon
import reedsolo
import unittest


class ReedSolomonCodec:
    def __init__(self, n, k):
        """
        Initialize Reed-Solomon codec with parameters:
        n: total number of symbols in codeword
        k: number of data symbols (message length)
        """
        self.n = n  # codeword length
        self.k = k  # message length
        self.rs = reedsolo.RSCodec(n - k)
    
    def encode(self, message):
        """
        Encode a message (binary string or list of integers) using Reed-Solomon.
        Returns the encoded message as a list of integers.
        """
        # Convert binary string to bytes if necessary
        if isinstance(message, str):
            # Ensure message length is correct
            if len(message) != self.k * 8:
                raise ValueError(f"Message must be exactly {self.k * 8} bits long")
            
            # Convert groups of 8 bits to integers
            message_symbols = []
            for i in range(0, len(message), 8):
                if i + 8 <= len(message):
                    byte = int(message[i:i+8], 2)
                    message_symbols.append(byte)
        else:
            message_symbols = message

        # Ensure we have the correct number of symbols
        if len(message_symbols) != self.k:
            raise ValueError(f"Message must be exactly {self.k} symbols long")
            
        # Encode the message
        try:
            encoded_message = self.rs.encode(bytearray(message_symbols))
            return list(encoded_message)
        except Exception as e:
            raise RuntimeError(f"Encoding failed: {str(e)}")

    def decode(self, codeword):
        """
        Decode a Reed-Solomon codeword and return the original message symbols.
        """
        try:
            # Decode the message and extract only the data portion
            decoded_message, _, _ = self.rs.decode(bytearray(codeword))
            return list(decoded_message[:self.k])
        except Exception as e:
            raise RuntimeError(f"Decoding failed: {str(e)}")



    @staticmethod
    def symbols_to_bits(symbols):
        """
        Convert a list of symbols (integers or bytes) to a binary string.
        """
        bits = []
        for symbol in symbols:
            if isinstance(symbol, (bytearray, bytes)):
                bits.extend(f'{byte:08b}' for byte in symbol)
            else:
                bits.append(f'{symbol:08b}')
        return ''.join(bits)

    @staticmethod
    def bits_to_symbols(bits):
        """
        Convert a binary string to a list of symbols (integers).
        """
        if len(bits) % 8 != 0:
            raise ValueError("The length of the binary string must be a multiple of 8")
        
        symbols = []
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            symbols.append(int(byte, 2))
        
        return symbols

class Generalized_Reed_Solomon(basereedsolomon.Base_Reed_Solomon):

    def __init__(self, field_size:int, message_length:int, payload_length:int,symbol_size:int,p_factor:int,irr_poly=None,multi_processing = False,debug=True) -> None:
        super().__init__(field_size,message_length,payload_length,symbol_size,multi_processing,irr_poly,debug,p_factor)    
        self.p = p_factor
        self.primitive = self.galois_field.primitive_element #gl.primitive_root(1)
            
        self.helper.debug_print("get unity root")
        self.p_unity = self.helper.get_nth_unity_root_of_field(p_factor)
        self.generator_set = []

        ## handles cases where either k or r are not divisible by p, implies by extension the n may not be divisible by p
        self.d = self.two_s % self.p #remainder of r
        self.c = self.payload_length % self.p #remainder of k
        self.num_of_padded_zeros =  (self.p - self.c) if self.c != 0 else 0
    
        #adjust parameters to handle r not divisble by p
        if self.d != 0:
            raise ValueError("r % p  != 0 is so far not implemented")
            self.T = self.calculate_t_matrix()
            self.V_inverse = np.linalg.inv(self.calculate_v_matrix())
            self.tv_transformation_matrix = self.return_transformation_values_for_f(self.T,self.V_inverse)
        
        for index in range(0,p_factor):
            self.helper.debug_print("init generator:",index)
            g_i = self.return_generalized_generator(index)
            self.helper.debug_print("roots",g_i)
            self.generator_set.append(g_i)
        
        
        # divisibility of n, k and r should be checked
    
    def encode(self,message):
        if len(message) != self.payload_length:
            raise ValueError("Length not as specified")
        self.helper.debug_print("encode input message:",message,len(message))
        if self.c != 0 :
            # add conceptual zeros
            message = ([0]*self.num_of_padded_zeros) + message 
            
        self.helper.debug_print("before encode",message,len(message))
        msg = self.encode_classic(message)
        self.helper.debug_print("before cutoff",msg,len(msg))
        msg = msg[self.num_of_padded_zeros:len(msg)] #+ msg[self.payload_length + self.num_of_padded_zeros:len(msg)]            
        return msg

    def decode(self, received):
        """Main decoding procedure"""
        # Calculate syndromes
        syndromes = self.calc_syndrome(received)
        if all(s == 0 for s in syndromes):
            return received
        
        # Find error locator polynomial
        error_locator = self.berlekamp_massey(syndromes)
        
        # Find error positions
        error_positions = self.modified_chien_search(error_locator)
        
        # Calculate error values
        error_values = self.forney_algorithm(error_locator, syndromes, error_positions)
        
        # Correct errors
        result = list(received)
        for i, value in enumerate(error_values):
            if value != 0:
                result[i] ^= value
                
        return result
    
    def berlekamp_massey(self, syndromes):
        """Fixed Berlekamp-Massey algorithm implementation"""
        n = len(syndromes)
        L = 0  # Current error locator polynomial degree
        C = [1] + [0] * (n - 1)  # Current error locator polynomial
        B = [1] + [0] * (n - 1)  # Previous error locator polynomial
        
        for i in range(n):
            # Calculate discrepancy with bounds checking
            delta = syndromes[i]
            for j in range(1, L + 1):
                if i - j >= 0 and j < len(C):
                    delta ^= self.galois_multiply(C[j], syndromes[i - j])
            
            # Update polynomials
            if delta != 0:
                T = C[:]  # Save current C
                # Update C with proper length
                for j in range(len(B)):
                    idx = i - L + j
                    if 0 <= idx < len(C):
                        C[idx] ^= B[j]
                if 2 * L <= i:
                    L = i + 1 - L
                    B = T[:]
        
        return C[:L + 1]  # Return valid coefficients

    # def decode(self, recieved_msg):
    #     if len(recieved_msg) != (self.message_length):
    #         self.helper.debug_print("decode input message",recieved_msg,len(recieved_msg))
    #         raise ValueError("Length not as specified")
    #     if self.c != 0 :
    #         # add conceptual zeros
    #         #recieved_msg =  recieved_msg[0:self.payload_length] +([0]*self.num_of_padded_zeros) + recieved_msg[self.payload_length:len(recieved_msg)]
    #         recieved_msg = ([0]*self.num_of_padded_zeros) + recieved_msg
    #     corrected_msg = self.decode_classic(recieved_msg)
    #     self.helper.debug_print("after decode internal",corrected_msg)
    #     return self.return_info_symbols(corrected_msg)

    def encode_classic(self,message):
        #only used in case where self.d != 0
        coefficent_vector = []

        output_message = message

        message_matrix = self.input_arr_to_matrix(message)
        l = self.two_s//self.p
        self.helper.debug_print("messgae matrix before",message_matrix)
        #performs shift by x**l
        temp_message_matrix = []
        for row_id in  range(0,message_matrix.shape[0]):
            msg_slice = np.copy(message_matrix[row_id])
            msg_slice = np.append(msg_slice,np.zeros(l, dtype = int))
            self.helper.debug_print("msg_slice_poly",gl.Poly(msg_slice,field=self.galois_field))
            temp_message_matrix.append(msg_slice)
        message_matrix = np.array(temp_message_matrix)
        self.helper.debug_print("message matrix after",message_matrix)
        f_values = []

        fft = self.helper.fft_on_matrix_multi(message_matrix) if self.multi else self.helper.fft_on_matrix(message_matrix)
        self.helper.debug_print("fft",self.galois_field(fft))
        for f_index in range(0,self.p):
            f_i = gl.Poly(fft[f_index],field=self.galois_field)
            self.helper.debug_print("fi",f_i,fft[f_index],f_i% self.generator_set[f_index] )
            f_values.append((f_i % self.generator_set[f_index]).coeffs.tolist())

        
        

        #calculates adjusted f
        if  self.d != 0:
            #get coefficents of first d f values and remove them from vec
            for coeff_index in range(0,self.d):
                #gets coeff
                coeff_val = f_values[coeff_index][0] 
                #removes coeff
                f_values[coeff_index] = f_values[coeff_index][1:len(f_values[coeff_index])]
                coefficent_vector.append(coeff_val)
            coefficent_vector = self.galois_field(coefficent_vector)
            gamma_adjustments = - np.dot(self.tv_transformation_matrix,coefficent_vector) # - since f adj = f - gamma
            h_adjustments = np.dot(self.V_inverse,gamma_adjustments[0:self.d]) #used later to calc h adjusted

            for adjustment_f_index in range(self.d,self.p):
                
                f_i_to_be_adjusted = [int(gamma_adjustments[adjustment_f_index])] + ([0]*l)
                gamma_remainder = (gl.Poly(f_i_to_be_adjusted,field=self.galois_field) % self.generator_set[adjustment_f_index])
                f_values[adjustment_f_index] =  (gl.Poly(f_values[adjustment_f_index],field=self.galois_field) + gamma_remainder).coeffs

            self.helper.debug_print("coffs",coefficent_vector)
            self.helper.debug_print("gamma",gamma_adjustments)
            self.helper.debug_print("h adj",h_adjustments)
        self.helper.debug_print("fvalue",f_values)
        parity_values =[]
        
        ifft = self.helper.ifft_on_matrix(f_values) if self.multi else self.helper.ifft_on_matrix(f_values)
        self.helper.debug_print("ifft",ifft)
        for h_index in range(0,self.p):
            h_i = ifft[h_index]
            #calculates adjusted h
            if  self.d != 0:
                if h_index < self.d:
                    h_i = [int(h_adjustments[h_index])]+h_i
                
            parity_values.append(h_i)
        
        self.helper.debug_print("parity_values",parity_values)
        #for parity_value_index in range(0,l):
        #    for parity_array_index in range(0,self.p):
        #        output_message.append(int(parity_values[self.p-parity_array_index-1][parity_value_index]))
        output_message = self.append_parity_symbols(output_message,parity_values)
        self.helper.debug_print("output message",output_message)

        return output_message
    
    def decode_classic(self, received_msg):
        """Use fixed BM algorithm for decoding"""
        try:
            # Calculate syndromes
            syndromes = self.calc_syndrome(received_msg)
            if np.all(syndromes == self.galois_field(0)):
                return received_msg
                
            # Get error locator polynomial
            error_locator = self.berlekamp_massey(syndromes)
            
            # Find error positions using Chien search
            error_positions = []
            for i in range(len(received_msg)):
                # Evaluate error locator at position i
                sum = 0
                for j, coeff in enumerate(error_locator):
                    if coeff != 0:
                        power = (i * j) % (self.field_size - 1)
                        term = self.galois_multiply(coeff, 
                                1 if power == 0 else self.primitive ** power)
                        sum ^= term
                if sum == 0:
                    error_positions.append(i)
            
            # Correct errors
            result = list(received_msg)
            for pos in error_positions:
                if pos < len(result):
                    result[pos] ^= 1
                    
            return result
            
        except Exception as e:
            self.helper.debug_print(f"Decoding error: {str(e)}")
            raise

    def galois_multiply(self, x, y):
        """Multiply two elements in GF(2)"""
        return (x & y)  # For GF(2), multiplication is AND operation

    def galois_power(self, exp):
        """Calculate power in GF(2)"""  
        return 1 if exp == 0 else self.primitive ** exp

    def galois_inverse(self, x):
        """Calculate multiplicative inverse in GF(2)"""
        if x == 0:
            raise ValueError("Zero has no inverse")
        return x  # In GF(2), every non-zero element is its own inverse


    def decode_classic(self, received_msg):
        """Use fixed BM algorithm for decoding"""
        try:
            # Calculate syndromes
            syndromes = self.calc_syndrome(received_msg)
            if np.all(syndromes == self.galois_field(0)):
                return received_msg
                
            # Get error locator polynomial
            error_locator = self.berlekamp_massey(syndromes)
            
            # Use existing modified_chien_search instead of chien_search
            error_positions = self.modified_chien_search(error_locator)
            
            # Correct errors
            result = list(received_msg)
            for pos in error_positions:
                if pos < len(result):
                    result[pos] ^= 1
                    
            return result
            
        except Exception as e:
            self.helper.debug_print(f"Decoding error: {str(e)}")
            raise

    def primitive_element_adjusted(self,index):
        factor_unity = index % self.p
        factor_primitive = divmod(index,self.p)[0]
        return  self.p_unity**factor_unity * self.primitive**factor_primitive
    
    def return_generalized_generator(self, index):
        if index >= self.p:
            raise ValueError("Generator polynominals can only be generated from 0 to p-1") 
        ## index is first value that qualifies i%self.p == index condition
        output_poly = gl.Poly([1,-self.primitive**index],field=self.galois_field)
        ## start looping after index
        for i in range(index+1,self.two_s):
            if i % self.p == index:
                output_poly *= gl.Poly([1,-self.primitive**i],field=self.galois_field)
        return output_poly
    
    def input_arr_to_matrix(self,_arr):
        output = []
        p = self.p
        intermediate_arr = []
        for i in range(0,len(_arr)):
            intermediate_arr.append(_arr[i])
            if (i+1) % p == 0:
                output.append(intermediate_arr[::-1])
                intermediate_arr = []
        return np.array(np.transpose(output))

    
    def calc_syndrome(self,recieved_codeword):
        
        output_syndromes = []
        m = len(recieved_codeword)// self.p
        self.helper.debug_print("recieved codeword before syndrome",self.galois_field(recieved_codeword),m)
        #m = len(recieved_codeword) // self.p
        divided_codeword =[]
        for l_index in range(0,self.p):
            syndrome = []
            for m_index in range(0,m):
                recieved_code_index = self.p * m_index + l_index
                syndrome.append(recieved_codeword[len(recieved_codeword)-recieved_code_index -1])
            divided_codeword.append(syndrome[::-1])
        
        fourier_transformed_syndromes = []
        self.helper.debug_print("divided codeword",divided_codeword)
        fft = self.helper.fft_on_matrix_multi(divided_codeword) if self.multi else self.helper.fft_on_matrix(divided_codeword)
        self.helper.debug_print("fft_decode",self.galois_field( fft))
        for F_i in range(0,self.p):
            fourier_transformed_syndromes.append(fft[F_i])

        for s_i in range(0,self.two_s):
            primitive_root = self.primitive ** s_i 
            fourier_poly = gl.Poly(fourier_transformed_syndromes[s_i % self.p],field=self.galois_field)
            self.helper.debug_print("dec fourier_poly", fourier_poly,primitive_root)
            output_syndromes.append(fourier_poly(primitive_root))
            self.helper.debug_print("s_i",s_i,s_i % self.p,primitive_root,fourier_poly(primitive_root))
        

        return output_syndromes
    
    # modified version for parallelization
    def modified_chien_search(self,error_locator_poly):
        """Direct Chien search implementation for GF(2)"""
        error_locations = []
        n = self.message_length + self.num_of_padded_zeros
        
        # Evaluate error locator polynomial at each position
        for i in range(n):
            # Calculate sum = error_locator_poly(α^i)
            sum = self.galois_field(0)
            for j, coeff in enumerate(error_locator_poly):
                if coeff != 0:  # Skip zero coefficients
                    # Calculate α^(i*j) safely
                    if i*j == 0:
                        power = 1
                    else:
                        power = self.primitive ** (i*j)
                    sum += coeff * power
                    
            # If sum is zero, i is an error location
            if sum == self.galois_field(0):
                error_locations.append(i)
        
        self.helper.debug_print("Error locations found:", error_locations)
        return error_locations

    def forney_algorithm(self, error_locator, syndromes, error_positions):
        """Calculate error values using Forney algorithm over GF(2)"""
        # For GF(2), error values are always 1 (bit flips)
        # No need for complex error value calculation
        error_values = [0] * self.message_length
        for pos in error_positions:
            if pos < self.message_length:
                error_values[pos] = 1
        return error_values

    def decode(self, received):
        """Main decoding procedure"""
        try:
            # Convert received to GF(2) array
            received_gf = self.galois_field(received)
            
            # Calculate syndromes
            syndromes = self.calc_syndrome(received_gf)
            if np.all(syndromes == self.galois_field(0)):
                return list(received)
            
            # Find error locator polynomial
            error_locator = self.berlekamp_massey(syndromes)
            
            # Find error positions
            error_positions = self.modified_chien_search(error_locator)
            
            # For GF(2), just flip bits at error positions
            result = list(received)
            for pos in error_positions:
                if pos < len(result):
                    result[pos] ^= 1
                    
            return result
            
        except Exception as e:
            self.helper.debug_print(f"Decoding error: {str(e)}")
            raise

    # def modified_forney(self,error_location_poly,error_evaluator_poly,error_loc):
    #     error_magnitudes = []
    #     for r in error_loc:
    #         a_i = self.primitive_element_adjusted(r)
            
    #         error_magnitudes.append( ( a_i * error_evaluator_poly(a_i**-1))/error_location_poly.derivative()(a_i**-1))
    #     return error_magnitudes
        

    def return_info_symbols(self,msg):
        return msg[self.num_of_padded_zeros:self.num_of_padded_zeros +self.payload_length]
    

    def calculate_t_matrix(self):
        matrix = []
        for i in range(self.d,self.p):
            row = [1]
            for j in range(1,self.d):
                row.append(self.p_unity**(j * i))
            matrix.append(row)
        return self.galois_field(np.matrix(matrix))
    
    def calculate_v_matrix(self):
        matrix = []
        self.helper.debug_print(self.p_unity,self.p_unity^2)
        matrix.append([1] * self.d)
        for i in range(1,self.d ):
            row = [1]
            for j in range(1,self.d):
                self.helper.debug_print(self.p_unity^(j*i),j*i)
                row.append(self.p_unity**(j*i))
            matrix.append(row)
        return self.galois_field(np.matrix(matrix))
    
    def return_transformation_values_for_f(self,T,V_inverse):
        output_matrix = []
        t_vinverse_matrices = np.dot(T,V_inverse )
        #generate identity matrix
        for i in range(0,self.d):
            identity_row = [0] * self.d
            identity_row[i] = 1
            output_matrix.append(identity_row)
        for row in t_vinverse_matrices:
            output_matrix.append(row)
        self.helper.debug_print(output_matrix)
        return self.galois_field(np.matrix(output_matrix))
    
    def append_parity_symbols(self,message,parity_symbols):
        output_msg = message
        output_symbols =[]
        index = 0
        parity_symbols = parity_symbols
        while len(parity_symbols) != 0:
            for parity_arry in parity_symbols:
                if len(parity_arry) == 0:
                    #since we are back to forth the first element is always the first to be empty
                    parity_symbols.pop(0)
                    break
                parity_symbol = parity_arry.pop(len(parity_arry)-1)
                output_symbols.append(parity_symbol)

           
        return output_msg + output_symbols[::-1]
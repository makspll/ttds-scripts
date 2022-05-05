from typing import List
import struct


def delta_encode(sequence: List[int]) -> List[int]:
    """ delta encodes a sequence of integers """
    assert len(sequence) > 0
    return [sequence[0]] + [next - prev for prev,next in zip(sequence[:-1],sequence[1:])]

def delta_decode(sequence: List[int]) -> List[int]:
    """ decodes a delta encoded sequence of integers """
    assert len(sequence) > 0

    last = sequence[0]
    new_sequence = []
    for e in sequence[1:]:
        last = last + e
        new_sequence.append(last) 

    return [sequence[0]] + new_sequence


def vbyte_encode(sequence : List[int]) -> bytearray:
    PAYLOAD_MASK = 0x0000007F
    CONT_BIT_MASK= 0x00000080
    little_endian = (struct.unpack('<I', struct.pack('=I', 1))[0] == 1)

    out = bytearray()

    for e in sequence:
        # don't support negative numbers 
        assert e > 0 

        payloads: List[int] = []
        cont_bits: List[int] = []

        while e > 0:
            cont_bit = CONT_BIT_MASK
            if e > PAYLOAD_MASK:
                cont_bit = 0

            cont_bits.append(cont_bit)
            payloads.append(PAYLOAD_MASK & e)
            e = e >> 7
    #1111101000
    #   1101000= 104
    #111       = 7
    #
        if little_endian: # should work on big endian systems too
            payloads.reverse()

        # we store bytes in big endian order for consistency with lecture slides
        for p,b in (zip(payloads,cont_bits)): 
            # the argument to to_bytes doesn't matter, since it's just one byte
            out.append((p|b).to_bytes(1,'little')[0])

    return out 

def vbyte_decode(sequence : bytearray):
    PAYLOAD_MASK = 0x0000007F
    CONT_BIT_MASK= 0x00000080

    out = []

    value_bytes = []
    for b in sequence:
        value_bytes.append(b & PAYLOAD_MASK)

        if b & CONT_BIT_MASK:
            value = concat_bytes(value_bytes)
            out.append(value)
            value_bytes = []

    return out
            
def concat_bytes(bytes : List[int]):
    val = 0
    for b in bytes:
        val = val << 7 
        val |= b & 0x0000007F
    return val 

def print_in_binary(bytes : bytearray):
    print(" ".join([f"{x:08b}" for x in bytes]))


def bitstring_to_bytes(s : str) -> bytearray:
    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')
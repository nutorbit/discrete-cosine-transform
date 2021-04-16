import bitstring
import numpy as np

from typing import Dict


def number2bit(val: np.array, bits: int = 64) -> str:
    """
    Convert array of number to binary

    Args:
        val: array of number
        bits: number of bits

    Returns:
        string of bits
    """

    return bitstring.BitArray(float=val, length=bits).bin


def bit2number(val: np.array, bits: int = 64) -> str:
    """
    Convert binary to array of number

    Args:
        val: string of bits
        bits: number of bits

    Returns:
        array of number
    """

    return bitstring.BitArray(bin=val, length=bits).float


def zigzag(n: int = 8) -> Dict:
    """
    ZigZag encoder & decoder

    Args:
        n: size of chunk

    Returns:
        dictionary of encoder and decoder
    """

    idx_zigzag = []
    for a in sorted((p % n + p // n, (p % n, p // n)[(p % n - p // n) % 2], p) for p in range(n * n)):
        idx_zigzag.append(a[2])

    idx2zig = np.array(list(zip(range(n * n), idx_zigzag)))
    inverse_idx_zigzag = np.array(sorted(idx2zig, key=lambda x: x[-1]))[:, 0]

    return {
        "encoder": idx2zig,
        "decoder": inverse_idx_zigzag
    }


def apply_zigzag(chunk: np.array, zigzag_encoder: np.array) -> np.array:
    """
    Apply ZigZag

    Args:
        chunk: chunk
        zigzag_encoder: encoder

    Returns:
        result
    """

    flat = chunk.flatten()[zigzag_encoder]
    return flat


def apply_inverse_zigzag(flat: np.array, zigzag_decoder: np.array) -> np.array:
    """
    Apply Inverse ZigZag

    Args:
        flat: chunk
        zigzag_decoder: decoder

    Returns:
        result
    """

    chunk = np.array(flat[zigzag_decoder]).reshape((8, 8))
    return chunk


def apply_encode(chunk: np.array, bits: int = 32) -> np.array:
    """
    Apply encoder

    Args:
        chunk: chunk
        bits: bit size

    Returns:
        string of code
    """

    d = zigzag()
    chunk = apply_zigzag(chunk, d["encoder"])
    lst_enc = []
    for el in chunk:
        lst_enc.append(number2bit(el, bits))
    enc_str = "".join(lst_enc)
    return enc_str


def apply_decode(code: np.array, bits: int = 32) -> np.array:
    """
    Apply decoder

    Args:
        code: chunk
        bits: bit size

    Returns:
        result
    """

    d = zigzag()
    lst = []
    for i in range(0, 2048, bits):
        lst.append(bit2number(code[i: i+bits]))
    lst = np.array(lst)
    lst = apply_inverse_zigzag(lst, d["decoder"])
    return lst

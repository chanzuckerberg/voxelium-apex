#!/usr/bin/env python

"""
Module for handling MRC-files
"""
import io
import os.path
from typing import Union

import numpy as np
import struct


data_types = {
    0: np.int8,
    1: np.int16,
    2: np.float32,
    3: np.complex64,
    4: np.complex64,
    6: np.uint16,
    12: np.float16,
    101: '4-bit'  # Special handling may be required for 4-bit data
}


class MRCHeader:
    def __init__(self, file=None):
        self.header_data = None
        self.extended_header = None
        if file is not None:
            self.read_from_file(file)
        else:
            self._init_default_header()

    def _init_default_header(self):
        # Initialize header with default values (0 or empty bytes)
        # The header structure: 10 integers, 12 floats, 3 integers, 3 floats, 3 integers,
        # 3 floats, 40 char, 80 char, 80 char, 80 char, integer, 80 char, 10 integers
        defaults = (0,) * 10 + (0.0,) * 12 + (0,) * 3 + (0.0,) * 3 + (0,) * 3 + \
                   (0.0,) * 3 + ('',) * 4 + (0,) + ('',) + (0,) * 10
        self.header_data = struct.pack('10i12f3i3f3i3f40s80s80s80si80s10i', *defaults)

    def _read_from_file(self, file):
        # Go to start of file
        file.seek(0, io.SEEK_SET)

        # Read the main header data
        self.header_data = file.read(1024)

        # Read the extended header data if it exists
        nsymbt = self.get_nsymtb()
        if nsymbt > 0:
            self.extended_header = file.read(nsymbt)

    def read_from_file(self, file):
        if isinstance(file, str):
            if not os.path.isfile(file):
                raise RuntimeError(f"Could not find MRC file: {file}")
            with open(file, 'rb') as f:
                self._read_from_file(f)
        else:
            self._read_from_file(file)

    def _write_to_file(self, file):
        # Go to start of file
        file.seek(0, io.SEEK_SET)

        # Write the main header data
        file.write(self.header_data)

        # Write the extended header data if it exists
        if self.extended_header:
            file.write(self.extended_header)

    def write_to_file(self, file):
        if isinstance(file, str):
            with open(file, 'wb') as f:
                self._write_to_file(f)
        else:
            self._write_to_file(file)

    def get_field(self, format, offset):
        return struct.unpack_from(format, self.header_data, offset)

    # Add methods for each field in the header
    def get_nx(self): return self.get_field('i', 0)[0]
    def get_ny(self): return self.get_field('i', 4)[0]
    def get_nz(self): return self.get_field('i', 8)[0]
    def get_mode(self): return self.get_field('i', 12)[0]
    def get_nxstart(self): return self.get_field('i', 16)[0]
    def get_nystart(self): return self.get_field('i', 20)[0]
    def get_nzstart(self): return self.get_field('i', 24)[0]
    def get_mx(self): return self.get_field('i', 28)[0]
    def get_my(self): return self.get_field('i', 32)[0]
    def get_mz(self): return self.get_field('i', 36)[0]
    def get_cella(self): return self.get_field('3f', 40)
    def get_cellb(self): return self.get_field('3f', 52)
    def get_mapc(self): return self.get_field('i', 64)[0]
    def get_mapr(self): return self.get_field('i', 68)[0]
    def get_maps(self): return self.get_field('i', 72)[0]
    def get_dmin(self): return self.get_field('f', 76)[0]
    def get_dmax(self): return self.get_field('f', 80)[0]
    def get_dmean(self): return self.get_field('f', 84)[0]
    def get_ispg(self): return self.get_field('i', 88)[0]
    def get_nsymtb(self): return self.get_field('i', 92)[0]
    def get_extra(self): return self.get_field('100s', 96)[0]
    def get_exttyp(self): return self.get_field('4s', 104)[0]
    def get_nversion(self): return self.get_field('i', 108)[0]
    def get_origin(self): return self.get_field('3f', 196)[0]
    def get_map(self): return self.get_field('4s', 208)[0]
    def get_machst(self): return self.get_field('4s', 212)[0]
    def get_rms(self): return self.get_field('f', 216)[0]
    def get_nlabl(self): return self.get_field('i', 220)[0]
    def get_label(self, n): return self.get_field('80s', 224 + 80 * n)[0].strip()

    def get_extended_header(self):
        return self.extended_header

    def get_data_size(self):
        xy = self.get_nx() * self.get_ny()
        z = self.get_nz()
        return xy * z if z > 0 else xy

    def get_data_type(self):
        mode = self.get_mode()
        return data_types.get(mode, np.float32)

    def get_data_start_position(self):
        # The start position of the data is the sum of the main header size (1024 bytes)
        # and the size of the extended header
        nsymbt = self.get_nsymtb()  # Size of the extended header
        return 1024 + nsymbt

    def __str__(self):
        return (
            f"nx: {self.get_nx()}\n"
            f"ny: {self.get_ny()}\n"
            f"nz: {self.get_nz()}\n"
            f"mode: {self.get_mode()} ({np.dtype(self.get_data_type()).name})\n"
            f"nxstart: {self.get_nxstart()}\n"
            f"nystart: {self.get_nystart()}\n"
            f"nzstart: {self.get_nzstart()}\n"
            f"mx: {self.get_mx()}\n"
            f"my: {self.get_my()}\n"
            f"mz: {self.get_mz()}\n"
            f"cella: {self.get_cella()}\n"
            f"cellb: {self.get_cellb()}\n"
            f"mapc: {self.get_mapc()}\n"
            f"mapr: {self.get_mapr()}\n"
            f"maps: {self.get_maps()}\n"
            f"dmin: {self.get_dmin()}\n"
            f"dmax: {self.get_dmax()}\n"
            f"dmean: {self.get_dmean()}\n"
            f"ispg: {self.get_ispg()}\n"
            f"nsymtb: {self.get_nsymtb()}\n"
            f"extra: {self.get_extra()}\n"
            f"exttyp: {self.get_exttyp()}\n"
            f"nversion: {self.get_nversion()}\n"
            f"origin: {self.get_origin()}\n"
            f"map: {self.get_map()}\n"
            f"machst: {self.get_machst()}\n"
            f"rms: {self.get_rms()}\n"
            f"nlabl: {self.get_nlabl()}"
        )

    def print(self):
        print(self)

    def set_field(self, format, offset, *values):
        struct.pack_into(format, self.header_data, offset, *values)

    # Set methods for each field in the header
    def set_nx(self, value): self.set_field('i', 0, value)
    def set_ny(self, value): self.set_field('i', 4, value)
    def set_nz(self, value): self.set_field('i', 8, value)
    def set_mode(self, value): self.set_field('i', 12, value)
    def set_nxstart(self, value): self.set_field('i', 16, value)
    def set_nystart(self, value): self.set_field('i', 20, value)
    def set_nzstart(self, value): self.set_field('i', 24, value)
    def set_mx(self, value): self.set_field('i', 28, value)
    def set_my(self, value): self.set_field('i', 32, value)
    def set_mz(self, value): self.set_field('i', 36, value)
    def set_cella(self, value): self.set_field('3f', 40, *value)
    def set_cellb(self, value): self.set_field('3f', 52, *value)
    def set_mapc(self, value): self.set_field('i', 64, value)
    def set_mapr(self, value): self.set_field('i', 68, value)
    def set_maps(self, value): self.set_field('i', 72, value)
    def set_dmin(self, value): self.set_field('f', 76, value)
    def set_dmax(self, value): self.set_field('f', 80, value)
    def set_dmean(self, value): self.set_field('f', 84, value)
    def set_ispg(self, value): self.set_field('i', 88, value)
    def set_extra(self, value): self.set_field('100s', 96, value)
    def set_exttyp(self, value): self.set_field('4s', 104, value)
    def set_nversion(self, value): self.set_field('i', 108, value)
    def set_origin(self, value): self.set_field('3f', 196, *value)
    def set_map(self, value): self.set_field('4s', 208, value)
    def set_machst(self, value): self.set_field('4s', 212, value)
    def set_rms(self, value): self.set_field('f', 216, value)
    def set_nlabl(self, value): self.set_field('i', 220, value)
    def set_label(self, n, value): self.set_field('80s', 224 + 80 * n, value)

    def set_extended_header(self, ext_header_data):
        # Convert the input data to bytes if it's not already
        if isinstance(ext_header_data, str):
            ext_header_data = ext_header_data.encode()

        # Update the extended header data
        self.extended_header = ext_header_data

        # Automatically set the NSYMBT field (size of the extended header)
        self.set_field('i', 92, len(ext_header_data))


class MRCReader:
    def __init__(self, file):
        self.header = None
        self.data = None

        if isinstance(file, str):
            if not os.path.isfile(file):
                raise RuntimeError(f"Could not find MRC file: {file}")
            with open(file, 'rb') as f:
                self._read_from_file(f)
        else:
            self._read_from_file(file)

    def _read_from_file(self, file):
        self.header = MRCHeader(file)

        data_position = self.header.get_data_start_position()
        file.seek(data_position, io.SEEK_SET)

        data_type = self.header.get_data_type()
        self.data = np.fromfile(file, dtype=data_type)

        x = self.header.get_nx()
        y = self.header.get_ny()
        z = self.header.get_nz()
        if z > 0:
            self.data.resize(x, y, z)
        else:
            self.data.resize(x, y)


if __name__ == "__main__":
    mrc_header = MRCHeader('/home/dari/Downloads/waving_spike/rec.mrc')
    print(mrc_header)

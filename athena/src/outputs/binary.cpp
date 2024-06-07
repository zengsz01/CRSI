//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file binary.cpp
//  \brief writes output data as a binary file. Temporary only support
//  particles.  Writes one file per Meshblock.

// C headers

// C++ headers
#include <cstdio>  // fwrite(), fclose(), fopen(), fnprintf(), snprintf()
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../particles/particles.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
//! \fn void BinOutput:::WriteOutputFile(Mesh *pm)
//  \brief writes OutputData to a binary file. Format is same as Athena-C-version.
//         Writes one file per MeshBlock

void BinOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin,
                                           bool flag) {
  // Output particle data if any.
  if (PARTICLES > 0) Particles::BinaryOutput(pm, output_params);

  // increment counters
  output_params.file_number++;
  output_params.next_time += output_params.dt;
  pin->SetInteger(output_params.block_name, "file_number",
                  output_params.file_number);
  pin->SetReal(output_params.block_name, "next_time", output_params.next_time);

  return;
}

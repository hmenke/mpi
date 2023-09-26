// Copyright (c) 2019-2024 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Thomas Hahn, Alexander Hampel, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Includes all relevant mpi headers.
 */

#pragma once

#include "./chunk.hpp"
#include "./communicator.hpp"
#include "./datatypes.hpp"
#include "./environment.hpp"
#include "./generic_communication.hpp"
#include "./lazy.hpp"
#include "./operators.hpp"
#include "./window.hpp"

namespace mpi {

#define MPI_TEST_MAIN                                                                                                                                \
  int main(int argc, char **argv) {                                                                                                                  \
    ::testing::InitGoogleTest(&argc, argv);                                                                                                          \
    if (mpi::has_env) {                                                                                                                              \
      mpi::environment env(argc, argv);                                                                                                              \
      std::cout << "MPI environment detected\n";                                                                                                     \
      return RUN_ALL_TESTS();                                                                                                                        \
    } else                                                                                                                                           \
      return RUN_ALL_TESTS();                                                                                                                        \
  }

} // namespace mpi

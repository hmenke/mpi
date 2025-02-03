// Copyright (c) 2020-2024 Simons Foundation
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
// Authors: Thomas Hahn, Nils Wentzell

#include <gtest/gtest.h>
#include <mpi/mpi.hpp>


TEST(MPI, CheckMPICall) {
  // test if check_mpi_call throws an exception
  try {
    mpi::check_mpi_call(MPI_SUCCESS - 1, "not_a_real_mpi_call");
  } catch (std::runtime_error const &e) { std::cout << e.what() << std::endl; }
}

MPI_TEST_MAIN;

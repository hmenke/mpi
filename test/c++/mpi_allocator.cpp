// Copyright (c) 2023 Simons Foundation
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
// Authors: Philipp Dumitrescu, Olivier Parcollet, Nils Wentzell

#include <mpi/mpi.hpp>
#include <mpi/allocator.hpp>
#include <mpi/vector.hpp>
#include <gtest/gtest.h>
#include <numeric>

TEST(MPI_Allocator, SharedAllocator) {
    mpi::shared_allocator<int> alloc;
    int *p = alloc.allocate(1);
    alloc.deallocate(p, 1);
}

TEST(MPI_Allocator, SharedAllocatorVector) {
    std::vector<int, mpi::shared_allocator<int>> v(128);
    auto shm = v.get_allocator().get_communicator();

    // Fill the vector in parallel
    v.get_allocator().get_window(v.data())->fence();
    auto slice = itertools::chunk_range(0, v.size(), shm.size(), shm.rank());
    for (auto i = slice.first; i < slice.second; ++i) {
        v.at(i) = i;
    }
    v.get_allocator().get_window(v.data())->fence();

    int const sum = std::accumulate(v.begin(), v.end(), int{0});
    EXPECT_EQ(sum, v.size() * (v.size() - 1) / 2);
}

MPI_TEST_MAIN;
